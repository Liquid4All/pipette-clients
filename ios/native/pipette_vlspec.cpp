// Implementation of pipette_vlspec.h. The loop follows
// `examples/speculative-simple/speculative-simple.cpp` closely on purpose, so an
// upstream change to that loop can be diffed against this one.
//
// Failures are returned, never fatal: this runs inside the app process, and a
// sample that cannot be evaluated fails its cell rather than the app.

#include "pipette_vlspec.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr llama_seq_id SEQ_ID = 0;

// build-llama.sh defines PIPETTE_VLSPEC_LLAMA_BATCH when common/common.h has no
// common_batch (upstream added it in b11236), as in a vendor/llamacpp-<name> fork.
#if PIPETTE_VLSPEC_LLAMA_BATCH
struct vlspec_batch {
    llama_batch batch;
    explicit vlspec_batch(llama_context * ctx) : batch(llama_batch_init(llama_n_batch(ctx), 0, 1)) {}
    ~vlspec_batch() { llama_batch_free(batch); }
    vlspec_batch(const vlspec_batch &) = delete;
    vlspec_batch & operator=(const vlspec_batch &) = delete;
    void clear() { common_batch_clear(batch); }
    void add(llama_token id, llama_pos pos, llama_seq_id seq, bool logits) {
        common_batch_add(batch, id, pos, { seq }, logits);
    }
    operator const llama_batch &() const { return batch; }
};

int32_t decode(llama_context * ctx, vlspec_batch & b) {
    return llama_decode(ctx, b.batch);
}
#else
using vlspec_batch = common_batch;

int32_t decode(llama_context * ctx, vlspec_batch & b) {
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get());
}
#endif

void set_err(char * err, size_t err_len, const std::string & message) {
    if (err == nullptr || err_len == 0) {
        return;
    }
    std::snprintf(err, err_len, "%s", message.c_str());
}

} // namespace

struct pipette_vlspec {
    // Borrowed: the app owns the model and the context and frees them after this.
    llama_model *   model_tgt = nullptr;
    llama_context * ctx_tgt   = nullptr;
    const llama_vocab * vocab = nullptr;

    common_params params;

    // `init` owns the draft model and its context; `spec` is the speculator over them.
    common_speculative_init_result_ptr init;
    common_speculative_ptr spec;
    llama_context * ctx_dft = nullptr;

    common_sampler_ptr smpl;

    ~pipette_vlspec() {
        // The speculator holds both contexts, so it goes before the draft context
        // its `init` owns.
        spec.reset();
        smpl.reset();
        init.reset();
    }
};

pipette_vlspec * pipette_vlspec_attach(llama_model * model_tgt,
                                       llama_context * ctx_tgt,
                                       const struct pipette_vlspec_params * p,
                                       char * err, size_t err_len) {
    if (model_tgt == nullptr || ctx_tgt == nullptr || p == nullptr) {
        set_err(err, err_len, "pipette_vlspec_attach: model, context and params are required");
        return nullptr;
    }
    if (p->drafter == PIPETTE_VLSPEC_DRAFTER_SEPARATE && p->draft_path == nullptr) {
        set_err(err, err_len, "pipette_vlspec_attach: a separate drafter needs draft_path");
        return nullptr;
    }

    auto spec = std::make_unique<pipette_vlspec>();
    spec->model_tgt = model_tgt;
    spec->ctx_tgt   = ctx_tgt;
    spec->vocab     = llama_model_get_vocab(model_tgt);

    common_params & params = spec->params;
    // Read off the live context so the draft context cannot disagree with the
    // target about positions or batch size.
    params.n_ctx    = (int32_t) llama_n_ctx(ctx_tgt);
    params.n_batch  = (int32_t) llama_n_batch(ctx_tgt);
    params.n_ubatch = (int32_t) llama_n_ubatch(ctx_tgt);
    params.n_gpu_layers = p->n_gpu_layers;
    params.cpuparams.n_threads       = p->n_threads;
    params.cpuparams_batch.n_threads = p->n_threads;
    // Greedy unless pinned: the greedy-identity check only holds between two
    // greedy traversals. A pinned temperature goes through the server's default chain.
    if (p->temperature > 0.0f) {
        params.sampling.temp = p->temperature;
    } else {
        params.sampling.temp     = 0.0f;
        params.sampling.top_k    = 1;
        params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    }

    const bool speculating = p->drafter != PIPETTE_VLSPEC_DRAFTER_NONE;
    if (speculating) {
        params.speculative.draft.n_max = p->n_draft_max;
        params.speculative.draft.n_min = 0;
        params.speculative.draft.n_gpu_layers = p->n_gpu_layers;
        params.speculative.draft.cpuparams.n_threads       = p->n_threads;
        params.speculative.draft.cpuparams_batch.n_threads = p->n_threads;
        switch (p->drafter) {
            case PIPETTE_VLSPEC_DRAFTER_SEPARATE:
                params.speculative.draft.mparams.path = p->draft_path;
                params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
                break;
            case PIPETTE_VLSPEC_DRAFTER_MERGED:
                params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
                break;
            default:
                set_err(err, err_len, "pipette_vlspec_attach: unknown drafter arrangement");
                return nullptr;
        }

        common_params params_dft = common_base_params_to_speculative(params);
        spec->init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        spec->ctx_dft = spec->init->context();
        if (spec->ctx_dft == nullptr) {
            // The merged case lands here when the weights have no MTP head (a DSpark or
            // DFlash head, which upstream drives only from a separate draft model).
            // Naming it beats reporting a speedup of one.
            set_err(err, err_len,
                    p->drafter == PIPETTE_VLSPEC_DRAFTER_MERGED
                        ? "this build cannot attach a merged drafter to these weights: "
                          "llama.cpp builds a draft context from a separate draft model or "
                          "from a multi-token-prediction head, and this model offers "
                          "neither. A model whose drafter ships as its own gguf (a `draft` "
                          "file in the model spec) runs here"
                        : "failed to load the draft model");
            return nullptr;
        }
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec->ctx_dft;
        spec->spec.reset(common_speculative_init(params.speculative, 1));
        if (!spec->spec) {
            set_err(err, err_len, "failed to initialize the speculator");
            return nullptr;
        }
    }

    // Not the app's sampler: `common_sampler_sample_and_accept_n` needs a
    // `common_sampler`. The baseline arm uses it too, so the arms differ only by the
    // drafter.
    spec->smpl.reset(common_sampler_init(model_tgt, params.sampling));
    if (!spec->smpl) {
        set_err(err, err_len, "failed to initialize the sampler");
        return nullptr;
    }
    return spec.release();
}

void pipette_vlspec_free(pipette_vlspec * spec) {
    delete spec;
}

bool pipette_vlspec_has_drafter(const pipette_vlspec * spec) {
    return spec != nullptr && spec->spec != nullptr;
}

void pipette_vlspec_begin(pipette_vlspec * spec) {
    if (spec == nullptr) {
        return;
    }
    if (spec->smpl) {
        common_sampler_reset(spec->smpl.get());
    }
    if (spec->ctx_dft != nullptr) {
        llama_memory_clear(llama_get_memory(spec->ctx_dft), true);
    }
}

int32_t pipette_vlspec_generate(pipette_vlspec * spec,
                                const llama_token * prompt_text, size_t n_prompt_text,
                                int32_t n_past, int32_t n_predict,
                                llama_token * out_tokens, size_t out_cap, size_t * out_n,
                                struct pipette_vlspec_turn_stats * stats,
                                char * err, size_t err_len) {
    if (spec == nullptr || stats == nullptr || out_tokens == nullptr) {
        set_err(err, err_len, "pipette_vlspec_generate: missing session, stats or output");
        return 1;
    }
    *stats = {};
    size_t n_out = 0;
    // Committed tokens, which is `n_out` plus the stop token the reply omits.
    int32_t n_generated = 0;

    llama_context * ctx = spec->ctx_tgt;
    common_sampler * smpl = spec->smpl.get();
    const int32_t n_ctx = (int32_t) llama_n_ctx(ctx);

    // The speculator's prompt; each committed token extends it.
    llama_tokens prompt(prompt_text, prompt_text + n_prompt_text);
    prompt.reserve(n_prompt_text + (size_t) n_predict + 1);

    const int64_t t_start = ggml_time_us();

    // Sampled from the prefill's last logits (the caller passes `logits_last`).
    llama_token id_last = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, id_last, true);
    stats->n_target_forwards += 1;
    stats->first_token_ms = (double) (ggml_time_us() - t_start) / 1000.0;

    int32_t stop = PIPETTE_VLSPEC_STOP_LIMIT;
    n_generated += 1;
    if (llama_vocab_is_eog(spec->vocab, id_last)) {
        stop = PIPETTE_VLSPEC_STOP_EOG;
    } else {
        if (n_out < out_cap) {
            out_tokens[n_out] = id_last;
        }
        n_out += 1;

        if (spec->spec) {
            common_speculative_begin(spec->spec.get(), SEQ_ID, prompt);
        }

        vlspec_batch batch(ctx);
        llama_tokens draft;

        while ((int32_t) n_out < n_predict) {
            if (n_past + 1 + (int32_t) draft.size() >= n_ctx) {
                stop = PIPETTE_VLSPEC_STOP_CONTEXT;
                break;
            }

            // A drafter may decline (too little context, a cache miss) and return
            // nothing, making the step a single-token decode; hence `n_drafted`
            // counts what was actually proposed.
            draft.clear();
            if (spec->spec) {
                common_speculative_get_draft_params(spec->spec.get(), SEQ_ID) = {
                    /* .drafting = */ true,
                    /* .n_max    = */ -1,
                    /* .pos0     = */ n_past,
                    /* .id_last  = */ id_last,
                    /* .prompt   = */ &prompt,
                    /* .result   = */ &draft,
                };
                common_speculative_draft(spec->spec.get());
            }

            // Logits on every position, so one forward pass verifies the whole block.
            batch.clear();
            batch.add(id_last, n_past, SEQ_ID, true);
            for (size_t i = 0; i < draft.size(); ++i) {
                batch.add(draft[i], n_past + 1 + (llama_pos) i, SEQ_ID, true);
            }
            if (decode(ctx, batch) != 0) {
                set_err(err, err_len, "the target model failed to decode a draft block");
                return 1;
            }
            // The prompt's batches are mtmd's and never reach the drafter, the same
            // view the server's speculator has of a multimodal prefill.
            if (spec->spec && !common_speculative_process(spec->spec.get(), batch)) {
                set_err(err, err_len, "the drafter rejected the target's batch");
                return 1;
            }
            stats->n_target_forwards += 1;
            stats->n_drafted += (int32_t) draft.size();

            // The first disagreement ends the block, and from b11259 so does a stop
            // token before the last draft. `ids` is the accepted drafts plus one
            // token the target chose itself, so it is never empty.
            const llama_tokens ids = common_sampler_sample_and_accept_n(smpl, ctx, draft);
            const int32_t n_accepted = (int32_t) ids.size() - 1;
            if (spec->spec) {
                common_speculative_accept(spec->spec.get(), SEQ_ID, n_accepted);
            }

            prompt.push_back(id_last);
            n_past += 1;
            for (size_t i = 0; i < ids.size(); ++i) {
                id_last = ids[i];
                // The target's own token follows a stop token only when that was
                // the last draft; the stop below leaves it uncommitted.
                if ((int32_t) i < n_accepted) {
                    stats->n_accepted += 1;
                }
                // The stop token is a committed token, as `llama-server` counts
                // it in `predicted_n`, but it is not part of the reply.
                n_generated += 1;
                if (llama_vocab_is_eog(spec->vocab, id_last)) {
                    stop = PIPETTE_VLSPEC_STOP_EOG;
                    break;
                }
                if (n_out < out_cap) {
                    out_tokens[n_out] = id_last;
                }
                n_out += 1;
                if (i + 1 < ids.size()) {
                    prompt.push_back(id_last);
                    n_past += 1;
                }
            }
            if (stop == PIPETTE_VLSPEC_STOP_EOG) {
                break;
            }

            // Drop the rejected tail of the draft from both caches.
            llama_memory_seq_rm(llama_get_memory(ctx), SEQ_ID, n_past, -1);
            if (spec->ctx_dft != nullptr) {
                llama_memory_seq_rm(llama_get_memory(spec->ctx_dft), SEQ_ID, n_past, -1);
            }
        }
    }

    // The committed reply has to reach the context before the next turn's prompt
    // is appended to it, and the loop above leaves its last token uncommitted.
    if (stop != PIPETTE_VLSPEC_STOP_CONTEXT && n_out > 0 && n_past + 1 < n_ctx) {
        vlspec_batch one(ctx);
        one.add(id_last, n_past, SEQ_ID, false);
        if (decode(ctx, one) != 0) {
            set_err(err, err_len, "the target model failed to commit the turn's last token");
            return 1;
        }
        n_past += 1;
    }

    llama_synchronize(ctx);
    stats->predicted_ms = (double) (ggml_time_us() - t_start) / 1000.0;
    stats->n_generated = n_generated;
    stats->n_past = n_past;
    stats->stop = stop;
    if (out_n != nullptr) {
        *out_n = n_out;
    }
    // A truncated reply cannot be compared against the other arm's.
    if (n_out > out_cap) {
        set_err(err, err_len, "the turn generated more tokens than the caller's buffer holds");
        return 1;
    }
    return 0;
}
