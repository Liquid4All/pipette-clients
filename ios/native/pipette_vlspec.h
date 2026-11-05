#ifndef PIPETTE_VLSPEC_H
#define PIPETTE_VLSPEC_H

// A C surface over llama.cpp's speculative decoding (`common/speculative.{h,cpp}`
// and `common_sampler_sample_and_accept_n`), for the iOS app's `vl_spec_decode`
// benchmark. That loop is C++ with no C API, and a Swift rewrite would make this
// client a second implementation of the procedure whose numbers the benchmark
// compares against the CLI's, so the loop stays upstream's.
//
// It owns one drafter attached to a context somebody else built, and the generation
// loop over it. Model load, mtmd, the chat template, the corpus traversal, the gate
// and the greedy-identity comparison are Swift's (`LlamaVisionChat.swift`,
// `Benchmarks/Turn.swift`, `Ops/VlSpecDecode.swift`).
//
// Compiled into each llama.cpp framework by `ios/build-llama.sh` rather than into
// the app target, so the app needs no C++ toolchain settings and no llama.cpp
// private headers on its search path. The Android counterpart is
// `android/native/llama_shim.cpp`.

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pipette_vlspec;
typedef struct pipette_vlspec pipette_vlspec;

/// Derived from the model spec, not a flag: a `gguf_vision_speculative` naming a
/// `draft` file is `SEPARATE`, one without is `MERGED`.
enum pipette_vlspec_drafter {
    /// The baseline arm runs through here too, so both arms share one loop.
    PIPETTE_VLSPEC_DRAFTER_NONE = 0,
    /// llama.cpp's `--spec-draft-model`, speculative type `draft-simple`.
    PIPETTE_VLSPEC_DRAFTER_SEPARATE = 1,
    /// Inside the target weights, reached as a multi-token-prediction context:
    /// llama.cpp's `--spec-type draft-mtp`. A merged DSpark or DFlash head is not an
    /// MTP one and upstream cannot build a draft context for it, so
    /// `pipette_vlspec_attach` refuses it (see docs/methodology/vl-spec-decode.md).
    PIPETTE_VLSPEC_DRAFTER_MERGED = 2,
};

/// The drafter's settings; the target is already loaded.
struct pipette_vlspec_params {
    int32_t      drafter;       // enum pipette_vlspec_drafter
    const char * draft_path;    // drafter weights; NULL unless drafter == SEPARATE
    int32_t      n_draft_max;   // k, the benchmark's parameter_draft_length
    int32_t      n_gpu_layers;  // inherited from the cell, as the target's is
    int32_t      n_threads;
    float        temperature;   // the benchmark's parameter_temperature; 0 is greedy
};

/// A turn that hit the context window is a measurement failure, not a short answer.
enum pipette_vlspec_stop {
    PIPETTE_VLSPEC_STOP_EOG = 0,
    PIPETTE_VLSPEC_STOP_LIMIT = 1,     // the turn's token budget
    PIPETTE_VLSPEC_STOP_CONTEXT = 2,
};

struct pipette_vlspec_turn_stats {
    /// Tokens committed, the stop token included as `llama-server`'s
    /// `predicted_n` counts it; the reply omits it.
    int32_t n_generated;
    /// Target forward passes: the prefill's, then one per verified block. Counted,
    /// not derived: a drafter proposes past the stop token, so
    /// `generated - accepted` is no forward count.
    int32_t n_target_forwards;
    int32_t n_drafted;          // tokens the drafter proposed
    /// Drafts accepted **and committed**: one the stop token discards is not counted.
    int32_t n_accepted;
    /// The KV position after the turn; the conversation's next turn extends from it.
    int32_t n_past;
    /// Generation only; the caller times its own prefill.
    double  predicted_ms;
    /// From the call; the caller adds its prefill to get the turn's TTFT.
    double  first_token_ms;
    int32_t stop;               // enum pipette_vlspec_stop
};

/// Returns NULL with a reason in `err` on failure, including a merged drafter the
/// vendored llama.cpp cannot reach: a run that silently dropped speculation would
/// submit a speedup of 1.0 and look like a finding. `DRAFTER_NONE` still returns a
/// handle, for the baseline arm.
pipette_vlspec * pipette_vlspec_attach(struct llama_model * model_tgt,
                                       struct llama_context * ctx_tgt,
                                       const struct pipette_vlspec_params * params,
                                       char * err, size_t err_len);

void pipette_vlspec_free(pipette_vlspec * spec);

/// The traversal asserts this on the speculative arm: an arm that never drafted
/// looks, from outside, like a model that gains nothing from a drafter.
bool pipette_vlspec_has_drafter(const pipette_vlspec * spec);

/// Once per corpus sample (one conversation). Clears the drafter's and the sampler's
/// state; the caller owns the context and clears the target's KV cache.
void pipette_vlspec_begin(pipette_vlspec * spec);

/// Generate one reply, extending the sequence the caller has already prefilled.
///
/// `prompt_text` is every *text* token in the context so far, media positions
/// omitted, as upstream's server hands the speculator
/// (`server_tokens::get_text_tokens`). `n_past` is the KV position the prefill left.
///
/// Returns non-zero with a reason in `err` on failure, including a reply longer
/// than `out_cap`.
int32_t pipette_vlspec_generate(pipette_vlspec * spec,
                                const llama_token * prompt_text, size_t n_prompt_text,
                                int32_t n_past, int32_t n_predict,
                                llama_token * out_tokens, size_t out_cap, size_t * out_n,
                                struct pipette_vlspec_turn_stats * stats,
                                char * err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* PIPETTE_VLSPEC_H */
