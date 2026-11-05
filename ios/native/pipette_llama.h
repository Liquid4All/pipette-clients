#ifndef PIPETTE_LLAMA_H
#define PIPETTE_LLAMA_H

// The boundary between the app and the llama.cpp builds it carries. Each build is a
// framework, `Frameworks/<pin>.framework` from `ios/build-llama.sh`, that exports one
// symbol: `pipette_llama_table`, the C calls the Swift engine makes, filled from that
// build's own headers (`pipette_llama_table.cpp`). A process loads one build and the app's
// forwarders (`Native/PipetteLlama.c`) call through its table, so Swift compiles once,
// against main's headers, and drives any build.
//
// A call is forwarded only when its prototype and every record it passes read the same in
// every build; the asserts hold those records and the constants Swift reads to main's
// values. Four `pipette_llama_*` calls stand in for the parameter records that differ
// between builds (`llama_model_params`, `llama_context_params`, `mtmd_context_params`,
// `mtmd_helper_init_opt`).

#include "llama.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "pipette_chat.h"
#include "pipette_vlspec.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define PIPETTE_LLAMA_ASSERT(cond) static_assert(cond, #cond)
#else
#define PIPETTE_LLAMA_ASSERT(cond) _Static_assert(cond, #cond)
#endif

// Records passed through unchanged, at main's layout.
PIPETTE_LLAMA_ASSERT(sizeof(llama_token) == 4 && sizeof(llama_pos) == 4 && sizeof(llama_seq_id) == 4);
PIPETTE_LLAMA_ASSERT(sizeof(struct llama_batch) == 56);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, n_tokens) == 0);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, token) == 8);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, embd) == 16);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, pos) == 24);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, n_seq_id) == 32);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, seq_id) == 40);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_batch, logits) == 48);
PIPETTE_LLAMA_ASSERT(sizeof(struct llama_chat_message) == 16);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_chat_message, role) == 0);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_chat_message, content) == 8);
PIPETTE_LLAMA_ASSERT(sizeof(llama_logit_bias) == 8);
PIPETTE_LLAMA_ASSERT(offsetof(llama_logit_bias, token) == 0);
PIPETTE_LLAMA_ASSERT(offsetof(llama_logit_bias, bias) == 4);
PIPETTE_LLAMA_ASSERT(sizeof(struct llama_sampler_chain_params) == 1);
PIPETTE_LLAMA_ASSERT(offsetof(struct llama_sampler_chain_params, no_perf) == 0);
PIPETTE_LLAMA_ASSERT(sizeof(mtmd_input_text) == 24);
PIPETTE_LLAMA_ASSERT(offsetof(mtmd_input_text, text) == 0);
PIPETTE_LLAMA_ASSERT(offsetof(mtmd_input_text, text_len) == 8);
PIPETTE_LLAMA_ASSERT(offsetof(mtmd_input_text, add_special) == 16);
PIPETTE_LLAMA_ASSERT(offsetof(mtmd_input_text, parse_special) == 17);
PIPETTE_LLAMA_ASSERT(sizeof(struct gguf_init_params) == 16);
PIPETTE_LLAMA_ASSERT(offsetof(struct gguf_init_params, no_alloc) == 0);
PIPETTE_LLAMA_ASSERT(offsetof(struct gguf_init_params, ctx) == 8);
PIPETTE_LLAMA_ASSERT(sizeof(struct ggml_init_params) == 24);
PIPETTE_LLAMA_ASSERT(offsetof(struct ggml_init_params, mem_size) == 0);
PIPETTE_LLAMA_ASSERT(offsetof(struct ggml_init_params, mem_buffer) == 8);
PIPETTE_LLAMA_ASSERT(offsetof(struct ggml_init_params, no_alloc) == 16);

// Constants Swift reads, at main's values.
PIPETTE_LLAMA_ASSERT(LLAMA_DEFAULT_SEED == 0xFFFFFFFF);
PIPETTE_LLAMA_ASSERT(LLAMA_LOAD_MODE_NONE == 0 && LLAMA_LOAD_MODE_MMAP == 1 && LLAMA_LOAD_MODE_MLOCK == 2);
PIPETTE_LLAMA_ASSERT(LLAMA_LOAD_MODE_MMAP_MLOCK == 3 && LLAMA_LOAD_MODE_DIRECT_IO == 4);
PIPETTE_LLAMA_ASSERT(MTMD_INPUT_CHUNK_TYPE_TEXT == 0 && MTMD_INPUT_CHUNK_TYPE_IMAGE == 1);
PIPETTE_LLAMA_ASSERT(MTMD_INPUT_CHUNK_TYPE_AUDIO == 2);
PIPETTE_LLAMA_ASSERT(GGUF_TYPE_UINT32 == 4 && GGUF_TYPE_BOOL == 7 && GGUF_TYPE_STRING == 8);
PIPETTE_LLAMA_ASSERT(GGML_TYPE_F32 == 0);

#ifdef __cplusplus
extern "C" {
#endif

/// `llama_model_default_params` with the two fields a cell sets.
struct llama_model * pipette_llama_load_model(const char * path, int32_t n_gpu_layers,
                                              enum llama_load_mode load_mode);

/// The context `LlamaCpp.initContext` asks for; `n_ctx` 0 keeps llama.cpp's default.
struct llama_context * pipette_llama_init_context(struct llama_model * model, uint32_t n_ctx,
                                                  uint32_t n_ubatch, int32_t n_threads,
                                                  bool swa_full);

/// `mtmd_context_params_default` with timings and mtmd's own warm-up off.
mtmd_context * pipette_llama_mtmd_init(const char * mmproj_path, const struct llama_model * model,
                                       bool use_gpu, int32_t n_threads);

/// `mtmd_helper_bitmap_init_from_file` with the default options; NULL when the file
/// does not read.
mtmd_bitmap * pipette_llama_mtmd_bitmap_from_file(mtmd_context * ctx, const char * path);

/// `mtmd_tokenize`, whose context parameter is const in some builds and not in others.
int32_t pipette_llama_mtmd_tokenize(const mtmd_context * ctx, mtmd_input_chunks * output,
                                    const mtmd_input_text * text,
                                    const mtmd_bitmap * const * bitmaps, size_t n_bitmaps);

// Every call the app makes into a build: R(return type, name, parameters, arguments) and
// V(name, parameters, arguments) for the void ones. The prototypes are main's, so the
// app's forwarders match the headers Swift imports, and a build whose prototype differs
// fails its fill.
#define PIPETTE_LLAMA_CALLS(R, V) \
    R(struct llama_model *, pipette_llama_load_model, (const char * path, int32_t n_gpu_layers, enum llama_load_mode load_mode), (path, n_gpu_layers, load_mode)) \
    R(struct llama_context *, pipette_llama_init_context, (struct llama_model * model, uint32_t n_ctx, uint32_t n_ubatch, int32_t n_threads, bool swa_full), (model, n_ctx, n_ubatch, n_threads, swa_full)) \
    R(mtmd_context *, pipette_llama_mtmd_init, (const char * mmproj_path, const struct llama_model * model, bool use_gpu, int32_t n_threads), (mmproj_path, model, use_gpu, n_threads)) \
    R(mtmd_bitmap *, pipette_llama_mtmd_bitmap_from_file, (mtmd_context * ctx, const char * path), (ctx, path)) \
    V(llama_backend_init, (void), ()) \
    V(llama_log_set, (ggml_log_callback log_callback, void * user_data), (log_callback, user_data)) \
    V(llama_model_free, (struct llama_model * model), (model)) \
    V(llama_free, (struct llama_context * ctx), (ctx)) \
    R(const struct llama_vocab *, llama_model_get_vocab, (const struct llama_model * model), (model)) \
    R(llama_memory_t, llama_get_memory, (const struct llama_context * ctx), (ctx)) \
    V(llama_memory_clear, (llama_memory_t mem, bool data), (mem, data)) \
    R(uint32_t, llama_n_batch, (const struct llama_context * ctx), (ctx)) \
    R(struct llama_batch, llama_batch_get_one, (llama_token * tokens, int32_t n_tokens), (tokens, n_tokens)) \
    R(int32_t, llama_decode, (struct llama_context * ctx, struct llama_batch batch), (ctx, batch)) \
    V(llama_synchronize, (struct llama_context * ctx), (ctx)) \
    R(int32_t, llama_tokenize, (const struct llama_vocab * vocab, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special), (vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)) \
    R(int32_t, llama_token_to_piece, (const struct llama_vocab * vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special), (vocab, token, buf, length, lstrip, special)) \
    R(int32_t, llama_vocab_n_tokens, (const struct llama_vocab * vocab), (vocab)) \
    R(bool, llama_vocab_get_add_bos, (const struct llama_vocab * vocab), (vocab)) \
    R(llama_token, llama_vocab_bos, (const struct llama_vocab * vocab), (vocab)) \
    R(bool, llama_vocab_is_eog, (const struct llama_vocab * vocab, llama_token token), (vocab, token)) \
    R(const char *, llama_model_chat_template, (const struct llama_model * model, const char * name), (model, name)) \
    R(int32_t, llama_chat_apply_template, (const char * tmpl, const struct llama_chat_message * chat, size_t n_msg, bool add_ass, char * buf, int32_t length), (tmpl, chat, n_msg, add_ass, buf, length)) \
    R(struct llama_sampler_chain_params, llama_sampler_chain_default_params, (void), ()) \
    R(struct llama_sampler *, llama_sampler_chain_init, (struct llama_sampler_chain_params params), (params)) \
    V(llama_sampler_chain_add, (struct llama_sampler * chain, struct llama_sampler * smpl), (chain, smpl)) \
    R(struct llama_sampler *, llama_sampler_init_greedy, (void), ()) \
    R(struct llama_sampler *, llama_sampler_init_logit_bias, (int32_t n_vocab, int32_t n_logit_bias, const llama_logit_bias * logit_bias), (n_vocab, n_logit_bias, logit_bias)) \
    R(struct llama_sampler *, llama_sampler_init_grammar, (const struct llama_vocab * vocab, const char * grammar_str, const char * grammar_root), (vocab, grammar_str, grammar_root)) \
    R(struct llama_sampler *, llama_sampler_init_top_k, (int32_t k), (k)) \
    R(struct llama_sampler *, llama_sampler_init_top_p, (float p, size_t min_keep), (p, min_keep)) \
    R(struct llama_sampler *, llama_sampler_init_min_p, (float p, size_t min_keep), (p, min_keep)) \
    R(struct llama_sampler *, llama_sampler_init_temp, (float t), (t)) \
    R(struct llama_sampler *, llama_sampler_init_dist, (uint32_t seed), (seed)) \
    R(llama_token, llama_sampler_sample, (struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx), (smpl, ctx, idx)) \
    V(llama_sampler_accept, (struct llama_sampler * smpl, llama_token token), (smpl, token)) \
    V(llama_sampler_reset, (struct llama_sampler * smpl), (smpl)) \
    V(llama_sampler_free, (struct llama_sampler * smpl), (smpl)) \
    V(mtmd_helper_log_set, (ggml_log_callback log_callback, void * user_data), (log_callback, user_data)) \
    V(mtmd_free, (mtmd_context * ctx), (ctx)) \
    R(bool, mtmd_support_vision, (const mtmd_context * ctx), (ctx)) \
    R(bool, mtmd_support_audio, (const mtmd_context * ctx), (ctx)) \
    R(int, mtmd_get_audio_sample_rate, (const mtmd_context * ctx), (ctx)) \
    R(const char *, mtmd_default_marker, (void), ()) \
    R(mtmd_bitmap *, mtmd_bitmap_init, (uint32_t nx, uint32_t ny, const unsigned char * data), (nx, ny, data)) \
    V(mtmd_bitmap_free, (mtmd_bitmap * bitmap), (bitmap)) \
    R(mtmd_input_chunks *, mtmd_input_chunks_init, (void), ()) \
    V(mtmd_input_chunks_free, (mtmd_input_chunks * chunks), (chunks)) \
    R(int32_t, pipette_llama_mtmd_tokenize, (const mtmd_context * ctx, mtmd_input_chunks * output, const mtmd_input_text * text, const mtmd_bitmap * const * bitmaps, size_t n_bitmaps), (ctx, output, text, bitmaps, n_bitmaps)) \
    R(size_t, mtmd_helper_get_n_tokens, (const mtmd_input_chunks * chunks), (chunks)) \
    R(size_t, mtmd_input_chunks_size, (const mtmd_input_chunks * chunks), (chunks)) \
    R(const mtmd_input_chunk *, mtmd_input_chunks_get, (const mtmd_input_chunks * chunks, size_t idx), (chunks, idx)) \
    R(enum mtmd_input_chunk_type, mtmd_input_chunk_get_type, (const mtmd_input_chunk * chunk), (chunk)) \
    R(size_t, mtmd_input_chunk_get_n_tokens, (const mtmd_input_chunk * chunk), (chunk)) \
    R(const llama_token *, mtmd_input_chunk_get_tokens_text, (const mtmd_input_chunk * chunk, size_t * n_tokens_output), (chunk, n_tokens_output)) \
    R(int32_t, mtmd_helper_eval_chunks, (mtmd_context * ctx, struct llama_context * lctx, const mtmd_input_chunks * chunks, llama_pos n_past, llama_seq_id seq_id, int32_t n_batch, bool logits_last, llama_pos * new_n_past), (ctx, lctx, chunks, n_past, seq_id, n_batch, logits_last, new_n_past)) \
    R(struct gguf_context *, gguf_init_from_file, (const char * fname, struct gguf_init_params params), (fname, params)) \
    V(gguf_free, (struct gguf_context * ctx), (ctx)) \
    R(int64_t, gguf_find_key, (const struct gguf_context * ctx, const char * key), (ctx, key)) \
    R(enum gguf_type, gguf_get_kv_type, (const struct gguf_context * ctx, int64_t key_id), (ctx, key_id)) \
    R(bool, gguf_get_val_bool, (const struct gguf_context * ctx, int64_t key_id), (ctx, key_id)) \
    R(const char *, gguf_get_val_str, (const struct gguf_context * ctx, int64_t key_id), (ctx, key_id)) \
    R(int64_t, gguf_find_tensor, (const struct gguf_context * ctx, const char * name), (ctx, name)) \
    R(struct gguf_context *, gguf_init_empty, (void), ()) \
    R(bool, gguf_write_to_file, (const struct gguf_context * ctx, const char * fname, bool only_meta), (ctx, fname, only_meta)) \
    V(gguf_set_val_bool, (struct gguf_context * ctx, const char * key, bool val), (ctx, key, val)) \
    V(gguf_set_val_str, (struct gguf_context * ctx, const char * key, const char * val), (ctx, key, val)) \
    V(gguf_set_val_u32, (struct gguf_context * ctx, const char * key, uint32_t val), (ctx, key, val)) \
    V(gguf_add_tensor, (struct gguf_context * ctx, const struct ggml_tensor * tensor), (ctx, tensor)) \
    R(struct ggml_context *, ggml_init, (struct ggml_init_params params), (params)) \
    V(ggml_free, (struct ggml_context * ctx), (ctx)) \
    R(struct ggml_tensor *, ggml_new_tensor_1d, (struct ggml_context * ctx, enum ggml_type type, int64_t ne0), (ctx, type, ne0)) \
    R(struct ggml_tensor *, ggml_set_name, (struct ggml_tensor * tensor, const char * name), (tensor, name)) \
    R(int32_t, pipette_chat_apply_jinja, (const struct llama_model * model, const char * tmpl_override, const char * const * roles, const char * const * contents, size_t n_messages, bool add_generation_prompt, bool enable_thinking, const char * const * kwarg_keys, const char * const * kwarg_values, size_t n_kwargs, char * buf, int32_t buf_len, char * err, size_t err_len), (model, tmpl_override, roles, contents, n_messages, add_generation_prompt, enable_thinking, kwarg_keys, kwarg_values, n_kwargs, buf, buf_len, err, err_len)) \
    R(pipette_vlspec *, pipette_vlspec_attach, (struct llama_model * model_tgt, struct llama_context * ctx_tgt, const struct pipette_vlspec_params * params, char * err, size_t err_len), (model_tgt, ctx_tgt, params, err, err_len)) \
    V(pipette_vlspec_free, (pipette_vlspec * spec), (spec)) \
    R(bool, pipette_vlspec_has_drafter, (const pipette_vlspec * spec), (spec)) \
    V(pipette_vlspec_begin, (pipette_vlspec * spec), (spec)) \
    R(int32_t, pipette_vlspec_generate, (pipette_vlspec * spec, const llama_token * prompt_text, size_t n_prompt_text, int32_t n_past, int32_t n_predict, llama_token * out_tokens, size_t out_cap, size_t * out_n, struct pipette_vlspec_turn_stats * stats, char * err, size_t err_len), (spec, prompt_text, n_prompt_text, n_past, n_predict, out_tokens, out_cap, out_n, stats, err, err_len))

/// Increment this value when a field of the table changes. The app refuses a framework
/// that another revision of this header built.
#define PIPETTE_LLAMA_TABLE_VERSION 1

struct pipette_llama_table {
    uint32_t version;
    uint32_t size;
#define PIPETTE_LLAMA_FIELD_R(ret, name, params, args) ret (*name) params;
#define PIPETTE_LLAMA_FIELD_V(name, params, args) void (*name) params;
    PIPETTE_LLAMA_CALLS(PIPETTE_LLAMA_FIELD_R, PIPETTE_LLAMA_FIELD_V)
#undef PIPETTE_LLAMA_FIELD_R
#undef PIPETTE_LLAMA_FIELD_V
};

/// The one symbol a build exports.
#define PIPETTE_LLAMA_TABLE_SYMBOL "pipette_llama_table"
typedef const struct pipette_llama_table * (*pipette_llama_table_fn)(void);

/// `vendor/llama.cpp`'s framework, the build a process runs when nothing chose another.
#define PIPETTE_LLAMA_MAIN_PIN "llamacpp"

/// The longest pin name, in bytes. ios/build-llama.sh refuses a longer variant.
#define PIPETTE_LLAMA_PIN_MAX 63

// The app's side (`Native/PipetteLlama.c`); a build defines none of these.

/// Loads `Frameworks/<pin>.framework` and makes it this process's build. Loading the
/// build already loaded returns its table again; any other is refused, as an image
/// cannot be unloaded. NULL with a reason in `err` on failure.
const struct pipette_llama_table * pipette_llama_load(const char * pin, char * err, size_t err_len);

/// Loads `pin` without making it this process's build, to prove it loads. Only for a
/// process that runs no cell (`runtimes list`): the image stays mapped.
const struct pipette_llama_table * pipette_llama_probe(const char * pin, char * err, size_t err_len);

/// The pin `pipette_llama_load` loaded, or NULL before one.
const char * pipette_llama_loaded_pin(void);

#ifdef __cplusplus
}
#endif

#endif /* PIPETTE_LLAMA_H */
