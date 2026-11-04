// The table one llama.cpp build exports (see pipette_llama.h). `ios/build-llama.sh` compiles
// it against that build's headers and exports only `pipette_llama_table`.

#include "pipette_llama.h"

#include <algorithm>

extern "C" {

struct llama_model * pipette_llama_load_model(const char * path, int32_t n_gpu_layers,
                                              enum llama_load_mode load_mode) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.load_mode = load_mode;
    return llama_model_load_from_file(path, params);
}

struct llama_context * pipette_llama_init_context(struct llama_model * model, uint32_t n_ctx,
                                                  uint32_t n_ubatch, int32_t n_threads,
                                                  bool swa_full) {
    llama_context_params params = llama_context_default_params();
    if (n_ctx > 0) {
        params.n_ctx = n_ctx;
    }
    const uint32_t ubatch = n_ubatch > 0 ? n_ubatch : 512;
    params.n_ubatch = std::min(ubatch, params.n_ctx);
    params.n_batch = std::min(std::max(2048u, ubatch), params.n_ctx);
    params.n_seq_max = 1;
    params.no_perf = true;
    // Never inherited: the library default (on) differs from the stock tools' (off).
    params.swa_full = swa_full;
    // Both, as `llama-bench` sets them: the cell names one thread count, and a prefill
    // measured with a different one than decode would not be the cell it asked for.
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    return llama_init_from_model(model, params);
}

mtmd_context * pipette_llama_mtmd_init(const char * mmproj_path, const struct llama_model * model,
                                       bool use_gpu, int32_t n_threads) {
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = use_gpu;
    params.print_timings = false;
    params.n_threads = n_threads;
    params.warmup = false;
    return mtmd_init_from_file(mmproj_path, model, params);
}

mtmd_bitmap * pipette_llama_mtmd_bitmap_from_file(mtmd_context * ctx, const char * path) {
    return mtmd_helper_bitmap_init_from_file(ctx, path, false, mtmd_helper_init_opt_default()).bitmap;
}

int32_t pipette_llama_mtmd_tokenize(const mtmd_context * ctx, mtmd_input_chunks * output,
                                    const mtmd_input_text * text,
                                    const mtmd_bitmap * const * bitmaps, size_t n_bitmaps) {
    // The prismml fork takes the context non-const.
    return mtmd_tokenize(const_cast<mtmd_context *>(ctx), output, text, bitmaps, n_bitmaps);
}

__attribute__((visibility("default")))
const struct pipette_llama_table * pipette_llama_table(void) {
    static const struct pipette_llama_table table = [] {
        struct pipette_llama_table t{};
        t.version = PIPETTE_LLAMA_TABLE_VERSION;
        t.size = sizeof(struct pipette_llama_table);
        // Typed assignments: a build whose prototype differs from main's fails here.
#define PIPETTE_LLAMA_FILL_R(ret, name, params, args) t.name = &name;
#define PIPETTE_LLAMA_FILL_V(name, params, args) t.name = &name;
        PIPETTE_LLAMA_CALLS(PIPETTE_LLAMA_FILL_R, PIPETTE_LLAMA_FILL_V)
#undef PIPETTE_LLAMA_FILL_R
#undef PIPETTE_LLAMA_FILL_V
        return t;
    }();
    return &table;
}

}  // extern "C"
