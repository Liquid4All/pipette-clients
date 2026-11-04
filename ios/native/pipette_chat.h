#ifndef PIPETTE_CHAT_H
#define PIPETTE_CHAT_H

// A C surface over llama.cpp's Jinja chat-template renderer (`common/chat.cpp`),
// for templates the legacy `llama_chat_apply_template` cannot detect by their
// markers (it returns -1, e.g. for Gemma-4's `<|turn>`). That renderer is C++ with
// no C API, and a Swift reimplementation would make the phone a second
// implementation of the prompt the desktop measures, so the render stays upstream's.
//
// A fallback only: `LlamaCpp.renderChatTemplate` tries the legacy path first, so a
// template the legacy path renders keeps the prompt it had.
//
// Compiled into each llama.cpp framework by `ios/build-llama.sh` rather than into
// the app target, so the app needs no C++ toolchain settings and no llama.cpp
// private headers on its search path.

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Render `n_messages` messages as `llama-server` renders a `/v1/chat/completions`
/// request with `--jinja` (its default), with the server's default `reasoning_format`.
///
/// `enable_thinking` false is `--reasoning off`; true is `--reasoning` unset: thinking on
/// exactly when the template supports it.
///
/// Each `kwarg_values` entry is JSON text, as the server holds `--chat-template-kwargs`.
/// Unlike the server, an `enable_thinking` kwarg does not override `enable_thinking`.
///
/// `model` supplies the BOS/EOS strings and whether the vocabulary adds a BOS. When it
/// does, the template's leading BOS is stripped as on the server, so the caller
/// tokenizes the result with `add_special` on. `model` may be NULL when
/// `tmpl_override` is given (tests): BOS/EOS are then empty and nothing is stripped.
///
/// `tmpl_override` NULL or empty reads the model's own template.
///
/// Returns the prompt's length in bytes, not counting a terminator, as
/// `llama_chat_apply_template` does. At most `buf_len` bytes are written, NUL-terminated
/// when there is room, so a caller whose buffer was too small calls again with the
/// returned size (plus one for a terminator).
///
/// Returns -1 with a reason in `err` on failure. No C++ exception escapes: a template
/// that cannot be rendered fails its cell rather than the app.
int32_t pipette_chat_apply_jinja(const struct llama_model * model,
                                 const char * tmpl_override,
                                 const char * const * roles,
                                 const char * const * contents,
                                 size_t n_messages,
                                 bool add_generation_prompt,
                                 bool enable_thinking,
                                 const char * const * kwarg_keys,
                                 const char * const * kwarg_values,
                                 size_t n_kwargs,
                                 char * buf, int32_t buf_len,
                                 char * err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* PIPETTE_CHAT_H */
