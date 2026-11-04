// Implementation of pipette_chat.h. The inputs mirror what `llama-server` builds
// for a messages-only chat request (`oaicompat_chat_params_parse` plus the
// `enable_thinking` the server resolves at startup). Only symbols every vendored
// llama.cpp declares identically in `common/chat.h` are used, so one source builds
// against each `vendor/llamacpp-<name>` variant without a probe; the rendered
// prompt's type is the exception, which `prompt_text` reads either way.
//
// Failures are returned, never fatal: this runs inside the app process.

#include "pipette_chat.h"

#include "chat.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace {

void set_err(char * err, size_t err_len, const char * message) {
    if (err == nullptr || err_len == 0) {
        return;
    }
    std::snprintf(err, err_len, "%s", message);
}

// `common_chat_params::prompt` is a std::string before upstream #30389 (b11737 has
// it) and a `common_chat_prompt` of parts after it; `str()` joins the parts.
template <typename Prompt> std::string prompt_text(const Prompt & prompt) {
    if constexpr (std::is_same_v<Prompt, std::string>) {
        return prompt;
    } else {
        return prompt.str();
    }
}

} // namespace

int32_t pipette_chat_apply_jinja(const llama_model * model,
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
                                 char * err, size_t err_len) {
    set_err(err, err_len, "");
    if (n_messages > 0 && (roles == nullptr || contents == nullptr)) {
        set_err(err, err_len, "pipette_chat_apply_jinja: roles and contents are required");
        return -1;
    }
    if (n_kwargs > 0 && (kwarg_keys == nullptr || kwarg_values == nullptr)) {
        set_err(err, err_len, "pipette_chat_apply_jinja: kwarg keys and values are required");
        return -1;
    }
    for (size_t i = 0; i < n_kwargs; ++i) {
        if (kwarg_keys[i] == nullptr || kwarg_values[i] == nullptr) {
            set_err(err, err_len, "pipette_chat_apply_jinja: a kwarg key or value is NULL");
            return -1;
        }
    }
    const bool has_override = tmpl_override != nullptr && tmpl_override[0] != '\0';
    // `common_chat_templates_init` would GGML_ASSERT (abort, not throw) here.
    if (!has_override && model == nullptr) {
        set_err(err, err_len, "pipette_chat_apply_jinja: a model or a template is required");
        return -1;
    }

    // `common_chat_templates_init` rethrows a parse error sliced to `std::exception`,
    // whose `what()` says nothing, so the failing step is the useful context.
    const char * stage = "parsing the template";
    try {
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(model, has_override ? std::string(tmpl_override) : std::string());
        stage = "rendering the template";

        common_chat_templates_inputs inputs;
        inputs.messages.reserve(n_messages);
        for (size_t i = 0; i < n_messages; ++i) {
            common_chat_msg msg;
            msg.role    = roles[i] != nullptr ? roles[i] : "";
            msg.content = contents[i] != nullptr ? contents[i] : "";
            inputs.messages.push_back(std::move(msg));
        }
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = add_generation_prompt;
        // The server's default (`common_params::reasoning_format`), not the struct's
        // NONE: a template handler may read it.
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        // As the server resolves `enable_reasoning != 0 && template supports it`.
        inputs.enable_thinking = enable_thinking && common_chat_templates_support_enable_thinking(tmpls.get());
        for (size_t i = 0; i < n_kwargs; ++i) {
            inputs.chat_template_kwargs[kwarg_keys[i]] = kwarg_values[i];
        }

        const std::string prompt = prompt_text(common_chat_templates_apply(tmpls.get(), inputs).prompt);
        if (prompt.size() > (size_t) INT32_MAX) {
            set_err(err, err_len, "pipette_chat_apply_jinja: the rendered prompt is too long");
            return -1;
        }
        if (buf != nullptr && buf_len > 0) {
            const size_t n_copy = std::min(prompt.size(), (size_t) buf_len);
            std::memcpy(buf, prompt.data(), n_copy);
            if (n_copy < (size_t) buf_len) {
                buf[n_copy] = '\0';
            }
        }
        return (int32_t) prompt.size();
    } catch (const std::exception & e) {
        const std::string message = std::string("chat template (jinja): failed ") + stage + ": " + e.what();
        set_err(err, err_len, message.c_str());
    } catch (...) {
        const std::string message = std::string("chat template (jinja): failed ") + stage;
        set_err(err, err_len, message.c_str());
    }
    return -1;
}
