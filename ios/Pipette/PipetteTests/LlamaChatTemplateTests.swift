import Testing
@testable import Pipette

/// `LlamaCpp.renderChatTemplate` over the library's built-in templates and a real Jinja one,
/// with no model since the template is passed in. The vision cell needs one user turn, the
/// content untouched inside it, and the assistant turn opened after it.
@Suite struct LlamaChatTemplateTests {

    /// A `nil` template renders as chatml, as `llama_chat_apply_template` reads it and as
    /// `llama-server` falls back for a model that ships none.
    @Test func aNilTemplateRendersChatml() throws {
        #expect(try LlamaCpp.renderChatTemplate(nil, model: nil, [(role: "user", content: "hello")])
            == .init(text: "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
                     viaJinja: false))
    }

    /// The content is carried verbatim: a media marker inside it survives the render, so
    /// the templated prompt still tokenizes to an image chunk.
    @Test func carriesTheContentVerbatim() throws {
        let content = "some text<__media__><__media__>"
        #expect(try LlamaCpp.renderChatTemplate(nil, model: nil, [(role: "user", content: content)])
            .text.contains(content))
    }

    /// A named built-in template is honoured when passed by its template string, and a
    /// template the legacy call knows stays on it: the Jinja path is a fallback, so no
    /// prompt that already rendered moves.
    @Test func rendersANamedTemplate() throws {
        let rendered = try "llama3".withCString {
            try LlamaCpp.renderChatTemplate($0, model: nil, [(role: "user", content: "hi")])
        }
        #expect(rendered == .init(
            text: "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n",
            viaJinja: false))
    }

    /// The fixture is the published template, byte for byte, so the renders below are
    /// renders of what the phone reads out of the GGUF.
    @Test func theGemma4FixtureIsThePublishedTemplate() {
        #expect(Gemma4ChatTemplateFixture.digest == Gemma4ChatTemplateFixture.sha256)
    }

    /// The legacy call does not know Gemma-4's template (it returns -1), so it renders through
    /// llama.cpp's Jinja engine, to what `llama-server` renders for the same request:
    /// thinking on, as the server turns it on for a template that supports it, and no BOS,
    /// which the server strips and its tokenizer adds back. `vl_throughput`'s prompt shape:
    /// the text, then one marker per frame, in one user turn.
    @Test func gemma4FallsBackToJinjaAndRendersTheServersPrompt() throws {
        let rendered = try Gemma4ChatTemplateFixture.source.withCString {
            try LlamaCpp.renderChatTemplate($0, model: nil, [
                (role: "user", content: "Describe this image.<__media__><__media__>"),
            ])
        }
        #expect(rendered == .init(
            text: "<|turn>system\n<|think|>\n<turn|>\n"
                + "<|turn>user\nDescribe this image.<__media__><__media__><turn|>\n"
                + "<|turn>model\n",
            viaJinja: true))
    }

    /// `vl_spec_decode`'s shape: a system prompt and a conversation re-rendered whole on
    /// every turn, earlier replies and images included.
    @Test func gemma4RendersAConversationThroughJinja() throws {
        let rendered = try Gemma4ChatTemplateFixture.source.withCString {
            try LlamaCpp.renderChatTemplate($0, model: nil, [
                (role: "system", content: "Be brief."),
                (role: "user", content: "What is this?<__media__>"),
                (role: "assistant", content: "A cat."),
                (role: "user", content: "And this?<__media__>"),
            ])
        }
        #expect(rendered == .init(
            text: "<|turn>system\n<|think|>\nBe brief.<turn|>\n"
                + "<|turn>user\nWhat is this?<__media__><turn|>\n"
                + "<|turn>model\nA cat.<turn|>\n"
                + "<|turn>user\nAnd this?<__media__><turn|>\n"
                + "<|turn>model\n",
            viaJinja: true))
    }

    /// A template neither path can render fails the cell with the engine's reason rather
    /// than taking the app down: the C++ exception stops at the veneer.
    @Test func aTemplateNeitherPathRendersIsRefused() {
        #expect(throws: RuntimeError.self) {
            try "{% for %}".withCString {
                try LlamaCpp.renderChatTemplate($0, model: nil, [(role: "user", content: "hi")])
            }
        }
    }
}
