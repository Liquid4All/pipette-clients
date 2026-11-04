# asr_eval chat templates

`voxtral.jinja` is a byte-identical copy of
[`src/asr_templates/voxtral.jinja`](../../asr_templates/voxtral.jinja), the
Voxtral Mini transcription template that `asr_throughput` and `asr_max_memory`
use. `asr_eval.rs` includes this copy. Each copy has its own unit test that pins
the SHA-256
`3d7b2f9d649544f6fe3223620a84d91eb64bd9ff4a7467b0edc843c20b90f5de`, so an edit
to either copy fails its test until the pinned hash changes too.
[`src/asr_templates/README.md`](../../asr_templates/README.md) describes the
format the template follows and how it was verified.

Change it only after review: an edit here changes every Voxtral prompt that
pipette sends.

The LFM3 template is a string constant in `asr_eval.rs`, pipette's own, for the
LFM3 audio entries. It renders the model's native ASR chat format:
`<|startoftext|>`, a system turn, a user turn of the clip alone, and an open
assistant turn with no think block.
