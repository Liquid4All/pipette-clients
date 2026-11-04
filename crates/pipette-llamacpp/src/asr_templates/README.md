# ASR chat templates

`voxtral.jinja` is pipette's Voxtral Mini transcription template. It follows
Mistral's documented transcription format, as these sources describe it:

- `mistral-common`'s `TranscriptionRequest` (mistralai/mistral-common at
  `b5416172`, `protocol/transcription/request.py`) and
  `InstructTokenizer.encode_transcription`;
- the Hugging Face `VoxtralProcessor.apply_transcription_request`, which
  follows `encode_transcription`;
- the "Transcription" usage on the Voxtral-Mini-3B-2507 model card.

It renders `<s>[INST]`, the audio, `[/INST]`, `lang:<code>` and `[TRANSCRIBE]`,
from the `transcription` and `language` template arguments that the recipe
passes. It renders `lang:<code>` only when the sample names a language. The
published GGUFs carry a text-only Devstral template instead, which renders
neither the audio nor the transcription turn.

The template is verified token for token against mistral-common 1.12.0's
`encode_transcription`, with the public Voxtral-Mini-3B-2507 tokenizer
(`tekken.json` at revision `3060fe34`). pipette's ASR build of llama-server,
serving this template, gives the same results as mistral-common:

- The same token ids around the audio:
  - with a language: `<s> [INST] [BEGIN_AUDIO] … [/INST] lang : en [TRANSCRIBE]`
  - without a language: `<s> [INST] [BEGIN_AUDIO] … [/INST] [TRANSCRIBE]`
- The same prompt token count, with and without a language: 383 and 380 tokens
  on a 2.5 s clip, and 758 and 755 tokens on a 45 s clip.

`src/execute/asr_templates/` holds a byte-identical copy of `voxtral.jinja` for
the `asr_eval` benchmark; its README describes the same file.

A unit test pins its SHA-256
(`3d7b2f9d649544f6fe3223620a84d91eb64bd9ff4a7467b0edc843c20b90f5de`), because
an edit here changes every Voxtral prompt the fleet sends: the test fails until
the new render has been checked against Mistral's format.
