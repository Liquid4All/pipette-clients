# Speech Recognition Throughput and Latency Methodology

## Measured Quantity

`asr_throughput` measures how fast an audio-capable language model turns one
clip of speech, and any text beside it, into generated tokens: the clip passes
through the audio encoder and projector, then the language model prefills the
resulting embeddings (with the model's transcription prompt and the text) and
decodes. The benchmark's definition, parameters, submission fields, derived
metrics and catalog entries are management's
([pipette-mgmt docs/benchmarks.md §2.10](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#210-asr-throughput)).
This doc covers how the client measures it.

**Purpose.** Speech recognition on a language-model backbone is
prefill-heavy in a way text is not: a clip becomes a long run of audio tokens
before the first output token, and how many depends on the encoder rather than
on the model's size. The row reports the audio token count beside the times, so
a model that represents a second of speech in fewer tokens is visibly doing
less work.

`asr_prompt_ms` is the server-reported prefill time, the audio encode
included, not a client wall-clock time to the first token. Runtime install,
model and projector load, server startup, warm-up, readiness waits and result
sync are outside the timed windows.

## End-to-end latency

`total_time_ms` is one request on this client's clock: from the send, once
the request's body (the clip's base64 included) is encoded, to the arrival of
the whole reply, which is not streamed. It is measured, not
`asr_prompt_ms + asr_predicted_ms`, and is the window `vl_throughput` times on
`llama-server` ([vl-throughput.md](vl-throughput.md#end-to-end-latency)).

- Beyond the two timers it covers what `llama-server` does on its HTTP thread
  before the slot's prefill timer starts (reading the body, decoding the
  base64 and the WAV, resampling, the mel spectrogram, rendering the chat
  template and tokenizing), and the reply after the decode timer stops. The
  mel spectrogram grows with the clip, so this share grows with
  `parameter_audio_duration_ms`.
- `asr_predicted_ms` covers every decode step (see
  [The workload](#the-workload)), but a request runs only the
  `parameter_decode_tokens - 1` steps after the first token, so
  `total_time_ms` can read below `asr_prompt_ms + asr_predicted_ms` by up to
  one decode step.

The request goes out on the client's pooled connection, so a rep the readiness
gate held past `llama-server`'s 5 s idle timeout opens a new connection inside
its window, which costs most on Windows
([vl-throughput.md](vl-throughput.md#end-to-end-latency) has the figures). The
debug line for the reply is written after its arrival is stamped, so a slow
`-v` terminal is not timed.

## The Workload

The clip is the synthetic tone management's definition fixes. The calibration
counts tokens on the very clip the cell sends, which its helper resamples as
`llama-server` does, so the count is the sent clip's. Its content does not
matter: how many tokens a clip becomes, and what the encoder costs to run,
depend on the number of samples and not on what they are (a tone, silence and
speech of one length counted alike on every model measured).

The client holds the whole clip in the request that it sends:

- The longest cell in the catalog (119.9 s, 16 kHz mono) is under 4 MB of WAV.
- The extreme that the parameters allow (600 s, 48 kHz stereo) is 115 MB of
  WAV and 154 MB of base64. The encoded request takes as much again while it
  is sent, so the client holds about half a gigabyte beside the server's own
  copy.

Run long high-rate clips on a desktop host.

The first token comes out of the prefill, so either clock (the server's
`timings` or the client's) covers `decode_tokens - 1` decode steps. The reading
scales that time to all `decode_tokens` steps,
`predicted_ms = after_first_ms / (N - 1) * N`, as `vl_throughput` does
([vl-throughput.md](vl-throughput.md#decode-time)). A generation that reports
fewer than 2 tokens is refused.

Generation is forced to the requested length (`ignore_eos`), so every rep
decodes the same number of tokens and the decode time is comparable across
models. Every request disables llama-server's per-request KV reuse, so every rep
prefills the whole prompt ([Runtime details](#runtime-details)), and the entry
refuses a rep whose prompt differs from the first attempt's.

Five reps follow one unrecorded warm-up, each behind the
[readiness gate](end-to-end-latency.md#system-readiness-control), and
each of the three times reduces to a mean and a sample standard deviation over
every rep, as `vl_throughput` reduces its own: no rep is set aside for its
time. The readiness gate before each rep is what keeps a busy host out of the
reps.

## The Ladder

The catalog's clip lengths, text lengths and decode length, and why its rungs
sit at 29.9 s and 119.9 s rather than on a 30 s window edge, are management's
([§2.10, The deployment's ladder](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#the-deployments-ladder)).
`asr_max_memory` runs the same ladder. On pipette's ASR build of llama.cpp,
every family's count at 29.9 s and 119.9 s equals its reference's where one
exists (Gemma 4 E2B and E4B have none past 30 s), but Qwen3-ASR's and
Qwen3-Omni's, which stay over only by their padded last part of a second
([Clips longer than 30 s](#clips-longer-than-30-s)).

## Clips Longer Than 30 s

Past one 30 s window, each family's audio-token count on pipette's ASR build of
llama.cpp was compared with its reference's, as for the short clips
([Reference parity](#reference-parity)): the benchmark's own tone at 60 s, at
119.9 s and at exactly 120 s, counted the way the calibration helper counts it,
against the count the family's official Hugging Face processor gives the same
samples.

| Family | Past 30 s, against the reference |
|---|---|
| LFM2-Audio, LFM2.5-Audio | equal at every length |
| LFM3-Audio | equal to its native ASR front end's count at every length; not yet compared with the `LiquidAI/LFM3-Audio-2B` processor |
| Voxtral Mini | equal at every length, a window of audio for each 30 s |
| Qwen2.5-Omni | equal at every length |
| MiniCPM-o 4.5 | equal at every length |
| Qwen3-ASR, Qwen3-Omni | over by the same rounding as on the short clips: the last part of a second is counted as a whole one, as at 29.9 s, and a clip of whole seconds counts one more second |
| Gemma 4 E2B, E4B | no reference count past 30 s: the processor cuts the clip there (see below) |
| Gemma 4 12B | equal at every length: its processor counts the whole clip |

In this article, "stock" is upstream llama.cpp b11256, and pipette's ASR build
is a set of patches on upstream b11623
([Pipette's ASR build](#pipettes-asr-build-of-llamacpp)). Stock b11256 counts
the same as pipette's ASR build for LFM2-Audio,
LFM2.5-Audio, Qwen3-ASR and Gemma 4. On stock, Voxtral and Qwen2.5-Omni
miscount at every length, and at an exact multiple of 30 s add one more empty
window, so they are run on pipette's ASR build.

**Gemma 4 past its documented limit.** Every Gemma 4 model card documents at
most 30 s of audio, and the E2B and E4B processors cut a longer clip there,
while llama.cpp encodes the whole clip, at the 25 tokens a second it gives a
short one, and the model decodes after it (checked on E2B, E4B and 12B, on
pipette's ASR build and on stock). Both ASR benchmarks run such a clip as a
speed and memory scenario, and the Gemma 4 recipe is the only one that names
a limit, so its rows past 30 s (the catalog's 119.9 s rung) are the only ones
that carry `client_extras.asr_beyond_model_audio_limit = true`. How such a row
reads is management's
([§2.10, Clips longer than 30 s](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#clips-longer-than-30-s)).
12B's rows are flagged too: its processor counts the whole clip, and
llama.cpp's count equals it, but its card documents the same 30 s limit.

**A clip of whole windows at another rate.** The Gemma 4 E2B and E4B encoder
(`gemma4a`) works in 30 s windows of 16 kHz audio, and a last window of a
sample or two fails the encode. A clip of a whole number of windows sent at a
rate other than 16 kHz can resample to just past the last edge (it did at 32
and 48 kHz, and not at the other rates tried), so an E2B or E4B cell of that
shape is refused before anything loads, rather than left to the resampler's
rounding. 12B's encoder (`gemma4ua`) frames the whole clip in one piece, so
the refusal is per encoder, not per family: 12B takes such a clip. At 16 kHz
the clip holds exactly its samples, and a last window of even one millisecond
encodes. Other families take such a clip, but the stray sample is audio too:
Voxtral opens one more window for it, so a clip of whole windows is best sent
at 16 kHz, as every catalog entry is.

## The Text Beside The Clip

`parameter_text_tokens` counts only the extra text the cell adds: exactly that
many tokens under the model's own tokenizer. The chat template, the audio
markers, the recipe's system prompt and its instruction come on top, and are
the same for every text length of a family, so a cell's prompt with text N is
exactly N tokens longer than the same cell's with text 0.

The text is built as `vl_throughput` builds its text, from the shared seed
passage (`pipette_ops::prompt_seed::build_prompt_text`). It is cut under the
server's own tokenizer through its `/tokenize`, after the server is up and
before the warm-up. `0` sends no text and asks for no count.

The count is a difference: the user text part as it is sent (the text, the
newline and the recipe's instruction, below), less the same part without the
text (the instruction alone). The newline that joins the text to the
instruction is a token of its own under some tokenizers (Gemma 4's) and merges
into a neighboring token under others. A count of the text on its own could
therefore make the prompt one token longer or shorter than N.

The count uses no special tokens (`add_special: false`), unlike the
`vl_throughput` count. The part sits inside the user turn, after any BOS that
the template writes, so a count that adds a BOS would build the text one token
short on a model that has one.

**The server confirms the text.** After the reps, besides the text-only request
that confirms the audio count (the same turns with the text and without the
clip), a cell with text sends the same turns without the text as well. The
first must be exactly `parameter_text_tokens` longer than the second, or the
cell is refused rather than publish a row whose text is not the entry's.
`asr_max_memory` makes the same check on its warm-up server, whether or not a
count sized its context.

**Placement, one rule for every family.** The text is the user's, so it joins
the user turn, first in the ask's own text part: the text, a newline, then the
recipe's instruction; the text alone where the recipe sends the clip alone. It
sits on the side of the clip the recipe puts its instruction
(`Recipe::user_text`). So:

- the recipe's system turn, where it has one, is unchanged: for LFM2-Audio,
  LFM3-Audio and Qwen2.5-Omni it is the model's mode switch, and text appended
  to it would change the mode;
- the recipe's ask stays next to the clip it asks about (Gemma 4's "the
  following speech segment", MiniCPM-o's prompt and newline before the clip);
- the newline is the separator llama.cpp itself puts between two text parts
  (`common/chat.cpp`);
- the text and the ask are one part, not two, so the server renders the same
  text with the clip and in the text-only request that confirms the audio
  count. llama.cpp joins the parts of a turn that holds media itself, while a
  template renders the parts of a turn without media, and the two could differ
  by a separator.

| Family | User turn with text |
|---|---|
| LFM2-Audio, LFM3-Audio, Qwen3-ASR | the text, then the clip (system turn, if any, as the recipe has it) |
| Gemma 4, MiniCPM-o 4.5 | the text, a newline and the recipe's prompt, then the clip |
| Qwen2.5-Omni, Qwen3-Omni | the clip, then the text, a newline and the recipe's prompt |
| Voxtral | the clip, then the text (inside `[INST]`, before `[/INST]lang:en[TRANSCRIBE]`), where Mistral's audio-instruct examples put a user's text beside audio |

No family is refused text: every recipe's template renders the user turn's
text, so the prefill holds it on every family. For the families asked in a
transcription mode (LFM2-Audio, LFM3-Audio, Qwen3-ASR, Voxtral) the text is not
an instruction their ASR format has a place for, and what they answer may
change. The timings do not depend on it: the prefill's cost is its token
count, which the row reports, and the decode runs to `parameter_decode_tokens`
whatever the model writes (`ignore_eos`).

The context is sized from the counts:

```text
ctx_size = audio_tokens + text_tokens + recipe_text_bytes + decode_tokens + 64
```

- `audio_tokens`: the exact count that libmtmd produces for the clip
  ([Counting audio tokens](#counting-audio-tokens)), or the upper bound when
  there is no count.
- `text_tokens`: `parameter_text_tokens`, exact, because the text is built to
  that count.
- `recipe_text_bytes`: the bytes of the recipe's system prompt and
  instruction, an upper bound on their tokens.
- `decode_tokens`: `parameter_decode_tokens`.
- `64`: the margin shared with the other benchmarks. It also covers the
  newline and a token that a tokenizer can merge or split where the text meets
  the template.

The entry refuses a row whose audio count and text together are not below the
prompt, which also holds the template's tokens.

The audio count does not depend on the text: the helper counts only the audio
batches it decodes, so one calibration row serves every text length of a clip
(below), and each cell's server confirms it against that cell's own text.

## Counting Audio Tokens

How many tokens a clip becomes is the encoder's business (frame rate, windows,
pooling), so it is read off the real thing rather than computed. The runtime's
own `llama-mtmd-cli` prefills the cell's clip with `-v`, and its log names
every batch of audio embeddings it decoded:

```text
decoding audio batch 1/1, n_tokens_batch = 104
decoding audio batch 1/1, n_tokens_batch = 26
```

The sum is what the text backbone receives between the model's audio markers,
which are text chunks and not part of the count. One clip is often several
chunks (Qwen3-ASR splits into 104-token sub-windows; Whisper-family encoders
and Gemma emit one chunk per 30 s window). The count is **all or nothing**: a
batch line that cannot be parsed fails the whole parse, because the count sizes
the context and an undercount would tip it short.

The result is memoized in `audio_calibration.json` beside the helper, keyed by:

- the projector's fingerprint;
- the helper's build: a SHA-256 of the whole `llama-mtmd-cli` executable and of
  every `libmtmd` library beside it or, for a helper in `bin/`, in its package
  `lib/` (where its launch finds it; a flat archive's parent `lib/` is outside
  the runtime). A patch rarely changes a binary's size or its first pages, so a
  runtime directory rebuilt in place counts again only if the whole content is
  hashed;
- the clip as sent (rate, channels, samples);
- a hash of the cell's environment forwards, so a forwarded value never lands
  in the file.

The helper runs before the server starts, so its model load never overlaps the
resident server's. It is placed like the server, as `vl_throughput`'s helper
is: the cell's `number_gpu_layers` and its `raw` placement flags (`--device`,
`--tensor-split`, `--no-kv-offload` and the like), and none of its other `raw`
flags. That includes the batch sizes (`-b`, `-ub`) and the image-token
overrides `vl_throughput`'s helper takes: the overrides are image-only, and the
batch sizes only split the audio batches the count sums. Its context is bounded
to the clip's upper bound (below) plus the 64-token margin, at least 4096.
Unset, llama.cpp would give it the model's training context (131072 tokens for
Gemma 4) and allocate that KV for a one-token prompt. Neither the placement nor
the context changes the count, so neither is in the cache key. The clip it reads
is written to a file of its own, created exclusively under a random name in the
temp directory and removed after the run.

**The server confirms the count.** `exact` means the count is the serving
path's own, not merely plausible, so every cell holds it to the server:

- A cached or computed count outside 1 to the upper bound (below) is ignored,
  re-measured and the row replaced. A stale count in the millions would
  otherwise size a KV cache of that many tokens.
- After the reps, one request sends the same turns with no clip. The prompt
  with the clip less that prompt is the clip's audio tokens and the audio
  markers the recipe's encoder wraps them in (text libmtmd adds once per clip,
  however many windows it spans), and nothing else. It must equal the count
  plus the recipe's marker count (table below). Anything else, an undercount
  the context's 64-token margin would hide included, means the count is not
  the server's: the cell fails, naming the row, and the row is dropped, so the
  next run counts the clip again. A row never says `exact` beside a count the
  server did not confirm. `asr_max_memory` confirms on its warm-up server and
  requires every later server to prefill the same prompt
  ([asr-max-memory.md](asr-max-memory.md#the-context-is-sized-from-the-exact-audio-token-count)).
  Behind a pinned `ctx_size`, which the count did not size, `asr_throughput`
  reports the measured difference instead of failing.
- A cell that fails after the count sized its context, other than by a
  refusal, drops the row as well, naming it. A stale undercount larger than the
  margin shows only as a failed run: llama-server refuses a prompt past the
  context with HTTP 400 (`exceed_context_size_error`), and a decode that
  reaches the context's end stops short. Dropping on every failure costs one
  helper run when the row was right; keeping a wrong one would fail every later
  cell on its key.
- With no count at all (no helper, or one that failed), the context is sized
  from the upper bound and the audio tokens are measured as the prompt with the
  clip less the same prompt without it. That difference includes the markers,
  so it is the fallback and never the reference.

Each recipe's marker count is a constant of the recipe (`MARKERS_*` in
`crates/pipette-llamacpp/src/asr.rs`), pinned per recipe by its unit tests. It
was read off the real model on the builds the suite runs, as the prompt with
the clip less the prompt without it less the helper's count, on clips of
several lengths, and did not depend on the length, as libmtmd adding the
markers once per clip predicts:

| Recipe (model it was read on) | Audio markers | Marker tokens |
|---|---|---|
| `lfm2a` (LFM2.5-Audio 1.5B Q8_0) | none | 0 |
| `lfm3a` (LFM3 Q8_0) | none | 0 |
| `voxtral` (Voxtral Mini 3B Q4_K_M) | `[BEGIN_AUDIO]` | 1 |
| `qwen2.5o` (Qwen2.5-Omni 3B Q8_0) | `<\|audio_bos\|>` `<\|audio_eos\|>` | 2 |
| `gemma4a` (Gemma 4 E2B Q4_0) | `<\|audio>` `<audio\|>` | 2 |
| `minicpmoa` (MiniCPM-o 4.5 Q4_K_M) | `<\|audio_start\|>` `<\|audio_end\|>` | 2 |
| `qwen3a` alone (Qwen3-ASR 1.7B Q8_0) | `<\|audio_start\|>` `<\|audio_end\|>` | 2 |
| `qwen3a` beside a vision encoder (Qwen3-Omni 30B-A3B Q4_K_M, on the runtime its plan runs) | `<\|audio_start\|>` `<\|audio_end\|>` | 2 |

Without a count the context is sized from an upper bound,
`max(25 tokens per second, 750 per window)`, counting `duration / 30 s + 1`
windows so stock builds that add a silent window at an exact multiple of 30 s
stay inside it. It over-sizes the sparser encoders by design and only runs when
the exact count is unavailable. It holds for every family with a recipe, and an
audio encoder without one is refused (below), so no unmeasured encoder sizes a
context from it.

## The Transcription Prompt

A chat model transcribes when asked the way it was trained to be asked, and the
ask is part of the timed prefill. Which ask follows from the audio encoder the
projector holds, which the projector names in its metadata
(`clip.audio.projector_type`), so no plan spells it out and none can pick the
wrong one for a model. OpenBMB's own MiniCPM-o 4.5 audio GGUF
(`audio/MiniCPM-o-4_5-audio-F16.gguf` in `openbmb/MiniCPM-o-4_5-gguf`) names
none: it is a `whisper`-architecture file with no `clip.*` keys. It is read as
the `minicpmoa` encoder, recognized as pipette's ASR build of llama.cpp
recognizes it, by that architecture, its `audio_pool_step` key and its
`audio_projector.linear2.weight` tensor.

**Thinking is off for every model.** Every ASR server runs with
`--reasoning off`, which needs llama.cpp b8287 or newer (where `-rea` /
`--reasoning` first appears). A transcript has nothing to reason about, no
reference ASR protocol reasons (Google's Gemma guidance and OpenBMB's MiniCPM-o
evaluation both turn it off, LFM3's native ASR opens no think block), and a
thought spends the decode budget outside the transcript. For a model without a
thinking mode the flag changes nothing. The flags below are what each family
adds on top.

Every recipe asks an English clip the way the model's public usage asks it:
its model card, its vendor's cookbook or library, or its vendor's own
evaluation kit, each named below with the reason it matters. Where no public
source gives a value, the value is pipette's choice and the reason given is
measured.

| Projector | Models | System turn | User text | Server flags | Public source |
|---|---|---|---|---|---|
| `lfm2a` | LFM2-Audio, LFM2.5-Audio | `Perform ASR.` | none (clip alone) | none | `LiquidAI/LFM2.5-Audio-1.5B` card, its `liquid-audio` package ("ASR": the fixed system prompt `Perform ASR.`) and its GGUF card (`-sys "Perform ASR."`) |
| `lfm3a` | LFM3-Audio | `Perform ASR in English.` | none (clip alone) | `--jinja --chat-template` (the native ASR prompt, below) | `LiquidAI/LFM3-Audio-2B` card ("Speech recognition (ASR)": the system prompt `Perform ASR in <language>.`) and its GGUF card (`-sys "Perform ASR in English." -p "<__media__>"`) |
| `qwen2.5o` | Qwen2.5-Omni | `You are a speech recognition model.` | `Transcribe the English audio into text without any punctuation marks.`, after the clip | none | Qwen2.5-Omni cookbook `universal_audio_understanding.ipynb` |
| `qwen3a` alone | Qwen3-ASR | none (rendered empty, below) | none (clip alone) | `--jinja --chat-template` (its processor's transcription prompt, below) `--chat-template-kwargs '{"language":"English"}'` | `Qwen/Qwen3-ASR-1.7B-hf` card and `Qwen3ASRProcessor.apply_transcription_request` (transformers) |
| `qwen3a` beside a vision encoder | Qwen3-Omni | none | `Transcribe the English audio into text.`, after the clip | none | Qwen3-Omni cookbook `speech_recognition.ipynb`; the card's default ASR prompt |
| `voxtral` | Voxtral Mini | none | none (clip alone) | `--jinja --chat-template` (its transcription template, below) `--chat-template-kwargs '{"transcription":true,"language":"en"}'` | `mistralai/Voxtral-Mini-3B-2507` card ("Transcription"), `mistral-common`'s transcription encoding |
| `minicpmoa` | MiniCPM-o 4.5 | none | `Please listen to the audio carefully and transcribe it with high precision.` and a newline | `--chat-template-kwargs '{"use_tts_template":true}'` | OpenBMB OmniEvalKit `generation_configs.json` (English general ASR, `fleurs_en`); the `openbmb/MiniCPM-o-4_5` card's `chat()` usage |
| `gemma4a`, `gemma4ua` | Gemma 4 | none | Google's ASR prompt for English (`Transcribe the following speech segment in English into English text.` and its formatting rules), before the clip | none | `google/gemma-4-E2B-it` card ("Audio") |
| other | any other audio encoder | refused | | | |

A cell's text beside the clip joins the user text above
([The text beside the clip](#the-text-beside-the-clip)).

The reasons are measured, not stylistic:

- **LFM2-Audio** enters its ASR mode through the fixed system prompt; a user
  instruction beside the clip turns it back into a chat model.
- **Voxtral** transcribes in its dedicated transcription mode: the clip alone,
  then `lang:en[TRANSCRIBE]`, as `mistral-common` encodes a
  `TranscriptionRequest`. The published GGUFs carry a text-only Devstral
  template that renders neither the audio nor that turn (and under it the
  server prepends a long system prompt, after which the model refuses to
  transcribe), so the recipe serves pipette's own transcription template
  ([`crates/pipette-llamacpp/src/asr_templates/`](../../crates/pipette-llamacpp/src/asr_templates/README.md),
  its SHA-256 pinned by a test).
- **LFM3-Audio** uses the native ASR prompt that its model card gives. Its
  GGUF template opens a `<think>` block by default, so the whole answer lands
  in the reasoning and the transcript is empty; with thinking off it renders
  an empty `<think></think>`, after which the model ended its turn at once on
  several of the clips tried. The recipe's template renders the
  native ASR prompt, which the plain ChatML path matches except for the BOS:
  `<|startoftext|><|im_start|>system\nPerform ASR in English.<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n`.
  It is unit-tested by rendering it (minijinja), and the same render was
  checked on the lfm3 `llama-server`'s `/apply-template`, whose jinja engine is
  llama.cpp's own.
- **Gemma 4** is asked with the ASR prompt its model card gives, with English
  for `{LANGUAGE}`, the text before the clip as the card's example sends it. It
  would think before it answers (its HF template says off, the server's default
  probe turns it on) and spend the decode budget on the thought; the universal
  `--reasoning off` keeps it from doing so.
- **Qwen2.5-Omni** summarizes instead of transcribing without its cookbook
  prompt, which the cookbook sends after the clip, under the speech-recognition
  system prompt.
- **MiniCPM-o 4.5** is asked as OpenBMB's own evaluation kit asks a general
  English clip: the prompt, a newline, then the clip, no system turn (the model
  card's `chat()` example asks the same way, text then audio, with a shorter
  prompt). Without the newline the model answers differently. `chat()` turns
  the TTS template on for any audio and closes the think block; with the TTS
  template argument and `--reasoning off`, which sets the template's
  `enable_thinking` to false (the kwarg spelling of it is deprecated), the
  server renders its prompt token for token.
- **Qwen3-ASR and Qwen3-Omni** share the `qwen3a` audio encoder; the projector
  tells them apart, Qwen3-Omni's holding its vision tower beside it
  (`clip.has_vision_encoder`, as llama.cpp's converter writes it). Qwen3-ASR's
  card recommends its Hugging Face processor's
  `apply_transcription_request(audio, language="English")`, which with the
  `Qwen/Qwen3-ASR-1.7B-hf` template renders
  `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>…<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>`:
  an empty system turn, the clip alone, and the assistant turn prefilled with
  the forced language so the model writes only the transcript. The GGUF's own
  template is plain ChatML, which renders neither the empty system turn nor the
  prefill, so the recipe serves a template that does, given the language as a
  kwarg. It is unit-tested by rendering it, and was checked against the
  processor itself (transformers 5.19, the `-hf` repository at `bcd2b5b7`): the
  processor's prompt and the server's are the same text around the audio, with
  as many tokens besides the audio. Qwen3-Omni is asked as its cookbook asks an
  English clip: the clip, then `Transcribe the English audio into text.`.

**The prompt is the recipe's.** A cell may not change the ask or turn thinking
on, since the ask is part of the prefill the benchmark times and a plan's own
prompt would time a different one under the same id. Management lists the
reserved `raw` flags and env prefixes, which plan-types refuses when a plan
loads
([§2.10, Runtime flags](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#runtime-flags)).
The client refuses them again before anything loads: a reasoning flag in any
spelling llama.cpp reads, refused rather than outranked because llama.cpp
keeps the last of two spellings and the order is the canonical sort's; a
template flag, whether or not the recipe sets one; and an env of either kind,
whether the cell forwards it, the host's configuration sets it or the worker's
own environment holds it. A key setting a reasoning, template, Jinja or
model-source option in a config file llama.cpp applies to every launch is
refused too, as on `vlm_eval` ([vlm-eval.md](vlm-eval.md)).

A recipe that serves its own template also checks the server renders it: once
the server is ready, its `/props` must report that template as the one it
renders, or the cell is refused before any clip is sent. A llama.cpp build
whose jinja engine rejected the template and fell back to its default would
otherwise time another prompt under the recipe's name.

## Reference Parity

The count llama.cpp produces is exact for llama.cpp; it is not always the count
the model's reference implementation produces. The two were compared on clips
from a few seconds to a minute by running each family's official processor for
the reference count. On an unpatched upstream build:

| Family | Result on stock llama.cpp |
|---|---|
| LFM2-Audio, LFM2.5-Audio | exact |
| Gemma 4 12B | exact (its processor counts a clip past 30 s too) |
| Gemma 4 E2B, E4B | exact up to 30 s (the reference stops at 30 s) |
| Qwen3-ASR, Qwen3-Omni | a little over on every clip (the last part of a second is padded to a whole one, and a clip of whole seconds counts one more) |
| Voxtral Mini | half the reference's tokens (the encoder output is average-pooled; the reference never applies its pooler) |
| Qwen2.5-Omni | many times the reference's tokens on a short clip (every clip is padded to a full 30 s window and the whole window is sent) |

Past 30 s each family kept the result it has on the short clips
([Clips longer than 30 s](#clips-longer-than-30-s)).

A result is comparable only against runs on the same llama.cpp build. Where a
family's count differs from its reference, the row's `asr_audio_tokens` says
what was actually prefilled, so the difference is visible, but the timings of
such a family are not a fair reading of the model until the runtime matches its
reference. This benchmark reports what ran and does not correct it.

### Pipette's ASR build of llama.cpp

Pipette's ASR build of llama.cpp carries patches on upstream b11623. Two of
them bring Voxtral and Qwen2.5-Omni to their references: Voxtral without the
average pooling, and Qwen2.5-Omni encoding only the clip's frames in per-chunk
windows. Their counts equal the references' on every clip measured, long ones
included, and the Qwen2.5-Omni encoder output matches the reference audio
tower to within rounding, with an f16 projector and with Q8_0. A third sets
both families' mel floor to the reference feature extractor's 1e-10.
Upstream's 2^-24 agrees with it only on loud audio: on quiet speech the encoder
output drifted visibly from the reference's without the fix, and stayed close
to it with the fix.

MiniCPM-o 4.5 has no audio support upstream. Its offline ASR path is ported
from llama.cpp-omni (a streaming runtime) into the same build, held to openbmb's
own code: equal token counts on every clip and cut tried (long ones included),
encoder output matching the reference's to within rounding with an f16
projector, and the reference's word error rate, its transcripts all but
identical at Q8_0. It also encodes a clip faster than llama.cpp-omni does.

Transcripts were also compared, on LibriSpeech clips from a few seconds to
over a minute, with each model asked as the recipes above ask it, greedy,
against the model's Hugging Face implementation in bf16 asked the way its model
card or cookbook asks it (Voxtral through `apply_transcription_request`,
Qwen2.5-Omni with the clip before the prompt). "Fork" is pipette's ASR build
of llama.cpp, the build that the management server serves as a
[vault source](../pipette-cli/models-and-runtimes.md#vault-sources); "stock" is
upstream b11256.

- **Voxtral Mini 3B (Q4_K_M):** the fork's word error rate equals the
  reference's, and its transcripts are word for word the reference's on most
  clips. Stock halves the audio tokens and loses words, for a clearly higher
  error rate.
- **Qwen2.5-Omni-3B (Q8_0):** the fork prefills the reference's token counts
  and transcribes as the reference does; its few extra errors are a possessive
  split into two words (`mason s`) on its longest clips, which the reference
  also writes on one of them. Stock transcribes about as well but prefills a
  whole 30 s window for every clip under 30 s, so its timings are not the
  model's.

The Hugging Face Qwen2.5-Omni thinker does not stop at its end of turn under
`generate()` and runs on into an invented next turn; the scorer cuts that tail
(on one clip it starts with a blank line rather than `Human:`, and is cut
there too).

## Runtime Details

On the desktop `asr_throughput` runs on `llama-server` only
(`llamacpp_cli_stock_tools` with a `gguf_vision` model, whose projector carries
the audio encoder), and the iOS app runs it in process on its own llama.cpp
([iOS](#ios)). Other engines refuse it. The request is `/v1/chat/completions` with the clip as an OpenAI
`input_audio` part, `max_tokens` set to the decode length, the server's own
`timings` as the clock for the two phases, and the client's for the whole
request.

Refused before anything loads a model, the calibration helper included:

- a projector that holds no audio encoder (`clip.has_audio_encoder` unset), an
  image-only model in an ASR cell;
- a `raw` slot count (`--parallel`, `-np`): the context is sized for one
  request, and llama-server divides it among its slots;
- an audio encoder no recipe asks (named in the error), since its prompt was
  never checked against a reference;
- a reasoning or template knob in `raw`, an env or a llama.cpp config file
  ([The transcription prompt](#the-transcription-prompt));
- a Gemma 4 E2B or E4B clip of a whole number of 30 s windows at a rate other
  than 16 kHz ([Clips longer than 30 s](#clips-longer-than-30-s)).

Refused once the server is up, before any clip is sent: a server whose `/props`
does not report the recipe's own template.

Every row of both benchmarks carries `client_extras.asr_context_sizing` and
`asr_beyond_model_audio_limit`, and `asr_throughput`'s also carries
`asr_audio_tokens_source`; management defines their values. The client sets
`exact` only for a count that sized the context and that the server confirmed,
so it comes only with `calibrated`, and `upper_bound` only when there was no
count, so only with `measured`.

Every ASR server runs one slot (`--parallel 1`), whether or not the cell sets
`no_cache`: the context is sized for one clip, and llama-server's default four
slots would each take a share of it and hold recurrent and sliding-window state
of their own.

The prompt cache, precisely: every request sends `cache_prompt: false`, and
`no_cache = true` (the like-for-like setting, as for `vl_throughput`) also
starts the server with `--no-cache-prompt`. Either disables llama-server's
per-request KV reuse, so no request reuses another's prefix and every rep
prefills the whole prompt. llama-server's separate host-RAM prompt cache
(`--cache-ram`) stays at its default: it cannot skip a prefill (with caching
off, a request prefills from its first token whatever that cache holds), and its
per-request bookkeeping was checked and does not measurably change the timings.

## iOS

The phone has no server to drive, so the app runs the cell in its own process
(`llamacpp_ios_pipette` with a `gguf_vision` model) through the same `mtmd` path
as `vl_throughput`, and rebuilds the server's request rather than approximating
it:

1. Pick the recipe from the projector file's metadata, read before anything
   loads, as the desktop reads it (`clip.has_audio_encoder`, the encoder name,
   `clip.has_vision_encoder`, and OpenBMB's MiniCPM-o file by its architecture,
   key and tensor). The desktop's refusals apply: no audio encoder, an encoder no
   recipe asks, and a Gemma 4 E2B or E4B clip of whole 30 s windows at another
   rate.
2. Write the clip, the same tone byte for byte, to a temp file once per cell,
   outside every timed window, and remove it after the cell.
3. Load the weights, then the projector against them. The projector must
   support audio and report its sample rate; an image-only projector fails the
   cell.
4. Count the clip's audio tokens with the projector (below) and create the
   context from that count:
   `ctx_size = audio_tokens + text_tokens + recipe_text_bytes + decode_tokens + 64`
   ([The text beside the clip](#the-text-beside-the-clip)), with no floor. A cell that pins `ctx_size` keeps it.
5. Build the text beside the clip to `parameter_text_tokens` under the model's
   tokenizer, counted as the desktop counts it (the user text part less the ask
   alone, without special tokens), and render the recipe's turns.
6. Run one unrecorded warm-up, whose prompt is the reference, then five reps,
   each behind the readiness gate, each on a cleared context and sampler. A rep
   that decodes fewer than `parameter_decode_tokens`, prefills another prompt
   than the warm-up's, or yields another audio token count fails the cell.
7. After the reps, confirm the text and the count (below), and refuse a row
   whose audio count and text together are not below the prompt.

The times reduce to a mean and a sample standard deviation over the reps, as
`vl_throughput`'s do. The row carries the same `client_extras` keys and values
as the desktop's.

What differs from the desktop runner:

- **The clock.** The windows are the server's. There is no request to time, so
  `total_time_ms` starts where the server's handler does: before the turns are
  rendered and the clip is read from the file (the WAV decode, the resample and
  the media hash), through the mel spectrogram, to the last sampled token.
  `asr_prompt_ms` is the audio encode and the prefill up to the first sample,
  and `asr_predicted_ms` the `decode_tokens - 1` steps after it, scaled to all
  of them as the desktop scales the server's. No step runs after the last
  sample, as the server releases the slot there. What the phone cannot add is
  the HTTP and JSON handling around a request, so its `total_time_ms` reads
  lower by that. No base64 is encoded: the desktop encodes it before its clock
  starts.
- **Where the count comes from.** The phone has no `llama-mtmd-cli`, so the
  projector tokenizes the media marker with the clip before the context exists,
  and the count is the sum of the audio chunks' tokens, which leaves the markers
  out as the helper's count does. It is not cached: each cell pays one mel
  spectrogram for it. A count that sized the context and is confirmed reports
  `asr_context_sizing = exact` and `asr_audio_tokens_source = calibrated`, and
  a pinned context reports `pinned`. The upper bound
  `max(25 tokens/s, 750 per 30 s window)` never sizes a context: it stands in
  for the count only before the model loads, to order a job's cells and in the
  context size the job records, and leaves the recipe's text out.
- **The confirmations are tokenizations.** With prompt caching off, the server's
  prompt count is the token count of the rendered turns as libmtmd tokenizes
  them, so the app tokenizes the same renders instead of sending requests: the
  turns without the clip, and for a cell with text, the turns without the text
  as well. The text must account for exactly `parameter_text_tokens`, or the
  cell is refused. The prompt with the clip less the turns without it must be
  the count plus the recipe's audio markers, or the cell fails; behind a pinned
  context the measured difference is reported instead.
- **Jinja only.** The turns render through llama.cpp's Jinja engine, never the
  legacy template path the app's other cells try first, with thinking off and
  the template arguments `llama-server` passes for the recipe
  (`enable_thinking`, `preserve_reasoning` and the recipe's own). The
  `llama-server` of the PrismML fork, one of the app's llama.cpp variants
  ([iOS build](../pipette-ios/build.md)), passes `preserve_reasoning` only when
  asked. No ASR turn carries the reasoning that it governs, so the render is
  the same on every build. The recipes'
  own templates (LFM3, Qwen3-ASR, Voxtral) are compiled into the app, and a test
  holds each byte for byte to the crate's copy, so there is no `/props` check to
  make: the app renders the template it was given.
- **No `raw` or envs.** The iOS runtime flags are the vision cells' knobs, so
  the slot-count, reasoning and template refusals have nothing to refuse.

A cell runs the llama.cpp build that its runtime names. The app that this
repository builds carries main's pin and the `prismml` and `lfm3` variants
([iOS build](../pipette-ios/build.md)), not pipette's ASR build. Published
iOS rows for Voxtral, Qwen2.5-Omni and MiniCPM-o come from an app build that
also carries the ASR build. As on the desktop, compare those rows only with
rows from that build.
`asr_eval` stays refused on iOS, and the Android app does not run either ASR
benchmark ([client-parity.md](../client-parity.md), gap 16).
