# Speech-Recognition Eval Methodology

## Measured Quantity

Speech-recognition eval benchmarks measure transcription accuracy on a fixed
set of audio clips:

- the nine Open ASR Leaderboard (English) test sets;
- seven FLEURS (`google/fleurs`) locales: `en_us`, `es_419`, `pt_br`, `de_de`,
  `fr_fr`, `it_it` and `nl_nl`. These are English and the languages of the
  leaderboard's multilingual FLEURS runs, without Armenian. The selection is
  pipette's own.

Each sample has a
stable ID, chat messages that carry the clip as an OpenAI `input_audio` part
(base64 WAV) beside the model family's prompt text, and a hidden
reference transcript held by the score server. As with text
[evals](evals.md), the client records the model's raw completion for every
clip; the score server normalizes reference and transcript and reports the
corpus word error rate (`wer`), with the Open ASR Leaderboard's two public
protocols (huggingface/open_asr_leaderboard @
[`62a457c5`](https://github.com/huggingface/open_asr_leaderboard/tree/62a457c5c32d0de4df77e9470308aae7d45b055f),
`normalizer/eval_utils.py`): ESB by its English protocol, and FLEURS by its
multilingual scoring (normalizer, number handling, alignment) applied to
FLEURS's own `transcription` references
([`google/fleurs` @ `70bb2e84`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)):
its multilingual normalizer
([`MultilingualNormalizer`](https://github.com/huggingface/open_asr_leaderboard/blob/62a457c5c32d0de4df77e9470308aae7d45b055f/normalizer/data_utils.py#L34-L92),
its English normalizer for `en_us`), then its compound-pair alignment
([`normalize_compound_pairs`](https://github.com/huggingface/open_asr_leaderboard/blob/62a457c5c32d0de4df77e9470308aae7d45b055f/normalizer/eval_utils.py#L68-L98))
and kaldialign. That normalizer spells every number out in the clip's
language (num2words 0.5.14) on the reference and the transcript alike, so a
model is not charged for writing a number in words where the reference has
digits, or the reverse. The leaderboard's published FLEURS rows score against
its own cleaned copy of the references, so its numbers can differ slightly
from these. Lower is better. A result is an answer to every sample, or no
result: the client never submits a cell with a sample missing or failed (the
`PIPETTE_ALLOW_INCOMPLETE_EVAL` waiver the text and VLM evals honor is ignored
here); mgmt refuses one when it arrives and the score server refuses one
(`400` each), so a gap is never scored as an empty transcript.

In pipette this is the `asr_eval` benchmark family. It reuses the eval
completions payload verbatim (`completions[].{id, completion, failed,
failed_reason, stop_reason, stop_detail, completion_tokens}`), so the same
score pipeline applies.

## Relationship to the VLM eval

`asr_eval` is the [VLM eval](vlm-eval.md) runner with audio in place of
images, on the same llama.cpp `gguf_vision` cell shape (the model and its audio
projector come from the vision model descriptor, `--mmproj`). It shares the
resumable per-sample checkpoint, bounded retries of failed samples,
crash-vs-transport recovery against one shared `llama-server`, the `parallel`
slot pool, and the refusal to publish an incomplete cell. What differs follows.

### The clips are the entry's corpus

An ESB set is gigabytes of audio, so the definition never carries its samples.
A catalog entry pins them, as a vlm_eval entry does ([VLM eval](vlm-eval.md)):
its `[dataset]` names an `asr-eval-corpus/1` corpus (`adapter =
"evals-server@1"`, `digest` and `bytes` of the packed tar, and upstream
`evals_server` with `eval_id`, `dataset_version`, `prompt_profile` and
`identity`). The management server builds it from the score server's samples
for that task, version and prompt profile (each distinct clip stored once,
under `media/<sha256>.wav`), and the run downloads it from `GET /datasets/{id}`.
A device that runs `asr_eval` needs `perform_work`, as a dataset is served only
to a working client (`pipette-mgmt clients permission add <client>
perform_work`). The entries are private to an organization, which puts them
into its own catalog: pipette uses the datasets only for internal,
non-commercial benchmarking of its models, with no redistribution.
SPGISpeech and URGENT 2024 are licensed for non-commercial use only, and
GigaSpeech is gated.

The corpus is checked before the first prompt, and any mismatch fails the run:
answers to other clips would be graded against the wrong references.

- The pin must name the entry's own samples: `asr-eval-corpus/1` from
  `evals-server@1`, with `eval_id`, `dataset_version` and `prompt_profile`
  equal to the entry's `parameter_asr_task`, `parameter_dataset_version` and
  `parameter_prompt_profile`. Refused before the download.
- The disk must hold the corpus, with 256 MiB to spare, before a byte is
  fetched (one profile's corpus of the largest set is about 11.5 GB). The tar
  is streamed to a file under the workspace's `datasets/` directory, hashed as
  it is written, and must be the pinned size and hash to the pinned `digest`.
  A server that sends a byte past the pinned size is cut off there, and one
  that sends nothing for five minutes fails the download.
- The tar is the closed format (`manifest.json`, `samples.jsonl`, `media/*`,
  nothing else, plain files only), its manifest names the pinned format,
  adapter and upstream and counts its samples and media files, and no sample
  id appears twice.
- Every sample is rebuilt, its clips read from the tar by offset and checked
  against their names and sizes, and the samples must hash to the pinned
  `identity`, the score server's digest of what it serves. The corpus must hold
  `parameter_sample_count` samples, and the sample at position `p` must have
  the id `sha256("<eval_id>|<p>")`, in hex, cut to 12 characters.

Nothing holds the corpus whole: the checks keep one sample at a time, and the
run reads the samples back a chunk at a time (at most 32 samples, or 8 per
slot, and 32 MiB of clips), each clip checked again as it is read. The file is
the run's own (two runs of one entry at once each write their own, locked while
the run holds it), deleted when the run ends, by ^C or SIGTERM included. A run
killed outright (SIGKILL, a crash) leaves it, and the next run of the same
entry deletes it, whether that run downloads or has nothing left to send.

Since an id follows from its position, a resumed run knows from its checkpoint
alone what is left. A run whose every row is settled (one killed after its
last answer, before its result was saved) downloads nothing and starts no
server; otherwise the
run downloads the corpus and sends only the rows with no checkpoint entry, or
a failed one with attempts left, skipping every settled stretch.

### The recipe

A catalog entry also carries a `[recipe]`, as a vlm_eval entry does: the model
family, the Hugging Face repos (and projector files) its settings are for,
`enable_thinking = false`, and the public sources the settings come from.
mgmt refuses a cell off it before it queues one, and the client refuses it
again before it downloads anything: a repo the recipe does not cover, a model
not from the Hub, another projector, or thinking not stated off. The cell
states it in its model flags:

    model_flags = { benchmark_type = "asr_eval", model_type = "gguf_vision", enable_thinking = false }

An entry that pins a corpus but names no recipe does not run. The recipe is
not part of the resume identity.

### Each profile's recipe

The entry's required `parameter_prompt_profile` names the model family whose
prompt the score server builds into each sample, and the client runs that
family by its public recipe (its "card"), as pipette-plan-types'
`KnownAsrPromptProfile` lists the sources:

- Gemma 4: Google's gemma-4-E2B-it model card at
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7` ("Best Practices": the audio ASR
  prompt, sampling at temperature 1.0, top_p 0.95, top_k 64, and audio of at
  most 30 seconds, so the score server cuts each clip there; its audio example
  generates `max_new_tokens=512`, the entries' cap).
- Qwen2.5-Omni: Qwen's ASR cookbook (QwenLM/Qwen2.5-Omni at `d8a31ca5`,
  `cookbooks/universal_audio_understanding.ipynb`): greedy, 256 tokens.
- Voxtral: Mistral's transcription usage (the Voxtral-Mini-3B-2507 card's
  "Transcription" sections, and mistral-common's `TranscriptionRequest`):
  temperature 0, 500 tokens, and the transcription prompt format the shipped
  template renders.
- MiniCPM-o: OpenBMB's `chat()` in `modeling_minicpmo.py`
  (openbmb/MiniCPM-o-4_5 at `d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4`).
  Its `prepare_generation_config` sets `repetition_penalty 1.02` on the
  greedy branch (Transformers applies it over the whole sequence) and three
  beams at `repetition_penalty 1.2` on the beam branch; the model's
  `terminators` are `<|im_end|>` and `<|endoftext|>`, and the answer is cut at
  `<|tts_eos|>`. OpenBMB/OmniEvalKit@1adac057 evaluates it by calling
  `chat(use_tts_template=True, enable_thinking=False)` with three beams and no
  sampling (`o_e_Kit/models/minicpm/minicpmo.py`,
  `o_e_Kit/configs/generation_configs.json`, whose ASR sets pin `max_tokens`
  128, the entries' cap). Transformers' length penalty
  defaults to 1.0, and the card runs the model in `torch.bfloat16`, so the
  beams score bf16-rounded logits.
- LFM2.5-Audio: the LiquidAI/LFM2.5-Audio-1.5B card (greedy text decoding).
  The card lists English as its only language. Like every family it runs every
  task, sent the card's one fixed prompt ("Perform ASR.") in every language;
  its non-English FLEURS cells are marked ([below](#the-languages-each-card-declares)).
- LFM3: the LiquidAI/LFM3-Audio-2B card at `2d6c1f99` ("Speech recognition
  (ASR)": the system prompt `Perform ASR in <language>.`, or `Perform ASR.` to
  let the model infer the language). The card gives no decoding settings for a
  transcript, so the entries' greedy decoding and 256-token cap are pipette's
  own, and pipette's template renders the model's native ASR chat format.

Pipette's own choices, where those sources leave a llama-server detail open:
stopping MiniCPM-o at `<|tts_eos|>` by overriding the GGUF's end-of-turn token,
`cache_prompt false` (each clip a fresh prompt, as each `chat()` call is) with
llama.cpp's prompt caches off at the server (`--no-cache-prompt`,
`--ctx-checkpoints 0`, `--cache-ram 0`, each unless the cell's `raw` sets it,
as on vlm_eval: a request's `cache_prompt false` does not stop the server
copying a slot's state out before each new clip, see
[VLM eval](vlm-eval.md#prompt-caches)), the
Gemma 4 and Qwen2.5-Omni stop words, the Voxtral and LFM3 Jinja templates, the
256-token cap the LFM2.5-Audio and LFM3 entries pin, Voxtral's sampler order
(temperature, then top_p), spelling every other profile's chain out as
llama.cpp's default, the per-sample seed (and the Gemma 4 entries' pinned
`parameter_seed = 0`: the card samples unseeded, and the seed makes a run
reproducible), `reasoning_format: none` (the completion comes back as the
model wrote it), and the three attempts a failed sample gets.

The prompt text is the score server's, not the client's: each sample arrives
with its turns built (MiniCPM-o's FLEURS prompts, for one, are written in the
clip's language there), and the client sends them as they are.

**MiniCPM-o's non-English FLEURS prompts were chosen on the validation
split.** OpenBMB publishes ASR prompts for English and Chinese only (and, in
OmniEvalKit, one French prompt written in French), so the prompts for the
other six locales are pipette's. Six candidates were compared on the FLEURS
validation split (`google/fleurs` at `70bb2e84`, split `validation`, every clip
of es_419, pt_br, de_de, fr_fr, it_it and nl_nl), never on the test split the
entries score:

- an English prompt that names the language ("... transcribe the content in
  {language}.");
- the model card's English usage prompt, without a language;
- OmniEvalKit's general English prompt, without a language;
- that English prompt with a clause asking for a verbatim transcript in the
  original language and no translation;
- that English prompt rendered in the clip's language (French: OmniEvalKit's
  `commonvoice_fr` string);
- the rendered prompt with the same clause in the clip's language.

Each ran as the entries run MiniCPM-o (`openbmb/MiniCPM-o-4_5-gguf` at
`db25077c`, Q8_0 weights and OpenBMB's audio GGUF, pipette's ASR build of
llama.cpp, three beams), and every clip was answered. The selection scored
each candidate by corpus word error rate per locale against the
`transcription` field of `google/fleurs`.

The rule was fixed before any candidate ran:

- A candidate is eligible only if at most 2% of its answers, pooled over the
  six locales, are identified by fastText's `lid.176` model as a language
  other than the clip's.
- The eligible candidate with the lowest mean word error rate over the six
  locales wins.

The two candidates with the no-translate clause were eligible, and the
native-language one won: it is the prompt the score server serves. Scored with
the FLEURS protocol that the score server applies (the leaderboard's
multilingual scoring against `transcription`), the rule picks the same prompt.
The test split is run once, by the catalog entries, with that prompt frozen.

| Profile | Server | Request |
|---|---|---|
| `gemma4` | | stop `<eos>`, `<turn\|>`, `<\|tool_response>`; `enable_thinking: false` |
| `qwen25_omni` | | stop `<\|im_end\|>`; `top_k 1` |
| `voxtral` | `--jinja --chat-template` pipette's transcription template | `samplers ["temperature"]` (then `top_p` when pinned), `top_k 0`; the sample's `chat_template_kwargs` (`transcription`, `language`) |
| `minicpm_o` | `--chat-template-kwargs {"use_tts_template":true,"enable_thinking":false}`, `--override-kv tokenizer.ggml.eot_token_id=int:151704`; `--parallel 1 --beam-search 3` | the same kwargs; `cache_prompt false`; three beams: `n_beams 3`, `beam_repetition_penalty 1.2`, `beam_length_penalty 1.0`, `beam_eos_token_ids [151645, 151643]`, `beam_bf16_logits true`; no sampling keys ([Beam search](#beam-search)) |
| `lfm25_audio` | | `top_k 1` |
| `lfm3` | `--jinja --chat-template` the native ASR chat format | `top_k 1` |

Every request carries the entry's effective decoding, pipette-plan-types'
`AsrEvalBenchmark::effective_decoding`, the one table mgmt refuses entries by
and the stored result records: each knob the entry pins, and each it leaves unset
at its card's value. Gemma 4's card samples (temperature 1.0, top_p 0.95,
top_k 64). Every MiniCPM-o catalog entry pins `parameter_num_beams = 3`, so
its requests are the beam requests in the table and carry no sampler at all.
The others transcribe greedily (temperature 0), Qwen2.5-Omni, LFM2.5-Audio
and LFM3 through `top_k 1`, which turns off when the entry pins a temperature
above 0 (so that it samples) unless it pins `top_k` too. Greedy MiniCPM-o is
only a fallback, for an entry outside the catalog that leaves
`parameter_num_beams` unset: it then decodes through `top_k 1` with the
greedy branch's `repeat_penalty 1.02` over the whole sequence, spelled as the
slot's context in `repeat_last_n`. Every sampled or greedy request spells its
sampler chain as `samplers`: Voxtral's is temperature then top_p (temperature alone when the
entry pins no top_p), and every other profile's is llama.cpp's default chain
(`penalties`, `dry`, `top_n_sigma`, `top_k`, `typ_p`, `top_p`, `min_p`, `xtc`,
`temperature`, the same in the vendored llama.cpp and both builds below).
Every other sampler is sent neutral (no presence or frequency penalty, XTC,
DRY or mirostat). So neither a GGUF's `general.sampling.*` defaults, its
`general.sampling.sequence`, nor a server's own chain can change the decode.
A pinned seed is a base from which the entry's `parameter_seed_rule` derives
each clip's, as on the VLM eval (`AsrEvalBenchmark::sample_seed`): under
`per_sample_sha256` every clip draws with a seed of its own, and with no rule
the base seed is offset per repeat. The template arguments a
sample carries pass through unchanged.

The completion is stored as llama-server returns it, as on every eval, and the
score server extracts each profile's transcript. A MiniCPM-o completion so
keeps the TTS template's opening, an empty think block and `<|tts_bos|>`; the
score server reads the text after the last `<|tts_bos|>` and before
`<|tts_eos|>`, as OpenBMB's `_generate_mel_spec` does.

Refused before the corpus is downloaded:

- a profile this client has no recipe for;
- a model that is not the profile's: the projector's audio encoder is read
  from its GGUF (`clip.audio.projector_type`, or `clip.projector_type`) and
  must be the profile's (`gemma4a`/`gemma4ua`, `qwen2.5o`/`qwen2.5oa`,
  `voxtral`, `minicpmoa`, `lfm2a`, `lfm3a`), and on `minicpm_o` the
  backbone's vocabulary must give ids 151704, 151645 and 151643 the pieces
  `<|tts_eos|>`, `<|im_end|>` and `<|endoftext|>` the recipe means by them;
- a cell off the entry's recipe (above);
- what the recipe, the profile or the entry owns, in `raw`: a vlm_eval
  recipe's flags (the samplers and their order, the output cap, grammars, the
  chat template, reply parsing, metadata overrides, adapters, positions,
  context shifting and drafters), and `--beam-search`, `--parallel`/`-np` (the
  typed `parallel` is the knob) and `--kv-unified-per-slot`;
- any `LLAMA_ARG_*` variable in the cell's or the client's envs: llama-server
  reads one before its argv, where the recorded command would not show it.
  Nor `LLAMA_CHAT_TEMPLATE_KWARGS`, which builds before b9360 read the
  template kwargs from. None of these reaches llama-server from pipette's own
  environment either;
- any of those options as a key in llama.cpp's `config.ini`, in any spelling
  or section, and any option through its variable name;
- a decoding the profile's card cannot run, as plan-types'
  `check_decoding` refuses it at mgmt's catalog load too: `num_beams = 1`, a
  beam penalty without beams, any sampler or seed beside beams, beams on a
  profile other than `minicpm_o`, and on `voxtral` a pinned `top_k`, `min_p`
  or `repeat_penalty` (its sampler chain would ignore it).

A task in a language the profile's card does not declare is not refused: the
cell runs and is scored, and is marked (below).

### The languages each card declares

Every family runs all seven FLEURS locales, so a model also runs locales its
card does not claim. Such a cell is marked as outside the card's declared
support, never refused. The entry's recipe carries the mark as `languages`:
the languages the card declares for speech recognition, as BCP 47 primary
subtags (ISO 639-1 where one exists, else ISO 639-3). These are the languages
the card states support for, or, where it states none, the languages it
reports speech-recognition results in. The field is absent when the card
enumerates none. pipette-plan-types' `AsrEvalBenchmark::within_declared_languages`
reads it against the task's language: English for the leaderboard's sets, the
locale's for FLEURS.

| Family | Card | What the card declares | `languages` | Marked FLEURS cells |
|---|---|---|---|---|
| Gemma 4 E2B | [google/gemma-4-E2B-it @ `3e22461f`](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/README.md) | "Automatic speech recognition (ASR) and speech-to-translated-text translation across multiple languages", and FLEURS results, with no list of languages | absent | none |
| Qwen2.5-Omni 3B | [Qwen/Qwen2.5-Omni-3B @ `f75b40e3`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/README.md) | no supported-languages statement; speech-recognition results on Common Voice 15 (English, Chinese, Cantonese, French) and FLEURS (Chinese, English) | `en`, `zh`, `yue`, `fr` | `es_419`, `pt_br`, `de_de`, `it_it`, `nl_nl` |
| Voxtral Mini 3B 2507 | [mistralai/Voxtral-Mini-3B-2507 @ `3060fe34`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/blob/3060fe34b35ba5d44202ce9ff3c097642914f8f3/README.md) | "Natively multilingual: ... (English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)" | `en`, `es`, `fr`, `pt`, `hi`, `de`, `nl`, `it` | none |
| MiniCPM-o 4.5 | [openbmb/MiniCPM-o-4_5 @ `d59117d1`](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/d59117d19ddb8a28a62f1b5f4e3b34ac48d510a4/README.md) | speech "in English and Chinese", and speech-recognition prompts for those two alone | `en`, `zh` | `es_419`, `pt_br`, `de_de`, `fr_fr`, `it_it`, `nl_nl` |
| LFM2.5-Audio 1.5B | [LiquidAI/LFM2.5-Audio-1.5B @ `c362a062`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B/blob/c362a0625dfe45aa588dce5f0ada28a7e5707628/README.md) | "Supported languages: English" | `en` | `es_419`, `pt_br`, `de_de`, `fr_fr`, `it_it`, `nl_nl` |
| LFM3-Audio 2B | [LiquidAI/LFM3-Audio-2B @ `2d6c1f99`](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1/README.md) | "Multilingual speech: transcribe, synthesize, and translate speech across 18 languages", listed as speech languages: Arabic, Danish, German, Greek, English, Finnish, French, Hebrew, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Swedish, Turkish, Chinese | `en`, `ar`, `da`, `de`, `el`, `fi`, `fr`, `he`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `es`, `sv`, `tr`, `zh` | none |

A report shows a marked cell flagged, and leaves it out of any average it
calls an average over supported cells. An average over all seven locales, if a
report gives one, takes the same seven for every model, marked cells included,
and says so. A family whose card enumerates no languages (Gemma 4) has no
marked cell. The mark
changes nothing the client runs: it is the entry's statement for whoever reads
the result.

The Voxtral template ships in the client
(`crates/pipette-llamacpp/src/execute/asr_templates/`). It is pipette's own,
written to Mistral's documented transcription format
(`[INST]...[/INST]lang:<code>[TRANSCRIBE]`); the published GGUFs carry a
text-only template that renders neither the audio nor the `[TRANSCRIBE]`
turn. The Voxtral and LFM3 templates are both passed inline, so the recorded
argv carries them whole.

### The build

Pipette runs these entries on pipette's ASR build of llama.cpp at `362d8dc3e`
(which contains its beam search, below), except LFM3, which needs a llama.cpp
build with its `lfm3a` encoder. The ASR build carries Voxtral, Qwen2.5-Omni and MiniCPM-o audio fixes
that no ggml-org release has, and `minicpmoa` exists only there. A stock
ggml-org release runs these entries, but lacks those fixes, so its numbers are
not comparable with the ASR build's, and it cannot run `minicpm_o` at all.

MiniCPM-o runs OpenBMB's published GGUFs, `openbmb/MiniCPM-o-4_5-gguf` at
`db25077c33951fe163b42986fba0132e279872a2`: the language model, and the audio
encoder `audio/MiniCPM-o-4_5-audio-F16.gguf` as the projector. The ASR build
and the client's projector check read that file as the `minicpmoa` encoder, as
[asr-throughput.md](asr-throughput.md#the-transcription-prompt) describes.

### Beam search

An entry with `parameter_num_beams > 1` decodes by beam search on the ASR
build's `llama-server`, as OpenBMB evaluates MiniCPM-o: the server starts with
`--parallel 1 --beam-search <n>`, and each request carries `n_beams`,
`beam_repetition_penalty`, `beam_length_penalty`, `beam_eos_token_ids [151645,
151643]` (`<|im_end|>`, `<|endoftext|>`) and `beam_bf16_logits true`, read as
one response rather than streamed. A penalty the entry leaves unpinned is the
card's, as plan-types' `effective_decoding` resolves it: repetition 1.2,
length 1.0. Beams are MiniCPM-o's recipe only, the one
whose end-of-beam tokens the client knows, and run on one slot: a beam entry
on another profile, or with `parallel` above one, is refused.

The ASR build splits the room left after the prompt among the beams
(`room = (n_ctx - prompt) / n_beams`, and the best beam stops at
`min(max_tokens, room)` with an ordinary length stop), so the slot holds
`max_tokens` once per beam; see the context sizing below.

### Thinking is always off

There is no `reasoning` knob on the cell: its model flags state
`enable_thinking = false`, every recipe's mode, and a cell that states
anything else (or nothing) is refused. Every server starts with
`--reasoning off`, and the raw thinking flags, environment variables and
config-file keys the other eval cells refuse are refused here too. Gemma 4 and
MiniCPM-o also send `enable_thinking: false` in their template arguments,
whatever a sample says.

## Context sizing

No calibration runs. While the corpus is checked, every clip's duration is
read from its WAV header (`data` bytes over the `fmt ` byte rate, so a 16-bit
PCM and a 32-bit float clip both size right), and one slot is sized for the
corpus's longest sample, floored at a 60-second clip:

    per slot   = audio + text + 256 + generation
    audio      = max(ceil(25 * seconds), 750 * (floor(seconds / 30) + 1))
    generation = max_tokens * parameter_num_beams (max_tokens without beams)

The rate term covers the densest supported encoder (25 tokens a second,
Whisper-style 50 Hz frames pooled by two), Gemma 4's included. The window term
covers the Whisper-family encoders (Qwen2.5-Omni, Voxtral), which run whole
30-second windows and, in stock mtmd, start an extra, silent window at an
exact multiple of 30 seconds (so 30 s bounds at 1500 and 60 s at 2250). 750
is a window at the 25-a-second rate; Voxtral's encoder pools further, so the
term bounds it too. These rates and the 256 below are pipette's own
conservative bounds, not published figures. The text is counted at one token per byte, and 256 covers the chat
template's own tokens. The total passed as `--ctx-size` is the per-slot figure
times `parallel`, and it must fit the 131072 ceiling.

A pinned `ctx_size` is a total and is passed as written, divided across the
slots as llama-server divides it; a pin whose per-slot share cannot hold the
corpus's longest sample is refused before any server starts, naming the
`ctx_size` that would.

An answer the server stops short of `max_tokens` for want of context (a beam
search whose room the prompt took), or a request it refuses for it (a clip
whose duration its header does not say is sent unsized), is not an answer.
It is recorded failed for this run's result, its reason naming the `ctx_size`
that would hold it, but not written to the checkpoint: it costs none of the
sample's three attempts, and rerunning the cell unchanged spends nothing. The
cell fails with no result, naming those rows (`unfit=[...]`); a rerun with a
larger `ctx_size` resumes the checkpoint and runs them.

## Failed samples

A sample whose generation fails is recorded failed in the checkpoint and the
cell fails: nothing is submitted. Each later run of the cell sends it again
while it has attempts left (three in all, counted across runs), and only the
samples still unanswered. Once a sample has spent its attempts, no rerun
publishes the cell.

## Resume identity

The checkpoint is keyed by the text eval's resume identity over the
`asr_eval` definition (task, dataset version, count, profile and decoding),
with its corpus pin reduced to the corpus's `digest`, which names every clip,
prompt and their order (another corpus is another run; the `url` the server
stamps on the pin, its `bytes` and the recipe are not identity), the runtime
flags as resolved, the model
without its Hugging Face token, and the client's recipe version
(`ASR_EVAL_RECIPE_VERSION`, which a test ties to a fingerprint of every
profile's server flags and requests, unpinned and with its sampling knobs
pinned, and of MiniCPM-o's beams, so a client that changes the recipe starts a
new checkpoint rather than resuming one written under the old).

Cleared from it: `parallel` and `ctx_size`, which decide only whether a sample
fits, never what a fitting sample's transcript is (a sample that does not fit
is never checkpointed, above); and every runtime env's value outside the
`GGML_*` namespace, as a result records envs, so a secret such as
`HF_TOKEN=...` is never hashed and rotating it resumes. An env's name, and a
`GGML_*` backend knob's value, stay in.

## Submission

The completions go up in one `POST /benchmarks`, as for every eval, with the
corpus's digest as the row's `observation_dataset_digest`: mgmt scores a run
only from the corpus its entry pins. The checkpoint is retired once the
result is saved, so the next run of the cell generates afresh: a result the
server did not accept stays pending, and `pipette sync` submits it. The
submission's `benchmark_params` are the entry's parameters without their
`parameter_` prefix, beam keys included (`asr_task`, `dataset_version`,
`sample_count`, `prompt_profile`, `max_tokens`, and `num_beams`,
`beam_repetition_penalty`, `beam_length_penalty` when set).

## vLLM

Everything above is the llama.cpp cell, a GGUF and its audio projector. The
same entries also run on vLLM (0.32), with a Hugging Face repo's own weights:
the `uv_vllm` x `torch` cell (`pipette-torch-oai`), as a vlm_eval does
([vlm-eval.md](vlm-eval.md#vllm)).

```toml
models = [{ type = "torch", source = "huggingface", org = "Qwen", repo_name = "Qwen2.5-Omni-3B" }]
runtime_flags = [
  { benchmark_type = "asr_eval", runtime_type = "uv_vllm", model_type = "torch", parallel = 64 },
]
model_flags = [{ benchmark_type = "asr_eval", model_type = "torch", enable_thinking = false }]
```

What is the same: the corpus, its two digest checks and its chunked reads,
the recipe (the repo must be one of the entry's `models` lines, at its root),
thinking off, the checkpoint, and the refusal to submit without an answer to
every clip. What vLLM does itself is not repeated: there is no context sizing
and no slot arithmetic, and a clip vLLM cuts short of the output budget for
want of context fails rather than reports a partial transcript.

- **Which entries.** vLLM implements four of the profiles' models:
  `gemma4`, `qwen25_omni` (its thinker), `voxtral` and `minicpm_o` (4.5). An
  `lfm25_audio` or `lfm3` entry is refused on vLLM, and so is an entry with
  beams: vLLM's beam search takes no repetition penalty, so it cannot run
  MiniCPM-o's beam recipe. Both refusals are plan-types' `check_runtime`,
  which mgmt applies before queueing.
- **Decoding.** Each request sends the entry's effective decoding (the table
  above), every knob: `temperature`, `top_p`, `top_k` (`-1` for none),
  `min_p`, `repetition_penalty`, `presence_penalty` and `frequency_penalty`
  `0`, and the per-clip seed when the entry pins one. The server starts with
  `--generation-config vllm`, so a repo's `generation_config.json` sets no
  default behind it. vLLM's repetition penalty covers the prompt and the
  answer, which is Transformers', the whole-sequence window MiniCPM-o's card
  sets; its sampler order is temperature then top_p, Voxtral's.
- **The prompt** is each repo's own chat template, rendered by vLLM with the
  clip's audio, plus the profile's switches: MiniCPM-o's TTS template
  (`use_tts_template`) with thinking off, stopped at `<|tts_eos|>`
  (`stop_token_ids`) as OpenBMB's `chat()` cuts the answer, with
  `--trust-remote-code` for its processor, and `--chat-template-content-format
  string` with `--interleave-mm-strings`, so the clip follows the instruction as
  its prompt puts it (vLLM otherwise moves a string template's media
  placeholder to the front); Gemma 4's thinking off.
- **Voxtral** is served as Mistral's card serves it (`--tokenizer-mode`,
  `--config-format` and `--load-format mistral`). A transcription clip goes to
  `/v1/audio/transcriptions` with its language: vLLM renders mistral-common's
  transcription request there, the format the llama.cpp profile writes by
  hand, and passes the whole clip to Mistral's encoder, which windows it as
  llama.cpp does. Such a clip must hold its audio alone.
- **The reply** is the model's text as written, special tokens included
  (`skip_special_tokens: false`), as llama.cpp returns it: the score server
  extracts each profile's transcript.

## Not supported

`asr_eval` runs on the llama.cpp `llamacpp_cli_stock_tools` x `gguf_vision`
cell and the vLLM `uv_vllm` x `torch` cell. MLX, ONNX GenAI, OpenVINO, OVMS,
SGLang and docker vLLM refuse it, the iOS client refuses it (and the `asr-eval-corpus/1` format) in
its contract mirror, and the Android client has no cell for it. A local
definition cannot run it (no local catalog entry is seeded): the clips come
only as the corpus a catalog entry pins, from a management server the
workspace is registered with.
