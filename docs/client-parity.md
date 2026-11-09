# iOS / Android client parity

The two mobile clients are meant to be one product on two platforms: a result
submitted from either should mean the same thing, and a fleet runbook should
read the same on both. This document is the standing record of where they
differ, so that a gap is filed once rather than rediscovered, and a divergence
chosen on purpose is not re-filed as a bug. Tracked in
PIP-492.

## Why the gaps cluster the way they do

Android has two engine flavors. The non-default `rustEngine` flavor runs the
Rust crates: `crates/pipette-android` `#[path]`-includes
`android/native/benchmarks.rs` and depends on `pipette-ops`, so it inherits
whatever those crates do. The default `kotlinEngine` flavor, which CI and the
release builds ship, ports the kernel to Kotlin over `android/native/jni.cpp`
(`android/Pipette/app/src/kotlinEngine/java/ai/liquid/pipette/llama/LlamaBenchmark.kt`)
and links no crate. iOS re-ports the engine-agnostic half into Swift under
`ios/Pipette/Pipette/Ops/`, and its llama.cpp execute functions after
`crates/pipette-llamacpp`'s `execute/*` (`Runtimes/Llama/Execute/`).

The consequence: **a commit that touches only the crates reaches `rustEngine`
alone and silently skips iOS and the shipping Android build.** #1204 is the
canonical example: it added per-sample measurement logging to
`crates/pipette-ops/src/measurement/mod.rs` and nothing else, and the Swift
port did not get it until a parity pass went looking. When a change to
`pipette-ops` alters observable client behavior, check whether `ios/Pipette/Pipette/Ops/` and
`LlamaBenchmark.kt` need the same change in the same PR, and check
`Runtimes/Llama/` against `crates/pipette-llamacpp`'s `execute/*` for a change
there. A change to `android/native/benchmarks.rs` reaches `rustEngine` only;
iOS has no counterpart of that file.

The reverse direction is rarer but real: iOS ships app-level surfaces (the
headless command tree, storage quota, identity export and import) that Android
has no crate to inherit them from, so they have to be written twice. `auth
export` and `auth import` are the clearest case. The iOS signing key has a
Keychain copy that no crate can reach and that no restore moves to another
device, so the pair of verbs that carries an identity across devices was iOS's
to design.

## Status

| # | Gap | Direction | State |
|---|---|---|---|
| 1 | Headless verb surface | Android trails | **Partly open.** 20 of 34 leaves; see below |
| 2 | Storage quota, GC, storage-limit setting | Android trails | **Open.** Needs a design call |
| 3 | Eval resume (checkpointed completions) | Android trails | **Open.** Needs a design call |
| 8 | `vl_spec_decode` | Android trails | **Open on Android.** Closed on iOS; see below |
| 9 | Vault sources (`vault_*`) | Both trail | **Open.** iOS fetches a gguf `vault_url` and an MLX or ONNX GenAI `vault_archive`. Android fetches none. See below. |
| 10 | Catalog-pinned eval sampling | Android trails | **Open on Android.** Closed on iOS; see below |
| 11 | ONNX Runtime GenAI runtime | Android trails | **Open.** PIP-593 |
| 12 | Declared `benchmark_type` + `benchmark_params` | Android trails | **Open on Android.** Closed on iOS; see below |
| 13 | Runtime output (`runtime_stdout` / `runtime_stderr`) | Both trail | **Open.** Sent by the iOS proxy, not the apps; see below |
| 14 | `vl_max_memory` | Android trails | **Open on Android.** Closed on iOS; see below |
| 15 | North-Mini-Code's reasoning markers in eval text | iOS trails | **Open on iOS.** Out of reach of the phones today; see below |
| 16 | `asr_throughput`, `asr_max_memory` | Android trails | **Open on Android.** Closed on iOS; see below |

Gaps 4, 5, 6, 7a and 7b are closed. Their numbers are not reused, because code
and other docs cite gaps by number.

---

# Open gaps

## Gap 1: the headless verb surface (partly open)

Android has a headless entry point (`headless/HeadlessRunActivity.kt`). What
remains is breadth.

iOS declares **33 leaf verbs** in
`ios/Pipette/Pipette/Headless/HeadlessTree.swift`
([pipette-ios/usage.md](pipette-ios/usage.md#the-verbs) lists them). With the
bare invocation, iOS has **34** leaves in the tables below. Android declares
**20** in `headless/HeadlessVerb.kt`:

| verb | takes | kind | state |
|---|---|---|---|
| `run` (also the bare invocation) | `match` `model` `benchmark` `benchmarks` `submit` `batch` `number_gpu_layers` `readiness_max_wait_secs` `readiness_skip_thermal` | RUN | Verified on device |
| `version` | | READ | Verified on device |
| `status` | | READ | Verified on device |
| `models` | | READ | Verified on device |
| `jobs` | | READ | Verified on device |
| `benchmarks` | `type` | READ | Verified on device |
| `benchmarks-show` | `benchmark` | READ | Verified on device |
| `storage-status` | | READ | Verified on device; no quota field, see gap 2 |
| `runtimes` | | READ | Verified on device; always one row, since MLX and AFM are Apple-only |
| `results` | | READ | Verified on device |
| `results-show` | `result` | READ | Verified on device; bare cell id, unlike iOS |
| `job-rm` | `id` | MUTATE | Verified on device, including the running-job refusal |
| `job-export` | `id` | MUTATE | Verified on device; writes a file where iOS prints |
| `results-delete` | `result` | MUTATE | Verified on device; bare cell id, unlike iOS |
| `models-delete` | `model` | MUTATE | Verified on device; the shared-projector rule is unit-tested only |
| `models-rm` | `name` `repo` | MUTATE | Verified on device; narrows on both selectors, unlike iOS |
| `auth-whoami` | | READ | Verified on device. No iOS leaf |
| `auth-me` | | READ | Verified on device. The only verb that reaches the network |
| `auth-export` | | READ | Verified on device. Prints the signing key, as iOS's does |
| `auth-import` | `bundle` `force` | MUTATE | Verified on device; does not read an iOS or CLI bundle, see below |

"Verified on device" means the verb and its refusals were run against a real
store on a phone, not that a unit test covers it. `verb=version` prints the
build identity, so a re-run can say which artifact it tested. Device passes on
this surface have found defects the unit tests could not, which is why the
column is worth keeping honest. `models-delete`'s projector rule needs a vision
model on the device, which the test phone does not have.

The `kind` column is the enum's own, and it carries no default, so a verb cannot
be added without stating whether it destroys anything.

The two counts do not subtract: some iOS leaves collapse onto one Android verb,
some cannot be built until another gap closes, and one is a divergence we chose.
Every iOS leaf, and where Android stands on it:

| iOS leaf | Android | State |
|---|---|---|
| (bare) | `run` | Done |
| `bench` | `run` | Done |
| `jobs` | `jobs` | Done |
| `status` | `status` | Done |
| `version` | `version` | Done |
| `models list` | `models` | Done |
| `runtimes list` | `runtimes` | Done |
| `results list` | `results` | Done |
| `results show` | `results-show` | Done, one difference |
| `benchmarks list` | `benchmarks` | Done |
| `benchmarks show` | `benchmarks-show` | Done |
| `storage status` | `storage-status` | Done |
| `job rm` | `job-rm` | Done |
| `job export` | `job-export` | Done, one difference |
| `results delete` | `results-delete` | Done, one difference |
| `models delete` | `models-delete` | Done, one difference |
| `models rm` | `models-rm` | Done, one difference |
| `auth export` | `auth-export` | Done |
| `auth import` | `auth-import` | Done, one difference |
| `auth me` | `auth-me` | Done |
| (none) | `auth-whoami` | Android only |
| `job run` | `job-run` | Buildable |
| `auth register` | `auth-register` | Needs network |
| `auth reset` | `auth-reset` | Needs network |
| `models pull` | `models-pull` | Needs network |
| `job submit` | `job-submit` | Needs network |
| `sync` | `sync` | Needs network |
| `benchmarks init-local` | `benchmarks-init-local` | Needs network |
| `benchmarks run` | `benchmarks-run` | Needs network |
| `storage gc` | `storage-gc` | Blocked on gap 2 |
| `diag probe` | none | No equivalent |
| `diag memseq` | none | No equivalent |
| `settings run` | none | Divergence |
| `worker` | none | Divergence |
| `auth set-device` | none | Buildable |

Reading the right-hand column: **20 of the 34 are covered by 19 of the 20
verbs**, because `bench` and the bare form both mean `run`, and `auth-whoami`
answers to no iOS leaf at all. Two more are buildable today and need no
decision: `job run`, and `auth set-device`, whose two device labels are local on
both platforms and whose `client_details` is the `PATCH /clients/me` Android
already makes at startup. Seven need the network. One waits on gap 2 for a quota
to collect against. Two have no Android equivalent to write. The last two rows
are one divergence, not two: `settings run` is iOS's spelling of `worker`, and
Android's claim loop is CLI-side.

Thirteen of the 20 only read, `run` starts work, and six delete or write. The
component is exported, so a verb that destroys data wants its own decision
rather than arriving as a side effect of adding a listing. The local deletes
were allowed on the grounds that the gate is `WRITE_SECURE_SETTINGS`, and anyone
with adb can already `pm uninstall` the app and take the same data with it.
(This protection is incomplete: the permission also carries `development`, so an app granted it
over adb reaches these verbs without being able to uninstall anything. See
`HeadlessCaller.kt`.)

`auth-whoami` is the one Android verb with no iOS leaf. iOS answers the same
question with `auth me local=1`, which prints `client_id`, `server_url` and the
public key from storage without the signing key, so the two could converge on
one spelling.

Six Done rows carry a difference, recorded here so none is re-filed as a bug.

- **`models-rm` with both selectors narrows on Android and does not on iOS.**
  iOS returns on `name` before it consults `repo`, so `name=X repo=Y` deletes
  every X on the device there; Android deletes the X under Y and nothing else.
  Narrowing is the safe reading of two selectors on a delete, and a caller who
  meant every copy can say `name=X` alone.
- **`models-delete` keeps a shared projector.** A vision coordinate names its
  weights and its projector, and iOS deletes both, because its artifact store
  keeps each spec's files as its own entry. Android stores by coordinate, so one
  `mmproj-model-f16.gguf` is the file every quant of that repo pairs with, and
  taking it with one quant would leave the others unable to run. It is deleted
  only when the last model that needed it goes.
- **`job-export` writes a file where iOS prints.** iOS emits the CSV to its
  console between marker lines; Android writes it under
  `getExternalFilesDir(null)/exports/` and prints the path. logcat is the wrong
  place for a multi-line document, and `adb pull` is how a fleet script collects
  one.
- **`results-delete` and `results-show` take a bare cell id where iOS takes
  `jobId/cellId`.** iOS's `ResultId.parse` requires both segments. A cell id is a
  UUID and unique across jobs, and the results walk already maps it back to its
  job, so the pair would be a second thing to keep in step.
- **`auth-import` refuses in three cases iOS does not, and its success line is a
  different record.** The verb writes the device's identity, and the two
  operating systems hold that identity in stores with different failure modes.

  Three refusals have no iOS counterpart:

  1. **A job is running.** The client signs submissions as cells finish, so a key
     swapped mid-run makes every cell that completes afterwards report under the
     restored `client_id`. That splits one device's measurements across two
     clients, and nothing records it. This is the rule the four deletes follow.
  2. **The Keystore cannot open the key this device holds.** Both clients read an
     unreadable slot as absent. On iOS that is safe, because a key it cannot read
     is a key it cannot overwrite: `SecItemAdd` and the `SecItemUpdate` it falls
     back to both need the protection class key that is unavailable in the same
     window (`KeychainHelper.loadData`). The Keystore gives Android no such
     pairing. `seal` can succeed while `unseal` fails, so reading an unreadable
     key as absent would let an unforced import destroy an identity the next
     successful read would have returned. Android therefore refuses, and
     `force=1` does not lift it: an unreadable slot reads exactly like an empty
     one, so the operator cannot tell a fresh device from the identity they would
     destroy. The outage passes and the key returns, so the verb is worth
     retrying rather than overriding.
  3. **A registration is in flight.** `RegistrationService.register` stages a key,
     waits on the server, then promotes it. An import that promoted its own key
     inside that window would leave the server holding a client record no device
     can sign for. `force=1` does lift this one: a registration killed between its
     two writes strands the staged key, nothing on this surface clears that slot,
     and an operator who knows the device is idle needs a way past. iOS's
     `register` (`Client/ClientAuth.swift`) has the same stage-then-promote shape
     and its `auth import` does not guard it, so this is a refusal Android adds
     rather than one the platform forces.

  The success lines are two records rather than one with drift:

  ```
  iOS      auth import       client_id= server_url= public_key= changed=<yes|no>
                             hf_tokens=N [hf_tokens_failed=N]
                             (a forced replace adds an `auth import replaced …` line
                             carrying the caches and model tokens it dropped)

  Android  IMPORTED identity client_id= server_url= signing_key=ok registration=ok
                             replaced_key=<bool> hf_tokens=N [hf_tokens_failed=N]
                             [model_tokens_discarded=N]
  ```

  - **The leading token.** Every mutating verb here leads with a past-tense kind
    token, as `DELETED job=` and `EXPORTED job=` do. `IMPORTED identity` follows
    that rule, so the two lines cannot share a parser whatever happens to the
    fields, which is why the rest of this list is recorded rather than closed.
  - **`replaced_key=` is Android's own.** It records that the import overwrote a
    key that signs for another client, which is the one thing this verb does that
    costs data it never touched.
  - **`model_tokens_cleared=` and `model_tokens_discarded=` count different
    things.** iOS reports how many per-model tokens it **deleted from the
    device**, which are the replaced identity's, on the `auth import replaced`
    line a forced replace emits. Android reports how many the **incoming bundle
    carried** that it cannot store, because it holds one Hugging Face token slot
    rather than one per model, and prints it only when the count is not zero.

  The refusal text differs too: it points at Settings rather than at an `auth
  reset` verb, which Android does not have.

  The read verbs beside it differ less. `auth-whoami` and `auth-me` print iOS's
  fields in iOS's order. `auth-export` also prints `device_labels=no
  settings=no`, which iOS's `auth export` line does not carry.

`job run` is buildable in the same sense the deletes were: the machinery is
already here and blocking. iOS's `job run --id --scope` flips the scoped
terminal cells back to pending and re-runs them on the shared runner, which is
what `JobController.rerunCells` and `retryFailed` already do on Android, and
what the Jobs screen calls. What it needs is the rest of the scope vocabulary:
`retryFailed` already is iOS's `failed` scope, and what has no Android spelling
is `cancelled`, since `rerunCells` takes explicit cell ids. A rerun starts work,
so it carries the running-job and submission questions `run` carries rather
than the ones the deletes raised.

**Identity bundles do not cross to Android.** The CLI and iOS write and read the
same flat version-1 record (`client_id`, `server_url`, `private_key_hex`; iOS
adds its Hugging Face tokens as optional fields the CLI ignores), so a token
from either restores on the other
([moving an identity](pipette-cli/usage.md#moving-an-identity-to-another-machine)). Android
writes and reads only the older shape, with `client_id` and `server_url` nested
under `registration` and `device_labels`, `settings` and `model_tokens` written
empty (`IdentityBundle.kt`). iOS still reads that shape, so an Android bundle
restores on iOS. The CLI does not read it, and Android refuses a flat bundle
from either, so iOS-to-Android and CLI-to-Android restores fail. Closing it is
Android's decoder reading the flat record, then its encoder writing it. Android
keeps its key encrypted under a Keystore AES key and readable in process, the
same custody shape as iOS, so the verbs needed no new mechanism.

What separated the local rows from the ones that need the network is
**mechanism**, not risk. `HeadlessRunActivity` is a synchronous `Activity` that
finishes in `onCreate`, and `JobStore`, `ModelStore` and `Secrets` are all
blocking, so the deletes, the export and the identity pair needed no seam at
all, and neither would `job run`. The intuitive split, "the destructive ones are
the hard ones", is not the one the code is organized around: the deletes were
the easy half to build and the careful half to decide.

`auth-me` is the one verb that crosses that line, and it costs one
`runBlocking(Dispatchers.IO)` in `HeadlessFacts`: the socket work moves off the
main thread, because Android throws `NetworkOnMainThreadException` otherwise,
and the main thread blocks on the result so `onCreate` stays synchronous. That
is affordable for a read that answers one question and finishes. The verbs
still listed as needing the network *publish*: a submission, a registration, a
sync. Each has progress, retries and partial failure to report, which is what
wants a real seam rather than a blocking call.

Two named omissions:

- `spec=`, the JSON `Model` input spelling. Left out because it is a second
  spelling for a thing `model=` already expresses, and Android has no decoder
  for the plan-types tagged shape.
- `model_to_uri`. `models` prints `repo=` and `quant=`, which is the coordinate
  in pieces (`results` prints `quant=` beside a `model=` that carries the repo).
  What is missing is the single-string spelling `model=` accepts, so a caller
  cannot yet copy one line of output into the next command.

**Spelling.** An `am start` line carries only `-e key value` extras, so the verb
arrives as `verb=` and a two-word iOS verb becomes one hyphenated token:
`benchmarks show` is `verb=benchmarks-show`. The spellings are otherwise the
iOS names, so one runbook translates by eye.

**A trap worth carrying to any runbook.** `adb shell` joins its arguments into
one string and hands it to the *device's* shell. A single layer of quotes is
stripped locally, and the `&` separating a URI's pairs backgrounds the command,
so the activity receives a truncated URI and refuses the run for a missing key
the caller can plainly see in their own command line. Use the nested `"'…'"`
form. No unit test catches this: tests hand `parse` a map no shell ever touched.

## Gap 2: no storage quota, GC, or storage-limit setting on Android

iOS has a quota, GC, a storage-limit preset ladder in Settings, `storage gc` in
the headless tree, and LRU eviction; [storage-quota.md](storage-quota.md)
describes the mechanism and where its pieces live on the CLI and iOS.

Android has none of it. The only disk affordance is
`LocalStorage.resetDeviceData()` plus a models-dir path in the debug panel.
Nothing caps the model store, and nothing evicts. Android fetches weights on
demand, including for an unattended `model=` run, so a fleet device can fill
its own disk without a person present, which is the condition a quota exists
for.

Android has no quota because nobody designed one for it, not because a port
missed it. Closing the gap means
storage-quota.md grows an Android column, which is a product decision about
eviction behavior rather than a port.

## Gap 3: no eval resume on Android

iOS `Ops/EvalCompletions.swift` ports `crates/pipette-ops/src/eval_completions.rs`,
so an interrupted eval picks up where it stopped
([eval-checkpoint.md](pipette-cli/eval-checkpoint.md)). Android never references
`eval_completions`, so an interrupted eval re-runs every sample from zero.

In the `rustEngine` flavor, `crates/pipette-android` already depends on
`pipette-ops`, so the store is reachable there. Two things stand in the way,
and both are design questions rather than wiring:

1. `run_eval_impl` (`android/native/benchmarks.rs`, `rustEngine`) and
   `runEvalImpl` (`LlamaBenchmark.kt`, `kotlinEngine`) accumulate completions
   with no checkpoint. The Rust one needs a writable state root threaded
   across JNI.
2. `EvalCompletionsStore::open` keys a session off a `RunRequest`, which the
   Android path never constructs. Something has to decide what keys the run
   digest in its absence.

`android/native/benchmarks.rs` is `#[path]`-included into `pipette-android` and
nothing else, so a fix there reaches `rustEngine` alone. The default
`kotlinEngine` flavor links no crate, so `EvalCompletionsStore` is out of its
reach: closing the gap there means a Kotlin checkpoint store in
`LlamaBenchmark.kt`'s `runEvalImpl`.

## Gap 8: `vl_spec_decode` runs on iOS and not on Android

iOS runs `vl_spec_decode` in process, speculation on or off, where the CLI runs
a `llama-server` per row: `mtmd` and `LlamaMtmd` provide the vision tower, and
`ios/native/pipette_vlspec.{h,cpp}` is a C veneer over llama.cpp's
`common/speculative` for the drafter. The veneer is C++ for the draft-and-verify
loop alone, which has no C API and which a Swift rewrite would turn into a
second implementation of the procedure this benchmark's numbers are compared
against. What the phone shares with the CLI, and the limits it carries (a merged
drafter refused by name, no `envs` or `raw`, one sample in flight), are in
[vl-spec-decode.md](methodology/vl-spec-decode.md#on-ios).

Android's engine has the vision tower `vl_throughput` uses but no drafter, so
`is_compatible` does not pair `gguf_vision_speculative` with
`llamacpp_apk_pipette` and the app cannot claim such a cell. Closing it means
adding a drafter to that engine, then the plan-types pairing.

The batched form of the timing cells (`parameter_batch_size`) is not a gap
between the phones: every in-process engine refuses a batched entry by name
([end-to-end-latency.md](methodology/end-to-end-latency.md#batched)).

## Gap 9: vault sources are fetched by the CLI, and by iOS for a model

`vault_url`, `vault_archive`, `vault_remote_archive` and `vault_wheelhouse`
name a private artifact the management server serves to the devices of the
principal that owns it. The CLI fetches them through
`crates/pipette-artifacts/src/vault.rs`
([models-and-runtimes.md](pipette-cli/models-and-runtimes.md#vault-sources)
has the fetch order and dispositions).

iOS decodes `vault_url` and `vault_archive` in `PlanTypes/Model/`. It fetches a
`vault_url` gguf (text, vision or speculative vision) through
`Artifacts/Vault.swift`, with the CLI's fetch order and dispositions. The
transfer is an in-process task. Each signed request is single-use, and the
content route does not serve a `Range` request.

iOS also fetches a `vault_archive` MLX or ONNX GenAI model through the same
route. Where it differs from the CLI:

- It unpacks a tar or a tar.gz with its own streaming reader
  (`Artifacts/Archive.swift`), because the app has no archive dependency. A zip
  archive is a permanent refusal on iOS; the CLI unpacks one.
- It requests twice the record's size from the storage quota, because the
  archive and the unpacked tree are on disk together. The CLI does not size an
  archive before the fetch.
- It retries an unpack that fails on `EIO`, `EAGAIN`, `EINTR`, `EMFILE` or
  `ENFILE`. The CLI refuses those for good.
- It skips the AppleDouble `._<name>` files that a Mac's `tar` adds; the CLI
  writes them.

The apps do not need `vault_remote_archive` or `vault_wheelhouse`, because
their runtimes ship inside the app. A vault source's `custom_label` is treated
alike on the CLI and iOS: it is no part of a store key, a descriptor digest or
the eval resume identity, and the submitted descriptor reports it back as the
claim carried it.

Android does not know the arms. Its download path is Kotlin code that fetches
from Hugging Face only, and `SubmissionRef.kt` describes every model it ran as
`source: huggingface`.

## Gap 10: the catalog's eval sampling does not reach Android

An `eval` entry may pin `parameter_temperature`, `parameter_top_p`,
`parameter_top_k`, `parameter_min_p` and `parameter_seed`, and a
`vl_spec_decode` entry a `parameter_temperature`. The CLI reads all of them
(`eval_cell::Sampling`, `turn::Envelope`), and so does iOS.

On iOS, `BenchmarkDefinition` decodes the pins (`PlanTypes/Benchmark/Sampling.swift`)
and every eval engine draws under `Sampling`, the port of `eval_cell::Sampling`: a
pinned temperature over the eval id's policy, each pinned cut in place of the
engine's default, and a seed per sample offset by its `#k` repeat index. llama.cpp
builds its sampler chain from them and MLX its `make_sampler` equivalent. Apple
Foundation takes one cut and no min-p, so it refuses an entry that pins min-p, or
top-k with top-p, rather than drawing without them. `vl_spec_decode` hands its
temperature to the native loop (`ios/native/pipette_vlspec.cpp`), greedy unless
pinned. The contract's `sampling.json` holds `Sampling` to the crate's, and
`EngineSamplingTests` holds the llama.cpp and MLX samplers to the servers the CLI drives.

Android sends no temperature at all: `llama::chat_completion`
(`crates/pipette-android/src/llama.rs`) takes the messages, the token cap and
the MCQ choices, and the native sampler is greedy. So Android runs every eval
greedy and trails the eval-id policy too. Closing it there is the same two
pieces: decode the parameters, then widen the inference seam to carry the cuts
and the seed.

## Gap 11: the ONNX Runtime GenAI runtime

iOS has a fourth in-process engine, `onnx_genai_ios_pipette`, with one catalog
row (`microsoft/Phi-3-mini-4k-instruct-onnx`). Android has none of it, so a plan
naming that runtime runs on iPhones alone. PIP-593 is the Android side. Two
things it closes besides the engine itself:

- `ci/checks/model-catalog-parity.py` carries every iOS ONNX row in
  `IOS_ONLY_ONNX`, declared rather than compared, because Android has no ONNX
  catalog to compare against. The check fails the moment the Android catalog
  mentions ONNX, so whoever adds it has to write the coordinate grammar.
- mgmt's capability rule for this runtime pins `os:ios` or `os:ipados`
  (pipette-mgmt's planner, `capability_rules.rs`; no mgmt doc states it).
  An Android engine needs its own rule.

An Android engine also needs a model store to land in. The iOS store admits
every model before a transfer starts, publishes staged and atomic, and pins a
directory transfer to one resolved commit. Android has no quota or eviction at
all (gap 2). [storage-quota.md](storage-quota.md) records how iOS admission
diverges from the CLI's sweep after publish.

## Gap 12: Android does not declare the benchmark it ran

A submission may carry `benchmark_type` with `benchmark_params`, which mgmt
checks against its catalog
([pipette-mgmt docs/httpapi.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/httpapi.md)).
The CLI sends the pair on every run (`BenchmarkParams::from` the definition it
ran), and so does iOS (`PlanTypes/Benchmark/Params.swift`, set by
`buildSubmissionPayloadFromRun`). Android does not, so a stale or edited
definition on an Android phone is still filed under the id.

The keys are optional, so Android loses nothing but the check. Closing it is
wiring: project the benchmark model into the same fields and put both keys on
the payload.

## Gap 13: neither app sends its runtime output

A submission may carry what the runtime printed as `runtime_stdout` and
`runtime_stderr`
([pipette-mgmt docs/httpapi.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/httpapi.md),
[docs/client-integration.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/client-integration.md)).
The CLI sends the streams of a successful run (`results/record.rs`). On iOS
the runtime runs in-process, so its output is the app's own;
`pipette-ios-proxy` captures that console and sends it, but a run the app
submits itself carries none, and neither does anything Android submits.
Capturing it in-app means redirecting the process's own stdout and stderr.

## Gap 14: `vl_max_memory` runs on iOS and not on Android

iOS measures it in process: the model and projector load, `vl_throughput`'s
prompt goes through the vision encoder, one token decodes, and the footprint
peak is the result (`LlamaExecute`, the `.vlMaxMemory` case). Plan-types has a
`vl_max_memory` × `llamacpp_ios_pipette` × `gguf_vision` flags variant and no
`llamacpp_apk_pipette` one, so Android has the vision engine `vl_throughput`
uses but cannot be handed this cell. Closing it there starts in plan-types.

## Gap 15: iOS evals drop North-Mini-Code's reasoning markers

North-Mini-Code (Cohere) brackets its reasoning with `<|START_THINKING|>` and
`<|END_THINKING|>` and its answer with `<|START_TEXT|>` and `<|END_TEXT|>`, and
the scorer splits the two on them. The GGUF converter makes all four control
tokens, which `llama-server` drops from a completion, so the llama.cpp CLI asks
for them as `preserved_tokens` on every `/completion`
(`crates/pipette-llamacpp/src/execute/eval.rs`). Android's engine renders every
control token (`ee_llama_token_to_str`), these included. iOS renders none
(`LlamaCpp.piece`, `special: false`), so an iOS row would carry the reasoning
and the answer with no boundary between them.

North is a 30b-class model, so no phone runs its evals, and the gap is recorded
rather than closed. Closing it is `llama-server`'s rule: tokenize each string with
special parsing once per model, keep the ones that are a single token, and
render those ids with `special` on in the free-text loop.

## Gap 16: `asr_throughput` and `asr_max_memory` run on iOS and not on Android

iOS measures both in process on a `llamacpp_ios_pipette` × `gguf_vision` cell:
the clip is the synthetic tone the desktop sends, written once per cell to a
temp file and read through `mtmd_helper_bitmap_init_from_file`, which decodes,
downmixes and resamples it as `llama-server` does; the recipe comes from the
projector's metadata; the prompt renders through Jinja alone; and the audio
token count that sizes the context is read off libmtmd's own tokenization of the
clip, since a phone runs no `llama-mtmd-cli` (`LlamaExecute`, the
`.asrThroughput` and `.asrMaxMemory` cases). What the phone shares with the CLI
and where it differs are in
[asr-throughput.md](methodology/asr-throughput.md#ios) and
[asr-max-memory.md](methodology/asr-max-memory.md#ios). `asr_eval` stays
refused on iOS (`ios/contract-refusals.json`).

Android has no audio input: the native mtmd kernel in
`android/native/benchmarks.rs` feeds the projector images only, plan-types has
no `llamacpp_apk_pipette` variant for either kind, and the catalog sync skips
the benchmark types as unrecognized (its native dispatch would refuse them as
unsupported). Closing it there starts in plan-types, then the same clip, recipe
and count on that engine.

---

# Deliberate divergences, not gaps

Do not file these.

- **Apple-only runtimes.** MLX and AFM exist on iOS and cannot exist on Android.
- **Planner claim loop.** In-app on iOS, CLI-side on Android. Documented at
  `JobRunner.kt`.
- **Reinstall identity.** iOS keeps both halves twice: `identity/id_ed25519` and
  `identity/registration.json` in the container, and a copy of each in the
  Keychain. The container survives a re-signed build and the Keychain survives
  an uninstall, so either event leaves the pair intact. Android deliberately
  excludes the identity from both backup channels because the key is
  Keystore-bound; if Android restored the key, two devices would share one
  `client_id` (`res/xml/data_extraction_rules.xml`). iOS reaches the same outcome
  by a different route: its Keychain copies are `ThisDeviceOnly`, and
  `identity/` is excluded from backup, so neither half crosses one. `auth
  export` stays the deliberate way to move an identity.
- **Top-app demotion guard** is Android-specific. Some Android builds (Samsung
  One UI, for example) move a process that is not the visible `top-app` into a
  restrictive cpuset, even a foreground service such as the `:benchmark`
  process. Android watches for that move (`CpusetWatcher.kt`,
  `CpuAffinityProbe.kt`). iOS has no cpusets.
- **Richer thermal telemetry on Android:** per-sensor HAL reads via `dumpsys`.
- **`match=` is not widened to the offerable catalog on Android.** It filters
  bytes already present; `model=` names bytes to obtain. Widening it would turn
  an existing runbook's `match=LFM` into a multi-gigabyte download nobody asked
  for.
- **Each preset GGUF catalog is the set of configurations the results dashboard
  publishes for that handset** (PIP-511), filtered to its llama.cpp engine:

      https://pipette.liquid.ai/results?hw=Galaxy+S26+Ultra+%2F+llama.cpp%3Aandroid-arm64-v8a
      https://pipette.liquid.ai/results?hw=iPhone+17+Pro+%2F+llama.cpp%3Aios

  That filter is the whole rule, and it is re-derivable: open the link and read
  the rows. A model absent from it has no published reading on that handset and
  is not offered; one present is. `ci/checks/model-catalog-parity.py` names the
  rows where the two lists differ and fails on any divergence it does not; read
  the list there rather than a copy here.

  The rule is scoped to llama.cpp, which is what those filters pin. The
  dashboard publishes no MLX results on any device, so it cannot speak to the
  iOS MLX catalog either way; that catalog keeps its own criterion (`hf_mlx` in
  the plan files) and so carries families that the GGUF catalogs do not offer.
  `CatalogEntryTests.mlxOnlyFamilies` audits which.
- **An ambiguous `match=` resolves differently per client** (PIP-511). Three
  behaviors, not two. iOS refuses, preferring a row named exactly `match`, on
  the `bench` verb and on any deep link that named a model. iOS's bare sweep
  form (`headlessrun` with no verb) takes the first match instead, which is what
  its CLI twin does. Android benches *every* match, because `selectModels`
  returns them all. So one runbook line can produce a refusal, one cell, or
  several, depending where it runs. Android has no deep-link bench surface at
  all. Neither remedy travels. `quant=` is an iOS `bench` key and is absent from
  Android's `run` verb (`HeadlessVerb.RUN`), which refuses it outright. Naming a
  row in full means a different string per iOS surface: headless compares the
  catalog display name and `repoIdentifier`, so `match=org/repo` resolves, while
  a deep link compares the on-device artifact leaf, where that same string
  matches nothing. Spell the pin for the client you are driving.

# Verified at parity

Checked and matching, so a pass can skip them unless something changes:

Auth is fully converged in both directions: password sign-in, second factor,
password reset, set-password on a passwordless account, device self-confirm,
sign-out unlink. iOS cites Android's naming directly at `EmailAuthModel.swift`.

Also matching: v1 canonical request signing, the `load_mode` runtime flag,
tolerant catalog ingestion, download bytes-and-rate progress, Settings
debug-card gating, per-cell result upload, CSV export, power-save reporting,
thermal waiver, readiness-policy recording, pocket mode, acknowledgements,
feedback.

# Running the pass again

1. `git log --oneline <last-pass-sha>..origin/main` (the last full pass was
   before the sync commit `8715581`
   ([#27](https://github.com/Liquid4All/pipette-clients/pull/27)), so use that
   commit; the verb counts were re-checked at the sync) and read every commit that touches `crates/pipette-ops`, `android/native/`,
   `ios/`, or `android/`. CI-only and plan-only commits can be skipped, but say
   so rather than assuming.
2. For each crate-level change, check whether `ios/Pipette/Pipette/Ops/` got the
   same change. That is the failure mode this document exists for.
3. Re-verify the open gaps above rather than trusting this table. Each one names
   the file to grep.
4. Update this file and PIP-492 together, and move the last-pass SHA in step 1.
