# Pipette Clients

Benchmark runners for evaluating LLM inference engines on edge devices,
from host-native CLIs to native iOS and Android apps. The CLIs and the Android
app share a common Rust core (the iOS app is native Swift). Remote runs can
submit results to the [pipette management
server](https://collector.pipette.liquid.ai); local runs stay local.

## What's in this repository

The **clients** are the benchmark runners you build and run.

### Clients: host-native CLIs

Command-line benchmark runners that run on the host machine. They share one
operating workflow: init → register → install runtime → run/sync, plus an
optional planner claim loop (`pipette worker`) that pulls jobs from the
management server.

| CLI | Inference backends | Build |
|-----|--------------------|-------|
| `pipette`      | `llama.cpp`, MLX, OpenVINO, vLLM, SGLang | `cargo build --release -p pipette-cli` |
| `pipette-plan` | orchestrates `pipette` across remote devices | `cargo build --release -p pipette-plan` |

### Clients: native mobile apps

Mobile clients run benchmarks on-device with engines compiled into the app.
The Android app shares the Rust workspace; the iOS app is native Swift. Linked
engines are platform-specific. See
[docs/README.md](docs/README.md#two-kinds-of-client). Each app has its own
build pipeline:

| App | Location | Build instructions |
|-----|----------|--------------------|
| iOS     | `ios/Pipette/`     | [docs/pipette-ios/build.md](docs/pipette-ios/build.md) |
| Android | `android/Pipette/` | [docs/pipette-android/build.md](docs/pipette-android/build.md) |

### Shared workspace

Not a client itself. `crates/` holds the shared Rust workspace for host CLIs,
runtime libraries, artifact stores, plan types, management client, execution
helpers, and the Android native core.

## Quick start

```bash
cargo build --release -p pipette-cli   # produces target/release/pipette
export PATH="$PWD/target/release:$PATH"   # or copy it somewhere on PATH
```

Then follow the [usage guide](docs/pipette-cli/usage.md) (init → register →
install a runtime → run → sync). Mobile app builds are linked in the table above.

## Docs

Full index and documentation convention: [docs/README.md](docs/README.md).
Architecture lives in [docs/architecture.md](docs/architecture.md); benchmark
measurement rules in [docs/methodology/README.md](docs/methodology/README.md).

## Contributing

This is an Apache-2.0 licensed open source project. See
[CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and [SECURITY.md](SECURITY.md).

## License

Copyright 2026 Liquid AI, Inc.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

The full license text is in [LICENSE](LICENSE), with attribution in
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
