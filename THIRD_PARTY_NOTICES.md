# Third-Party Notices

Pipette Clients includes and distributes third-party software, tools, and assets.
This file summarizes notable bundled components. Generated app/runtime
dependency attributions may also appear in platform-specific acknowledgement
files.

## Vendored Source

| Component | Location | License | Notes |
|---|---|---|---|
| llama.cpp | `vendor/llama.cpp` | MIT | Built into the iOS and Android native clients. |
| llama.cpp forks | `vendor/llamacpp-prismml`, `vendor/llamacpp-lfm3` | MIT | The [private-API build](docs/pipette-ios/private-thermal-release-build.md) of the iOS app carries each fork in its own framework, beside the framework for `vendor/llama.cpp`; see [docs/pipette-ios/build.md](docs/pipette-ios/build.md). |
| nlohmann/json | `vendor/llama.cpp/vendor/nlohmann` | MIT | Header-only, bundled by llama.cpp and compiled into the native bridge. |
| stb_image | `vendor/llama.cpp/vendor/stb` | Public domain or MIT (licensee's choice) | Header-only, bundled by llama.cpp and compiled into its `mtmd` library, which the iOS and Android apps link. |
| miniaudio | `vendor/llama.cpp/vendor/miniaudio` | Public domain or MIT-0 (licensee's choice) | Header-only, bundled by llama.cpp and compiled into its `mtmd` library, which the iOS and Android apps link. |
| ONNX Runtime | `vendor/onnxruntime` | MIT | Built into the iOS app by `ios/build-onnx-genai.sh`. |
| ONNX Runtime GenAI | `vendor/onnxruntime-genai` | MIT | Built into the iOS app by `ios/build-onnx-genai.sh`. |
| sentry-native | `vendor/sentry-native` | MIT | Built into the Android native benchmark process. |
| KleidiAI | `vendor/kleidiai` | Apache-2.0 | Optional Android native build path when `PIPETTE_ENABLE_KLEIDIAI=1`. Its `LICENSES/` also carries BSD-3-Clause, which covers only the googletest archive under `third_party/` that its own tests use. |

## Bundled Binary Tools

| Component | Location | License | Notes |
|---|---|---|---|
| Gradle Wrapper | `android/Pipette/gradle/wrapper/gradle-wrapper.jar`, `tools/adb-connector-apk/gradle/wrapper/gradle-wrapper.jar` | Apache-2.0 | Build bootstrap tool. |

## Android App Assets

| Component | Location | License / Status | Notes |
|---|---|---|---|
| Bitstream Charter fonts | `android/Pipette/app/src/main/res/font/` | Bitstream-Charter | Used by the Android app UI. |
| AOSP vector icons | `android/Pipette/app/src/main/res/drawable/`: `ic_arrow_back`, `ic_check`, `ic_chevron_left`, `ic_chevron_right`, `ic_close`, `ic_download`, `ic_eye`, `ic_eye_off`, `ic_more`, `ic_pause`, `ic_play`, `ic_plus`, `ic_pocket`, `ic_retry`, `ic_search`, `ic_signout`, `ic_upload`, `ic_warning` (`.xml`) | Apache-2.0 | Each of these XML files keeps its AOSP license header. |
| Product/model provider logos | Android/iOS asset catalogs | Trademark/provenance review needed | Third-party marks are not licensed by this repository license. |

## Generated Dependency Attributions

- Android: `android/Pipette/app/src/main/assets/ThirdPartyLicenses.json`
  covers the vendored native code, the linked Rust crates and the Maven
  artifacts. `ci/android-attribution.py --write` generates it.
- iOS: the app shows its Swift package attributions through LicenseList, and
  `ios/Pipette/Pipette/ThirdPartyLicenses.json` covers the bundled components
  that are not Swift packages.

[docs/licensing.md](docs/licensing.md#attribution) describes how each file is
produced and checked.

## License Texts

### llama.cpp

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### nlohmann/json

Bundled by llama.cpp under its own copyright, so it is listed separately from
llama.cpp even though both are MIT.

MIT License

Copyright (c) 2013-2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### ONNX Runtime

MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### ONNX Runtime GenAI

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### sentry-native

MIT License

Copyright (c) 2019 Sentry (https://sentry.io) and individual contributors.
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Bitstream Charter

The Android app bundles Bitstream Charter fonts. The license text below is the
SPDX `Bitstream-Charter` text.

(c) Copyright 1989-1992, Bitstream Inc., Cambridge, MA.

You are hereby granted permission under all Bitstream propriety rights to use,
copy, modify, sublicense, sell, and redistribute the 4 Bitstream Charter (r)
Type 1 outline fonts for any purpose and without restriction; provided, that
this notice is left intact on all copies of such fonts and that Bitstream's
trademark is acknowledged as shown below on all unmodified copies of the 4
Charter Type 1 fonts.

BITSTREAM CHARTER is a registered trademark of Bitstream Inc.

### Apache-2.0 Components

Apache-2.0 components in this repository include the repository's own source,
KleidiAI, the Gradle Wrapper, and AOSP vector icons. See `LICENSE` for the
Apache License 2.0 text.
