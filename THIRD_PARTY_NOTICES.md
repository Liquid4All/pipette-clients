# Third-Party Notices

Pipette Clients includes and distributes third-party software, tools, and assets.
This file summarizes notable bundled components. Generated app/runtime
dependency attributions may also appear in platform-specific acknowledgement
files.

## Vendored Source

| Component | Location | License | Notes |
|---|---|---|---|
| llama.cpp | `vendor/llama.cpp` | MIT | Built into the iOS and Android native clients. |
| nlohmann/json | `vendor/llama.cpp/vendor/nlohmann` | MIT | Header-only, bundled by llama.cpp and compiled into the native bridge. |
| sentry-native | `vendor/sentry-native` | MIT | Built into the Android native benchmark process. |
| KleidiAI | `vendor/kleidiai` | Apache-2.0 | Optional Android native build path when `PIPETTE_ENABLE_KLEIDIAI=1`. |

## Bundled Binary Tools

| Component | Location | License | Notes |
|---|---|---|---|
| Toybox aarch64 | `crates/pipette-llamacpp/vendor/toybox/toybox-aarch64` | 0BSD | Embedded in the Android max-memory measurement path. |
| Gradle Wrapper | `android/Pipette/gradle/wrapper/gradle-wrapper.jar` | Apache-2.0 | Build bootstrap tool. |

## Android App Assets

| Component | Location | License / Status | Notes |
|---|---|---|---|
| Bitstream Charter fonts | `android/Pipette/app/src/main/res/font/` | Bitstream-Charter | Used by the Android app UI. |
| AOSP vector icons | `android/Pipette/app/src/main/res/drawable/ic_*.xml` | Apache-2.0 | Individual XML files retain AOSP license headers. |
| Product/model provider logos | Android/iOS asset catalogs | Trademark/provenance review needed | Third-party marks are not licensed by this repository license. |

## Generated Dependency Attributions

- Android release runtime dependency attributions are generated with
  `cd android/Pipette && ./gradlew :app:generateLicenseReport --no-parallel`.
  The report is written under `android/Pipette/app/build/reports/dependency-license/`.
- Android bundled Rust/native acknowledgements live in
  `android/Pipette/app/src/main/assets/ThirdPartyLicenses.json`.
- iOS Swift package attributions are shown in-app through LicenseList.
- iOS non-SwiftPM bundled component attributions live in
  `ios/Pipette/Pipette/ThirdPartyLicenses.json`.

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

### Toybox

Toybox is licensed under 0BSD.

Copyright (C) 2006 by Rob Landley <rob@landley.net>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

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
