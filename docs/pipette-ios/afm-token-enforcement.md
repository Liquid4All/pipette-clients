# AFM decode: guided vs free generation

The measurement record behind the AFM fixed-token-count choice. Why AFM needs a
fixed N and how production enforces it (free generation on a counting prompt, then
re-tokenize and verify the cap bound) is in [afm-runtime.md](afm-runtime.md).

The alternative is **guided generation** (constrained decoding): `@Generable` +
`@Guide` forces the output shape, so the framework will not emit EOS until the
constraint is met and `maximumResponseTokens` binds at exactly N. That is a hard
guarantee, but constrained decoding runs a per-token logit mask and emits
JSON-structure tokens, which was expected to slow decode and corrupt the throughput
number.

## Experiment

Question: **does guided generation slow AFM decode enough to disqualify it for a
throughput benchmark?**

Probe: `AFMRuntime.enforcementProbe` (headless `runtime=afm metrics=enforceprobe`),
decoding a fixed 100-token cap two ways, 5 measured reps each:

- **guided**: `streamResponse(generating: TokenBurst.self)` where
  `TokenBurst = @Guide(.minimumCount(500)) var items: [String]`; the 500-element
  floor cannot be met within 100 tokens, so the cap truncates the array
  mid-stream, and the output has exactly 100 constrained-decoded tokens.
- **free**: `streamResponse` with the counting prompt, which gives about 100
  free-decoded tokens.

Both decode the same 100-token budget, so the wall-time ratio (guided ÷ free) is
the constrained-decoding tax. Decode time is first→last token on a monotonic
`ContinuousClock`; `tps = 100 / decode_seconds`.

Controls:

- **Device:** iPhone 17 Pro (`iPhone18,1`), iOS 26.x, Apple
  Intelligence enabled, `SystemLanguageModel.availability == .available`.
- **Thermal gate:** each measured rep waits (via `BenchmarkReadiness`, real SoC
  die temp under the `PIPETTE_PRIVATE_THERMAL` build) until the die is below
  36 °C, so heat cannot confound the comparison.
- **Order inversion:** run 1 measured free-first, run 2 measured guided-first, to
  rule out a cold-start / first-mover artifact.

## Results

| Run | Order | Gate | Guided | Free | Guided ÷ Free |
|-----|-------|------|--------|------|---------------|
| 1 | free → guided | off | 820.5 ms (121.9 tps) | 848.3 ms (117.9 tps) | 0.97× |
| 2 | guided → free | on (<36 °C) | 809.9 ms (123.5 tps) | 847.0 ms (118.1 tps) | 0.96× |

All 10 guided reps hit the cap (`cap bound ✓`). Free decode was reproducible
across runs (117.9 / 118.1 tps); guided likewise (121.9 / 123.5 tps).

## Finding

**Guided generation does not slow AFM decode.** On iPhone 17 Pro / iOS 26.x it is
within run-to-run noise of free generation (guided ÷ free 0.96 to 0.97x), if
anything marginally faster. The result held under order inversion and per-rep
thermal gating, ruling out cold-start and heat confounds. The per-token logit mask
is negligible against the ANE forward pass.

So guided generation is a viable way to enforce a fixed N at no measurable
throughput cost. Production still uses free generation plus the cap-reached check;
switching the decode and end-to-end cells to guided generation would be a
measurement change of its own. For that path, a bare `String` guided by a
minimum-length `.pattern` suits better than the probe's `Array.minimumCount`: it
spends the token budget almost entirely on content rather than on per-element quotes
and commas.

## Reproduce

The thermal-gated run needs a private-API build. Build and install it as
[private-API builds § Building it](private-thermal-release-build.md#building-it)
describes, then launch the probe:

```bash
xcrun devicectl device process launch --device <UDID> --console \
  ai.liquid.liquid-pipette headlessrun runtime=afm metrics=enforceprobe
```

Look for the final `SUMMARY … slowdown=…x` line.
