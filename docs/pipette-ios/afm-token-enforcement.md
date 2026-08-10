# AFM decode: enforcing a fixed token count

## Problem

The Apple Foundation Models (AFM) `decode_throughput` / `end_to_end_latency`
benchmarks must decode a *fixed* number of tokens (`parameter_decode_tokens`) so
their timing is comparable to the llama.cpp / MLX runtimes, which force exactly
N tokens. AFM's high-level API makes that hard:

- `GenerationOptions.maximumResponseTokens` is a **ceiling, not a floor**: the
  model can still emit end-of-turn (EOS) early and produce fewer tokens.
- There is no min-tokens / ignore-EOS knob, no logit bias, and no sampling seed
  beyond `.greedy` / `.random`.
- The shipping SDK (iOS 26.5) exposes **no generated-token count**: `Usage` /
  `LanguageModelSession.usage` are in the online docs but not in the SDK, so a
  run can only be counted by re-tokenizing the output text (lossy).

Two ways to hit a fixed N:

1. **Free generation + verify.** Plain `streamResponse` with a non-terminating
   prompt (a counting seed: `… 9, 10,` → greedy keeps emitting `11, 12, …`) so
   the cap binds, then re-tokenize the output to confirm it reached the cap.
   Zero decode overhead, but *no guarantee*. A run can stop early and is only
   caught after the fact.
2. **Guided generation (constrained decoding).** `@Generable` + `@Guide` forces
   the output shape; the framework will not emit EOS until the constraint is
   satisfied, so `maximumResponseTokens` is guaranteed to bind at exactly N.
   This is true enforcement, but constrained decoding runs a per-token logit
   mask and emits JSON-structure tokens, which was expected to slow decode and
   corrupt the throughput number.

## Experiment

Question: **does guided generation slow AFM decode enough to disqualify it for a
throughput benchmark?** If not, it is the better choice (a hard guarantee at no
cost).

Probe: `AFMRuntime.enforcementProbe` (headless `metrics=enforceprobe`), decoding
a fixed 100-token cap two ways, 5 measured reps each:

- **guided**: `streamResponse(generating: TokenBurst.self)` where
  `TokenBurst = @Guide(.minimumCount(500)) var items: [String]`; the 500-element
  floor cannot be met within 100 tokens, so the cap truncates the array
  mid-stream → exactly 100 constrained-decoded tokens.
- **free**: `streamResponse` + the counting prompt → ~100 free-decoded tokens.

Both decode the same 100-token budget, so the wall-time ratio (guided ÷ free) is
the constrained-decoding tax. Decode time is first→last token on a monotonic
`ContinuousClock`; `tps = 100 / decode_seconds`.

Controls:

- **Device:** iPhone 17 Pro (`iPhone18,1`, `Boston-17-pro-3`), iOS 26.x, Apple
  Intelligence enabled, `SystemLanguageModel.availability == .available`.
- **Thermal gate:** each measured rep waits (via `BenchmarkReadiness`, real SoC
  die temp under the `PIPETTE_PRIVATE_THERMAL` build) until the die is below the
  36 °C threshold, so heat cannot confound the comparison.
- **Order inversion:** run 1 measured free-first, run 2 measured guided-first, to
  rule out a cold-start / first-mover artifact.

## Results

| Run | Order | Gate | Guided | Free | Guided ÷ Free |
|-----|-------|------|--------|------|---------------|
| 1 | free → guided | off | 820.5 ms (121.9 tps) | 848.3 ms (117.9 tps) | 0.97× |
| 2 | guided → free | on (≤36 °C) | 809.9 ms (123.5 tps) | 847.0 ms (118.1 tps) | 0.96× |

All 10 guided reps hit the cap (`cap bound ✓`). Free decode was reproducible
across runs (117.9 / 118.1 tps); guided likewise (121.9 / 123.5 tps).

## Conclusion

**Guided generation does not drown AFM decode performance.** On iPhone 17 Pro /
iOS 26.x it is within run-to-run noise of free generation (~3–4%), and if
anything marginally faster. The result held under order inversion and per-rep
thermal gating, ruling out cold-start and heat confounds. The per-token logit
mask is negligible against the ANE forward pass.

Therefore guided generation is the preferred mechanism for a fixed-N AFM decode
workload: it gives a **hard guarantee** of the token count (the model cannot EOS
before the constraint is met) at no measurable throughput cost, replacing the
free-generation + re-tokenize-and-verify path, which has no guarantee.

Implementation note: a bare `String` guided by a minimum-length `.pattern`
spends the token budget almost entirely on content (unlike the array's
per-element quotes/commas), so it is preferred over `Array.minimumCount` for the
production decode/e2e path.

## Reproduce

```bash
./ios/build.sh device -skipMacroValidation -derivedDataPath /tmp/pipette-dd
xcrun devicectl device install app --device <UDID> \
  /tmp/pipette-dd/Build/Products/Debug-iphoneos/Pipette.app
xcrun devicectl device process launch --device <UDID> --console \
  ai.liquid.liquid-pipette headlessrun runtime=afm metrics=enforceprobe
```

Look for the final `SUMMARY … slowdown=…x` line.
