# Apple Core AI specialization cache

Core AI does not run a portable `.aimodel` as-is. On first load, Core AI
*specializes* the model for the current device and OS. This step makes
executable artifacts. Core AI keeps the artifacts in `AIModelCache` (see Apple's
[Managing model specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)).
That cache is **outside** pipette's model store.

## Behavior

1. The Swift sidecar constructs `CoreAIEngine` **before** it prints
   `PIPETTE_COREAI_READY`. The construction loads the bundle. It specializes
   the model, or it reads the cached result. Timing cells start after the
   marker. The measured window for prefill, decode, and end-to-end excludes
   specialization. What each cell times is in
   [prefill-throughput.md](prefill-throughput.md#core-ai),
   [decode-throughput.md](decode-throughput.md#core-ai) and
   [end-to-end-latency.md](end-to-end-latency.md#core-ai).
2. `max_memory_usage` reports the lifetime peak `phys_footprint` of the
   sidecar process. The peak includes model load. On a cold cache, the peak
   also includes specialization. The `specialization_cache` value on the
   `PIPETTE_COREAI_READY` line ([Detection](#detection)) records which case
   applies to a run.
3. Pipette does **not** clear the specialization cache between runs, so a
   published number usually comes from a warm cache. Two events make it a
   cold-cache number: the operator deletes the cache, or the OS version
   changes. The `PIPETTE_COREAI_READY` line records which case applies (see
   [Detection](#detection)).

## Detection

The sidecar lists the files in two Apple cache directories before and after
`EngineFactory.createEngine`: `~/Library/Caches/com.apple.coreai` and
`~/Library/Caches/AIModelCache`. A hit means that the directories held files
before the load and held the same files after it. Any other result is a miss.

The sidecar prints `specialization_cache=hit` or `specialization_cache=miss` on
the `PIPETTE_COREAI_READY` line. `GET /health` returns the same fact as the
boolean `specialization_cache_hit`. Two runs of the same cell can differ by
cache state. The READY line records which state produced each run.

## Ahead-of-time compilation

Ahead-of-time compilation (`xcrun coreai-build compile`) can reduce the time
for on-device specialization. The operator chooses whether to use it. Pipette
does not run this step.
