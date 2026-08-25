# Apple Core AI specialization cache

Core AI does not run a portable `.aimodel` as-is. On first load it
*specializes* the model for the current device and OS, producing executable
artifacts that live in the process's `AIModelCache` (see Apple's
[Managing model specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)).
That cache is **outside** pipette's model store.

## Policy

1. The Swift sidecar constructs `CoreAIEngine` (which loads the bundle and
   therefore specializes, or hits the cache) **before** it prints
   `PIPETTE_COREAI_READY`. Timing cells only start after READY, so
   specialization is outside the measured window for prefill, decode, and
   end-to-end.
2. `max_memory_usage` starts its `phys_footprint` poller **after**
   `start_server` returns. The first-load specialization peak is therefore
   excluded by policy, not by accident. A later run against a warm cache
   has nothing to exclude.
3. Pipette does **not** clear the specialization cache between runs. A
   published number is a warm-cache number unless the operator wiped
   `AIModelCache` (or the OS version changed, which invalidates the cache).
4. Two runs of the same cell can still differ by cache state (cold vs warm,
   or a purge under storage pressure). The result payload does not yet
   record a cache-hit bit; treat that as part of the host environment
   alongside thermal state.

Ahead-of-time compilation (`xcrun coreai-build compile`) can shrink on-device
specialization. It is an operator choice, not something the harness does.
