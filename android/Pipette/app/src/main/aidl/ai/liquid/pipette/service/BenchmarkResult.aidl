package ai.liquid.pipette.service;

// Implemented as a Kotlin Parcelable in the same package. Carries the outcome
// of a load/run/unload operation: either success (an inline JSON payload, a
// reference to a spilled payload file, and/or an opaque engine handle) or an
// error message the UI-process proxy re-throws.
parcelable BenchmarkResult;
