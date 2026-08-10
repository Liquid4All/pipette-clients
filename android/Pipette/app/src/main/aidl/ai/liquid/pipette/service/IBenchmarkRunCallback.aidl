package ai.liquid.pipette.service;

// Service -> UI progress channel for an in-flight run. oneway so the service's
// benchmark thread never blocks on the UI process; the return-cancel decision
// is handled out-of-band via IBenchmarkService.requestCancel().
oneway interface IBenchmarkRunCallback {
    // completed/total describe rep progress (total <= 0 means status-only, e.g.
    // a "cooling down" message between reps); message is human-readable text.
    void onProgress(int completed, int total, String message);
}
