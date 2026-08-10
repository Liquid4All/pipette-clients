package ai.liquid.pipette.service;

// Reverse channel :benchmark -> main. The main process registers one of these via
// IBenchmarkService.setJobCancelCallback so BenchmarkActivity's Cancel button (which
// lives in the :benchmark process) can reach the main-process JobController and cancel
// the whole job, not just abort the current cell. oneway: fire-and-forget from the
// service's binder thread.
interface IJobCancelCallback {
    oneway void onCancelRequested();
}
