// Does the macOS thermal-pressure notification track the SoC actually cooling?
//
// `crates/pipette-readiness/src/macos.rs` gates benchmark readiness on
// `notify_get_state(kOSThermalNotificationPressureLevelName)`, proceeding
// only at level 0. Measured on a fanless MacBook Neo (A18 Pro):
//
//   - The notify signal is live: it reaches `moderate` under load, so it is
//     not a constant 0. (`notify(3)` accepts *any* name and reports 0 for
//     it, so that needed proving -- see the `bogus` control column.)
//   - It moves in lock-step with `ProcessInfo.thermalState` in both
//     directions, so switching to it bought no latency, only the removal of
//     a `swift -e` compiler spawn from the poll loop.
//   - It behaves like a *timer*, not a temperature threshold: one run showed
//     38.20C reading `moderate` and, two seconds later, the same 38.20C
//     reading `nominal`. The die had been at that temperature for ~5 minutes.
//
// So the question is how long the gate keeps waiting after the die is back at
// baseline. This samples an idle baseline, heats the SoC with an all-core
// load, then idles -- recording IOHID die temperature (`PMU tdie*`) next to
// both coarse enums -- and reports that interval. It also compares the
// enum's recovery time against `DEFAULT_MAX_WAIT`, because a host that needs
// longer than the deadline doesn't just wait: the gate times out and fails
// the cell.
//
// The cooldown runs until the machine is actually done cooling rather than
// for a fixed span: it ends once the die has held flat for a whole trailing
// window *and* the enum has returned to nominal. Waiting on the enum too is
// what keeps the headline interval measurable -- on the Neo the die settled
// in 24s and the enum took 317s, so a die-only exit would stop the run five
// minutes before the number this tool exists to produce.
//
// The run waits for a genuinely nominal start before measuring. A run that
// begins at `fair` is measuring the tail of some earlier thermal event rather
// than the load applied here; the first Neo run did exactly that and its
// numbers are an upper bound, not a measurement.
//
// Die temperature comes from the same IOHID sensors as the iOS client's
// `pipette_soc_temp` (`ios/Pipette/Pipette/Native/PipetteThermal.m`) --
// private symbols via dlsym, `PMU tdie*` services, max reading. Those names
// verified identical on Apple Silicon Macs. Private API, so this tool stays
// a diagnostic and nothing links it into a shipped binary.
//
// Objective-C rather than a shell loop on purpose: reading ProcessInfo
// in-process avoids `swift -e` spawning the Swift compiler on every sample,
// which would itself heat the machine being measured.
//
// Build/run: see README.md.

#import <Foundation/Foundation.h>
#include <libkern/OSThermalNotification.h>
#include <dlfcn.h>
#include <math.h>
#include <notify.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// A nonsense name. notify(3) accepts it and reports 0; if this column ever
// shows non-zero, the whole comparison is untrustworthy.
static const char *kBogusName = "com.example.pipette.no.such.notification";
// The shorter, undocumented sibling of the linked constant. Confirmed to sit
// at 0 while the linked one moves; kept as a second-machine cross-check.
static const char *kUndocumentedName = "com.apple.system.thermalpressure";
// The die counts as "steady" when its spread (max - min) across the trailing
// window stays inside a band derived from the idle baseline: it has settled
// once it is no jumpier than it was before the load. The band has to be
// measured per host, because the idle noise is wildly hardware-specific --
// sigma 0.26-0.41C on a 20-sensor actively-cooled Mac against 0.80C on the
// fanless 7-sensor Neo, whose die wanders 3.79C peak-to-peak doing nothing.
//
// It is calibrated by sliding the *same* window over the baseline and taking a
// high percentile of the ranges seen there. That is the identical statistic the
// test computes, on data known to be idle, so it needs no distributional
// assumption at all. An earlier version converted a sigma into an expected
// range with a fixed multiple; the Neo showed why that fails. Its noise is
// autocorrelated (lag-1 +0.68), not white, so the Gaussian range factor is
// wrong and drifts with window length -- measured p90/sigma was 3.51 at a
// 16-sample window but 4.73 at 31. No single multiple is right for both.
static const double kBandPercentile = 0.90;
// Sliding windows overlap, so they are not independent samples and the tail of
// their distribution is a soft estimate. Below this many, fall back to sigma.
static const int kMinCalibWindows = 10;
// Fallback only, when the baseline is too short to hold enough windows. Same
// Gaussian-range approximation as before, with all its caveats.
static const double kSigmaMultiple = 4.0;
// Floor, tied to hardware: these sensors quantize at ~0.10C, so a band under a
// couple of steps is asking for resolution the sensor does not have.
//
// There is deliberately NO ceiling. An earlier version clamped the band at
// 2.00C to catch a "contaminated" baseline; on the Neo that cut a correct
// 3.21C band by 38%, put it below the median idle window range of 2.71C, and
// made `steady` unsatisfiable -- then labelled the good data contaminated. A
// band too wide to be meaningful is now caught after the fact in the verdict,
// where the load-induced rise is actually known to compare it against.
static const double kBandFloorC = 0.2;
// Used when there is no usable baseline to derive a band from at all.
static const double kBandFallbackC = 0.5;
// A range test alone cannot see drift: max-min answers "how far did it move",
// never "was it moving". Measured on the Neo, `steady` fired 97s after load
// stop with the die still falling at 3.21 C/min, because a slow monotonic
// descent looks exactly like stationary noise to a spread. So the window's
// trend is tested too -- against a threshold calibrated the same empirical way,
// since idle |slope| over a 60s window has a median of 1.27 C/min on that host
// and any "sensible" fixed limit lands below it.
//
// Note the cost: resolving zero-slope needs a long window (idle p90 |slope|
// falls from ~2.6 C/min at 60s to ~0.3 at 180s), and a long window cannot
// report sooner than its own length. That trade is why `recovered`, not
// `steady`, ends the cooldown.
// "Recovered" = die back near the idle baseline. Tested on a short smoothed
// mean rather than one sample, because a single reading carries the full noise:
// on the Neo the old hardcoded 3.0C threshold sat *below* the 3.79C idle
// peak-to-peak, so a lone dip could read "recovered" while the die was still
// genuinely hot. Smoothing is kept short to avoid lagging a fast descent.
//
// This is the cooldown's exit criterion. Comparing a smoothed reading against a
// known absolute reference is a far easier estimation problem than detecting a
// zero slope, and it needs no long window: on the Neo it resolved at 147s where
// an honest trend test needed 224s.
// Smoothing is a DURATION, not a sample count. Noise decorrelates on a
// timescale (lag-1 +0.68 at 2s spacing implies a correlation time of ~5s), so
// sampling four times faster gives four times the samples and almost no extra
// noise reduction. Expressing this in samples silently changed the physics
// whenever `interval` changed.
//
// 2s by default: short enough not to lag a fast descent, long enough to be
// worth averaging. The threshold derived from it uses the EFFECTIVE sample
// count, not the raw one -- an earlier version divided by sqrt(5) and claimed
// a resolution the correlated samples could not actually support.
static const double kRecoveredSmoothSecs = 2.0;
static const double kRecoveredFloorC = 1.0;
static const double kRecoveredSigmas = 2.0;
// A baseline is "drifting" when the total change implied by its linear fit
// exceeds its own sigma. Run A came in at 0.25x (clean); run 3, which began
// `nominal after 0s` while the die was still descending, at 1.49x. `nominal`
// does not mean settled -- the enum is a release timer, not a thermometer.
static const double kDriftSigmas = 1.0;
// Cycle convergence, by EQUIVALENCE rather than significance.
//
// "No statistically significant trend" was the previous rule and it fails in a
// specific, systematic way: a search that walks forward shrinking the tail also
// shrinks the sample size, and t falls purely from lost degrees of freedom. On
// a real 15-rep Neo batch the peak slope estimate was +0.193, +0.216, +0.203,
// +0.193, +0.195 C/rep across successive tails -- essentially constant -- while
// t fell 4.19, 3.91, 2.87, 2.04, 1.45. The rule "declared convergence" at the
// exact point the trend stopped being *detectable*, not where it stopped.
//
// Non-significance is not evidence of flatness; it is often just absence of
// data. So the test is inverted: bound the slope from above at 95% and require
// the WORST-CASE drift it permits across the tail to be small next to the
// rep-to-rep scatter already present. This has the right incentive -- more reps
// narrow the interval and make convergence easier to demonstrate, where the old
// rule made it easier with fewer.
static const double kDriftScatterMultiple = 2.0;
// Minimum reps in a tail before a slope means anything.
static const int kMinTailReps = 4;

// Two-sided 95% t critical values by degrees of freedom; 2.0 beyond the table.
static double tCrit95(int df) {
    static const double t[] = {0.0,  12.71, 4.303, 3.182, 2.776, 2.571, 2.447,
                               2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
                               2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086};
    if (df < 1) return 12.71;
    if (df <= 20) return t[df];
    return 2.0;
}
// Tolerances reported in the settling curve, hottest first.
static const double kSettleTolerances[] = {4.0, 3.0, 2.0, 1.5, 1.0, 0.5};
#define kSettleToleranceCount \
    ((int)(sizeof(kSettleTolerances) / sizeof(kSettleTolerances[0])))
// The settled reference is the mean of the last this-many seconds of cooldown.
static const int kSettleRefSecs = 90;
// `DEFAULT_MAX_WAIT` in crates/pipette-readiness/src/macos.rs. If the enum
// needs longer than this to clear, the gate times out and fails the cell.
// Raised 300 -> 420 once this tool measured the hold-off at ~318 s: the old
// deadline sat below it, so the gate failed cells on fanless hosts. The enum is
// the gate's only thermal criterion; die temperature is recorded, not gated on.
static const int kGateDefaultMaxWaitSecs = 420;

static volatile bool gStop = false;

static void *burn(void *unused) {
    (void)unused;
    volatile double x = 1.0;
    while (!gStop) {
        for (int i = 0; i < 200000; i++) {
            x = x * 1.0000001 + 0.7;
            if (x > 1e30) x = 1.0;
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// IOHID die temperature, mirroring the iOS client's pipette_soc_temp.
// ---------------------------------------------------------------------------

typedef struct __IOHIDEvent *EvRef;
typedef struct __IOHIDServiceClient *SvcRef;
typedef struct __IOHIDEventSystemClient *CliRef;
typedef CliRef (*FnCreate)(CFAllocatorRef);
typedef void (*FnSetMatching)(CliRef, CFDictionaryRef);
typedef CFArrayRef (*FnCopyServices)(CliRef);
typedef CFTypeRef (*FnCopyProperty)(SvcRef, CFStringRef);
typedef EvRef (*FnCopyEvent)(SvcRef, int64_t, int32_t, int64_t);
typedef double (*FnEventFloat)(EvRef, int32_t);

#define kEventTypeTemperature 15
#define kTemperatureField (kEventTypeTemperature << 16)

static FnCopyEvent gCopyEvent;
static FnEventFloat gEventFloat;
static CFArrayRef gServices;    // held for the process lifetime
static SvcRef *gDieSensors;     // borrowed refs into gServices
static int gDieCount = -1;      // -1 = unavailable

// Resolve the private symbols, enumerate AppleVendor temperature sensors, and
// keep the `PMU tdie*` ones. Done once: the service list is stable, so only
// event copies need to repeat per sample.
static void dieTempInit(void) {
    FnCreate create = (FnCreate)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientCreate");
    FnSetMatching setMatching =
        (FnSetMatching)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientSetMatching");
    FnCopyServices copyServices =
        (FnCopyServices)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientCopyServices");
    FnCopyProperty copyProperty =
        (FnCopyProperty)dlsym(RTLD_DEFAULT, "IOHIDServiceClientCopyProperty");
    gCopyEvent = (FnCopyEvent)dlsym(RTLD_DEFAULT, "IOHIDServiceClientCopyEvent");
    gEventFloat = (FnEventFloat)dlsym(RTLD_DEFAULT, "IOHIDEventGetFloatValue");
    if (!create || !setMatching || !copyServices || !copyProperty || !gCopyEvent ||
        !gEventFloat) {
        fprintf(stderr, "warning: IOHID symbols unavailable; die temp disabled\n");
        return;
    }
    CliRef client = create(kCFAllocatorDefault);
    if (!client) {
        fprintf(stderr, "warning: IOHIDEventSystemClientCreate failed; die temp disabled\n");
        return;
    }
    // AppleVendor temperature sensors: PrimaryUsagePage 0xff00, PrimaryUsage 0x0005.
    setMatching(client,
                (__bridge CFDictionaryRef) @{@"PrimaryUsagePage" : @(0xff00),
                                             @"PrimaryUsage" : @(0x0005)});
    gServices = copyServices(client);
    long n = gServices ? CFArrayGetCount(gServices) : 0;
    gDieSensors = calloc((size_t)(n > 0 ? n : 1), sizeof(SvcRef));
    if (!gDieSensors) return;
    int kept = 0;
    for (long i = 0; i < n; i++) {
        SvcRef svc = (SvcRef)CFArrayGetValueAtIndex(gServices, i);
        CFTypeRef name = copyProperty(svc, CFSTR("Product"));
        if (name && CFGetTypeID(name) == CFStringGetTypeID() &&
            [(__bridge NSString *)name hasPrefix:@"PMU tdie"]) {
            gDieSensors[kept++] = svc;
        }
        if (name) CFRelease(name);
    }
    gDieCount = kept;
    // `client` is intentionally leaked: the borrowed service refs in
    // gDieSensors must outlive it, and this process is short-lived.
}

// Hottest `PMU tdie*` reading in C, or NAN if unavailable.
static double dieTempMax(void) {
    if (gDieCount <= 0) return NAN;
    double best = NAN;
    for (int i = 0; i < gDieCount; i++) {
        EvRef ev = gCopyEvent(gDieSensors[i], kEventTypeTemperature, 0, 0);
        if (!ev) continue;
        double t = gEventFloat(ev, kTemperatureField);
        CFRelease(ev);
        if (t > -50.0 && t < 150.0 && (isnan(best) || t > best)) best = t;
    }
    return best;
}

// Spread (max - min) of a die-sample window, or NAN if empty.
static double spreadOf(const double *v, int n) {
    if (n <= 0) return NAN;
    double lo = v[0], hi = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return hi - lo;
}

// Population standard deviation of die samples, or NAN if too few.
static double sigmaOf(const double *v, int n) {
    if (n < 2) return NAN;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i];
    double mean = sum / n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) sq += (v[i] - mean) * (v[i] - mean);
    double var = sq / n;
    return var > 0.0 ? sqrt(var) : 0.0;
}

static int cmpDouble(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

// A duration in seconds becomes however many samples that is at this interval,
// always at least one.
static int smoothSamples(double secs, double interval) {
    int n = (int)(secs / interval + 0.5);
    return n < 1 ? 1 : n;
}

static void sleepSecs(double s) {
    if (s <= 0.0) return;
    struct timespec ts;
    ts.tv_sec = (time_t)s;
    ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

// Lag-1 autocorrelation. Measured, because it decides whether sampling faster
// buys anything: this noise is strongly autocorrelated (+0.68 at 2s spacing on
// the Neo, and higher the closer together you sample), so averaging more
// samples from the same few seconds does NOT average the noise away.
static double lag1Of(const double *v, int n) {
    // A handful of samples gives a wild estimate -- a 6-sample baseline
    // produced -0.66 here, which is noise, not anticorrelation. Refuse rather
    // than hand back a number that would be trusted.
    if (n < 20) return NAN;
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += v[i];
    mean /= n;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n - 1; i++) num += (v[i] - mean) * (v[i + 1] - mean);
    for (int i = 0; i < n; i++) den += (v[i] - mean) * (v[i] - mean);
    return den > 0.0 ? num / den : NAN;
}

// How many INDEPENDENT samples `n` correlated ones are worth. For an AR(1)
// process the variance of the mean inflates by (1+r)/(1-r), so the honest
// sample count is n(1-r)/(1+r). At r=0.68 that is 0.19n -- five samples are
// worth about one, and a 5-sample mean is barely smoother than a single
// reading. Oversampling improves TIME resolution, never noise.
static double effectiveN(int n, double r) {
    if (n <= 0) return 0.0;
    // Unknown correlation resolves to the CONSERVATIVE answer: assume the
    // samples tell us nothing new about each other. Defaulting to `n` here
    // would quietly hand back the optimistic assumption that this whole
    // correction exists to remove.
    if (isnan(r)) return 1.0;
    if (r <= 0.0) return (double)n;
    if (r >= 0.99) return 1.0;
    double eff = n * (1.0 - r) / (1.0 + r);
    return eff < 1.0 ? 1.0 : eff;
}

// Least-squares fit of y against index, reporting the slope, its standard
// error, and the residual scatter. Used to ask whether a sequence is actually
// flat, which a spread cannot answer: five reps creeping +0.31C each span only
// 0.9C over four of them, which slips under any plausible spread threshold
// while climbing 3C over a longer batch.
static void fitTrend(const double *y, int n, double *slope, double *se, double *resid) {
    *slope = *se = *resid = NAN;
    if (n < 3) return;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += i;
        sy += y[i];
        sxx += (double)i * i;
        sxy += (double)i * y[i];
    }
    double den = n * sxx - sx * sx;
    if (den == 0.0) return;
    double sl = (n * sxy - sx * sy) / den;
    double ic = (sy - sl * sx) / n;
    double ss = 0;
    for (int i = 0; i < n; i++) {
        double r = y[i] - (sl * i + ic);
        ss += r * r;
    }
    double rs = sqrt(ss / (n - 2));
    *slope = sl;
    *resid = rs;
    *se = rs / sqrt(sxx - sx * sx / n);
}

// Least-squares trend of `n` samples spaced `interval` seconds apart, in C/min.
// NAN if there are too few points to fit.
static double slopeOf(const double *v, int n, double interval) {
    if (n < 2) return NAN;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        double t = (double)i * interval;
        sx += t;
        sy += v[i];
        sxx += t * t;
        sxy += t * v[i];
    }
    double den = n * sxx - sx * sx;
    if (den == 0.0) return NAN;
    return (n * sxy - sx * sy) / den * 60.0;
}

// Everything the steadiness test needs to know about how this host behaves when
// it is doing nothing: the distribution of range AND trend over the very same
// window the test will apply. Measured, never assumed -- see the header.
typedef struct {
    double rangeMedian, rangeP90;
    double slopeMedian, slopeP90;  // C/min, absolute values
    int windows;
} Calib;

static double percentileOf(double *sorted, int n, double p) {
    int i = (int)(n * p);
    if (i >= n) i = n - 1;
    if (i < 0) i = 0;
    return sorted[i];
}

// Slide a `w`-sample window over the baseline and collect both statistics.
// `scratch` must hold at least 2*n doubles. Returns the window count.
static int calibrate(const double *base, int n, int w, double interval, double *scratch,
                     Calib *out) {
    out->rangeMedian = out->rangeP90 = out->slopeMedian = out->slopeP90 = NAN;
    out->windows = 0;
    if (w < 2 || n < w) return 0;
    double *ranges = scratch, *slopes = scratch + n;
    int cnt = 0;
    for (int i = 0; i + w <= n; i++) {
        ranges[cnt] = spreadOf(base + i, w);
        double s = slopeOf(base + i, w, interval);
        slopes[cnt] = s < 0 ? -s : s;
        cnt++;
    }
    qsort(ranges, (size_t)cnt, sizeof(double), cmpDouble);
    qsort(slopes, (size_t)cnt, sizeof(double), cmpDouble);
    out->rangeMedian = percentileOf(ranges, cnt, 0.5);
    out->rangeP90 = percentileOf(ranges, cnt, kBandPercentile);
    out->slopeMedian = percentileOf(slopes, cnt, 0.5);
    out->slopeP90 = percentileOf(slopes, cnt, kBandPercentile);
    out->windows = cnt;
    return cnt;
}

// Turn the idle baseline into the steady band, preferring the empirical
// window-range calibration and falling back to sigma when the baseline is too
// short to support it. `*note` is set whenever the number did not come from the
// preferred path, so a fallback or floored band is always visible in the
// summary rather than passing as a measurement.
static double deriveSteadyBand(const Calib *c, double baseSigma, int baseSamples,
                               const char **note) {
    *note = NULL;
    double band;
    if (c->windows >= kMinCalibWindows && !isnan(c->rangeP90)) {
        band = c->rangeP90;
    } else if (baseSamples >= 2 && !isnan(baseSigma)) {
        band = kSigmaMultiple * baseSigma;
        *note = c->windows > 0 ? "baseline too short to calibrate; sigma fallback"
                               : "baseline shorter than the window; sigma fallback";
    } else {
        *note = "no usable baseline; fallback";
        return kBandFallbackC;
    }
    if (band < kBandFloorC) {
        *note = "raised to floor; baseline quieter than sensor resolution";
        return kBandFloorC;
    }
    return band;
}

// How long after load stop the die first reached, and then *held*, each
// tolerance around its settled value. The gap between those two columns is the
// noise: on the Neo the die first came within 1C at 105s but kept wandering
// back out until 318s. Reporting only "first reached" would overstate how
// quickly a repeatable thermal state is available.
//
// `t`/`v` are the cooldown trace, `n` long, and `stopAt` is when the load
// stopped. Prints nothing if the trace is too short to have a settled tail.
static void printSettlingCurve(const double *t, const double *v, int n, double stopAt,
                               double sigma, int smoothN) {
    if (n < smoothN + 2) return;
    double refSum = 0.0;
    int refCount = 0;
    for (int i = 0; i < n; i++) {
        if (t[i] >= t[n - 1] - kSettleRefSecs) {
            refSum += v[i];
            refCount++;
        }
    }
    if (refCount < 2) return;
    double ref = refSum / refCount;

    printf("\n=== settling curve (after load stop) ===\n");
    printf("  settled reference %.2fC = mean of the last %ds of cooldown\n", ref, kSettleRefSecs);
    // This asks a different question from `recovered`, and the two can differ
    // by minutes. Here the reference is where the die actually ENDED UP, so the
    // curve answers "how long until repeated runs start from the same place" --
    // the batch-spacing question. `recovered` compares against the pre-load
    // idle baseline instead, i.e. "how long until it is properly cold".
    printf("  (reference is where the die settled, NOT the pre-load baseline: this is the\n");
    printf("   batch-spacing question, `recovered` above is the return-to-idle one)\n");
    printf("  %-12s %-14s %s\n", "tolerance", "first within", "and stays within");
    for (int k = 0; k < kSettleToleranceCount; k++) {
        double tol = kSettleTolerances[k];
        double first = -1.0, stay = -1.0;
        for (int i = smoothN - 1; i < n; i++) {
            double m = 0.0;
            for (int j = i - smoothN + 1; j <= i; j++) m += v[j];
            m /= smoothN;
            double d = fabs(m - ref);
            if (d <= tol && first < 0) first = t[i];
            if (d > tol) stay = -1.0;
            else if (stay < 0) stay = t[i];
        }
        printf("  %-12s ", [[NSString stringWithFormat:@"%.1fC", tol] UTF8String]);
        if (first >= 0) printf("%-14s ", [[NSString stringWithFormat:@"%.0fs", first - stopAt]
                                             UTF8String]);
        else printf("%-14s ", "never");
        if (stay >= 0) printf("%.0fs\n", stay - stopAt);
        else printf("never\n");
    }
    // Below roughly 2 sigma the tolerance is smaller than the sensor's own idle
    // wander, so "never" there is a property of the hardware, not the cooling.
    if (!isnan(sigma)) {
        printf("  note: idle sigma is %.2fC, so tolerances under ~%.1fC cannot be held at all\n",
               sigma, 2.0 * sigma);
    }
}

// ---------------------------------------------------------------------------
// Calibration cache
// ---------------------------------------------------------------------------
//
// The 300s idle baseline is the most expensive part of a run and the least
// interesting to repeat: idle noise is a host property, not a session one.
// Across two Neo sessions the band moved 11% and the median idle |slope| not at
// all, so measuring it once and reusing it is a good trade -- as long as the
// cached numbers are tied to the window they were measured for, since both
// statistics depend on window length.
//
// Deliberately dumb format: one line of plain text, easy to read, delete, or
// eyeball. A cache that cannot be inspected would be a poor fit for a tool
// whose whole point is not trusting unexamined numbers.
#define kCalibMagic "thermal-probe-calib-v1"

static bool calibSave(const char *path, int winSamples, double interval, double sigma,
                      const Calib *c) {
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fprintf(f, "%s win=%d interval=%.4f sigma=%.4f rangeMedian=%.4f rangeP90=%.4f "
               "slopeMedian=%.4f slopeP90=%.4f windows=%d\n",
            kCalibMagic, winSamples, interval, sigma, c->rangeMedian, c->rangeP90,
            c->slopeMedian, c->slopeP90, c->windows);
    fclose(f);
    return true;
}

// Load only if the cached calibration was measured for this exact window and
// interval; the statistics are meaningless transplanted onto a different one.
static bool calibLoad(const char *path, int winSamples, double interval, double *sigma, Calib *c,
                      const char **why) {
    *why = NULL;
    FILE *f = fopen(path, "r");
    if (!f) {
        *why = "no cache file";
        return false;
    }
    char magic[64] = {0};
    int win = 0, windows = 0;
    double iv = 0;
    double sg = 0, rm = 0, rp = 0, sm = 0, sp = 0;
    int got = fscanf(f,
                     "%63s win=%d interval=%lf sigma=%lf rangeMedian=%lf rangeP90=%lf "
                     "slopeMedian=%lf slopeP90=%lf windows=%d",
                     magic, &win, &iv, &sg, &rm, &rp, &sm, &sp, &windows);
    fclose(f);
    if (got != 9 || strcmp(magic, kCalibMagic) != 0) {
        *why = "cache unreadable or wrong version";
        return false;
    }
    if (win != winSamples || fabs(iv - interval) > 1e-9) {
        *why = "cache was measured for a different window/interval";
        return false;
    }
    *sigma = sg;
    c->rangeMedian = rm;
    c->rangeP90 = rp;
    c->slopeMedian = sm;
    c->slopeP90 = sp;
    c->windows = windows;
    return true;
}

static void startBurn(pthread_t *th, int threads) {
    gStop = false;
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, burn, NULL);
}

// Joined, not just signalled: cycle mode starts the burn once per rep, and
// unjoined threads would pile up across reps.
static void stopBurn(pthread_t *th, int threads) {
    gStop = true;
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
}

// Mean of the last `k` samples in the ring, for the noise-robust "recovered"
// test. `next` is the write cursor.
static double tailMean(const double *win, int cap, int filled, int next, int k) {
    if (filled <= 0) return NAN;
    if (k > filled) k = filled;
    double sum = 0.0;
    for (int i = 1; i <= k; i++) sum += win[((next - i) % cap + cap) % cap];
    return sum / k;
}

// Has the die held within `band` across the whole trailing window? A window
// that isn't full yet is never steady: it holds less than the caller's
// requested seconds of history, so a flat spread across it proves nothing.
static bool dieSteady(const double *win, int cap, int filled, double band) {
    if (filled < cap) return false;
    return spreadOf(win, cap) < band;
}

// ---------------------------------------------------------------------------
// notify(3) levels
// ---------------------------------------------------------------------------

// Read a notify(3) state, or UINT64_MAX if the name can't be read at all.
static uint64_t readNotify(const char *name) {
    int token = 0;
    uint64_t state = UINT64_MAX;
    if (notify_register_check(name, &token) == NOTIFY_STATUS_OK) {
        if (notify_get_state(token, &state) != NOTIFY_STATUS_OK) state = UINT64_MAX;
        notify_cancel(token);
    }
    return state;
}

// `OSThermalPressureLevel`, macOS numbering. Mirrors `format_pressure_word`
// in macos.rs; the iOS numbering (0/10/20/30/40/50) is different.
static const char *pressureWord(uint64_t level) {
    switch (level) {
        case 0: return "nominal";
        case 1: return "moderate";
        case 2: return "heavy";
        case 3: return "trapping";
        case 4: return "sleeping";
        default: return "unknown";
    }
}

// `ProcessInfo.ThermalState` — four levels, a different scale again.
static const char *thermalStateWord(NSProcessInfoThermalState state) {
    switch (state) {
        case NSProcessInfoThermalStateNominal: return "nominal";
        case NSProcessInfoThermalStateFair: return "fair";
        case NSProcessInfoThermalStateSerious: return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
        default: return "unknown";
    }
}

// Per-source running tally across the whole run.
typedef struct {
    const char *label;
    uint64_t max;
    double firstNonZero;  // elapsed s when it first left 0, or -1
    double clearedAt;     // elapsed s when it returned to 0 after load, or -1
    bool everNonZero;
} Track;

static void trackSample(Track *t, uint64_t level, double elapsed, bool loadOver) {
    if (level == UINT64_MAX) return;  // unreadable; leave the tally alone
    if (level > t->max) t->max = level;
    if (level > 0) {
        t->everNonZero = true;
        if (t->firstNonZero < 0) t->firstNonZero = elapsed;
        t->clearedAt = -1;  // still hot; a later return to 0 is the real clear
    } else if (loadOver && t->everNonZero && t->clearedAt < 0) {
        t->clearedAt = elapsed;
    }
}

static double monotonicNow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void printTrack(const Track *t, bool isPressureScale, double loadStoppedAt) {
    printf("  %-28s %-10s max=%-3llu ", t->label,
           isPressureScale
               ? pressureWord(t->max)
               : thermalStateWord((NSProcessInfoThermalState)t->max),
           t->max);
    if (t->firstNonZero >= 0) printf("first_nonzero=%6.0fs ", t->firstNonZero);
    else printf("first_nonzero=%7s ", "never");
    if (t->clearedAt >= 0) {
        printf("cleared=%6.0fs (%.0fs after load stop)\n", t->clearedAt,
               t->clearedAt - loadStoppedAt);
    } else {
        printf("cleared=%7s\n", t->everNonZero ? "not yet" : "n/a");
    }
}

// ---------------------------------------------------------------------------
// Cycle mode: what a real benchmark batch does to the die.
// ---------------------------------------------------------------------------
//
// The question a batch actually asks is NOT "is the machine cold" but "does run
// N start where run 1 started". Cooling to idle between every run is the
// expensive way to buy that; letting the duty cycle reach its own limit cycle
// is usually far cheaper, and costs only the reps spent getting there.
//
// So this runs `reps` iterations of (load, rest) and records the die
// temperature at the moment each rep begins. If those converge, the tail of the
// batch is mutually comparable and the head should be discarded or pre-warmed.
// Convergence is judged against the measured noise floor, not a chosen number:
// two start temperatures that differ by less than the sensor's own idle wander
// are not distinguishable, so there is nothing left to wait for.

// Sample the die for `secs`, pushing into the caller's trailing ring and
// tracking a peak. Returns the hottest reading seen.
// Sample the die for `secs`, pushing into the caller's trailing ring and
// tracking a peak. Returns the hottest reading seen.
//
// Phases are timed against an ABSOLUTE deadline, not by sleeping `interval`
// after each sample. Sleeping a fixed amount after variable work accumulates
// error -- measured at 18% overrun, which turned a requested 3.5s load into
// 4.1s. For a tool whose entire job is characterizing a specific duty cycle,
// silently running a different one is fatal, and here it may have been what
// tripped the pressure enum.
static double sampleFor(double secs, double interval, const char *label, int rep, double *ring,
                        int ringCap, int *ringN, int *ringNext, double startClock) {
    double peak = NAN;
    double phase0 = monotonicNow();
    for (int i = 0;; i++) {
        if ((double)i * interval >= secs) break;
        double die = dieTempMax();
        if (!isnan(die)) {
            ring[*ringNext] = die;
            *ringNext = (*ringNext + 1) % ringCap;
            if (*ringN < ringCap) (*ringN)++;
            if (isnan(peak) || die > peak) peak = die;
        }
        uint64_t linked = readNotify(kOSThermalNotificationPressureLevelName);
        printf("%7.1fs  rep%-2d %-5s ", monotonicNow() - startClock, rep, label);
        if (isnan(die)) printf("%8s", "n/a");
        else printf("%7.2fC", die);
        printf("  pressurelevel=%llu(%s)\n", linked, pressureWord(linked));
        fflush(stdout);
        sleepSecs(phase0 + (double)(i + 1) * interval - monotonicNow());
    }
    // The phase ends when the requested duration has elapsed, not when the
    // sample count says so.
    sleepSecs(phase0 + secs - monotonicNow());
    return peak;
}

static int runCycle(int reps, double loadSecs, double restSecs, double interval, int threads,
                    double baseSecs) {
    dieTempInit();
    printf("PMU tdie sensors: %d\n", gDieCount);
    printf("cycle: %d reps of %gs load / %gs rest, threads=%d interval=%gs baseline=%gs\n\n",
           reps, loadSecs, restSecs, threads, interval, baseSecs);
    if (gDieCount <= 0) {
        fprintf(stderr, "die temp unavailable; cycle mode has nothing to measure\n");
        return 1;
    }

    int smoothN = smoothSamples(kRecoveredSmoothSecs, interval);
    int ringCap = smoothN;
    double *ring = calloc((size_t)ringCap, sizeof(double));
    int ringN = 0, ringNext = 0;
    int baseCap = (int)(baseSecs / interval) + 2;
    double *baseSamples = calloc((size_t)baseCap, sizeof(double));
    pthread_t *th = calloc((size_t)threads, sizeof(pthread_t));
    double *startTemp = calloc((size_t)reps, sizeof(double));
    double *peakTemp = calloc((size_t)reps, sizeof(double));
    if (!ring || !baseSamples || !th || !startTemp || !peakTemp) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    double clock0 = monotonicNow();
    printf("%7s  %-5s %-5s %8s  %s\n", "elapsed", "rep", "phase", "tdie_max", "pressure");

    // Baseline: only needed for the noise floor that convergence is judged
    // against, so it can be much shorter than the single-shot mode's.
    int baseN = 0;
    for (int bi = 0; (double)bi * interval < baseSecs; bi++) {
        double die = dieTempMax();
        if (!isnan(die)) {
            if (baseN < baseCap) baseSamples[baseN++] = die;
            ring[ringNext] = die;
            ringNext = (ringNext + 1) % ringCap;
            if (ringN < ringCap) ringN++;
        }
        printf("%7.1fs  %-5s %-5s ", monotonicNow() - clock0, "-", "base");
        if (isnan(die)) printf("%8s\n", "n/a");
        else printf("%7.2fC\n", die);
        fflush(stdout);
        sleepSecs(clock0 + (double)(bi + 1) * interval - monotonicNow());
    }
    double sigma = sigmaOf(baseSamples, baseN);
    double noiseFloor = isnan(sigma) ? kRecoveredFloorC : kRecoveredSigmas * sigma;
    printf("--- idle sigma %.2fC over %d samples; convergence floor %.2fC (%.0f sigma) ---\n",
           sigma, baseN, noiseFloor, kRecoveredSigmas);
    fflush(stdout);

    for (int r = 0; r < reps; r++) {
        startTemp[r] = tailMean(ring, ringCap, ringN, ringNext, smoothN);
        printf("--- rep %d starting at %.2fC ---\n", r + 1, startTemp[r]);
        fflush(stdout);
        startBurn(th, threads);
        peakTemp[r] =
            sampleFor(loadSecs, interval, "load", r + 1, ring, ringCap, &ringN, &ringNext, clock0);
        stopBurn(th, threads);
        if (restSecs > 0) {
            sampleFor(restSecs, interval, "rest", r + 1, ring, ringCap, &ringN, &ringNext, clock0);
        }
    }

    printf("\n=== cycle results ===\n");
    printf("  %-5s %-11s %-9s %s\n", "rep", "start_die", "delta", "peak");
    for (int r = 0; r < reps; r++) {
        printf("  %-5d %8.2fC   ", r + 1, startTemp[r]);
        if (r == 0) printf("%-9s", "--");
        else printf("%+8.2f ", startTemp[r] - startTemp[r - 1]);
        printf("  %.2fC\n", peakTemp[r]);
    }

    printf("\n=== cycle verdict ===\n");
    double lo = startTemp[0], hi = startTemp[0];
    for (int r = 1; r < reps; r++) {
        if (startTemp[r] < lo) lo = startTemp[r];
        if (startTemp[r] > hi) hi = startTemp[r];
    }
    printf("  start-temp spread across all %d reps: %.2fC\n", reps, hi - lo);
    // Convergence is a TREND test, not a spread test, and it is applied to the
    // peak as well as the start.
    //
    // Spread was the previous rule and it produced a confident false positive
    // on real data: reps 2-5 spanning 0.90C "inside the 1.79C noise floor",
    // while start temperature climbed +0.310 +/- 0.046 C/rep (t=6.7) and peak
    // +0.554 +/- 0.028 (t=20.1). A sustained climb spans little over four
    // points and a great deal over fifteen. Spread answers "how far apart are
    // these", never "are they still going somewhere".
    //
    // The idle noise floor is also the wrong yardstick here. It is 2 sigma of
    // single idle samples, but these are smoothed means taken at the same phase
    // of a repeating cycle, whose actual scatter was 0.10C -- 18x tighter. So
    // the test uses each tail's OWN residual scatter and asks whether the slope
    // is distinguishable from zero. Self-calibrating, no imported constant.
    //
    // Peak matters at least as much as start: the benchmark runs THROUGH the
    // excursion. On that same data the peak was diverging faster than the
    // start, and testing only the start would have missed it entirely.
    if (reps < kMinTailReps + 1) {
        printf("  [SKIP] need at least %d reps to separate a plateau from a slow climb.\n",
               kMinTailReps + 1);
    } else {
        int conv = -1;
        double cS = NAN, cSe = NAN, cR = NAN, pS = NAN, pSe = NAN, pR = NAN;
        for (int k = 0; reps - k >= kMinTailReps; k++) {
            int n = reps - k;
            double s1, e1, r1, s2, e2, r2;
            fitTrend(startTemp + k, n, &s1, &e1, &r1);
            fitTrend(peakTemp + k, n, &s2, &e2, &r2);
            double tc = tCrit95(n - 2);
            // Worst-case drift the data still permits across this tail, versus
            // the scatter already in it. Flat means "provably small", not
            // "too few points to tell".
            bool startFlat = !isnan(e1) && e1 > 0 &&
                             (fabs(s1) + tc * e1) * (n - 1) <= kDriftScatterMultiple * r1;
            bool peakFlat = !isnan(e2) && e2 > 0 &&
                            (fabs(s2) + tc * e2) * (n - 1) <= kDriftScatterMultiple * r2;
            if (startFlat && peakFlat) {
                conv = k;
                cS = s1; cSe = e1; cR = r1;
                pS = s2; pSe = e2; pR = r2;
                break;
            }
        }
        // Always show the trend over the last few reps, converged or not.
        {
            int k = reps - kMinTailReps;
            double s1, e1, r1, s2, e2, r2;
            fitTrend(startTemp + k, kMinTailReps, &s1, &e1, &r1);
            fitTrend(peakTemp + k, kMinTailReps, &s2, &e2, &r2);
            printf("  trend over last %d reps:  start %+.3f+/-%.3f C/rep (t=%.1f)   "
                   "peak %+.3f+/-%.3f (t=%.1f)\n",
                   kMinTailReps, s1, e1, e1 > 0 ? s1 / e1 : 0.0, s2, e2, e2 > 0 ? s2 / e2 : 0.0);
        }
        if (conv < 0) {
            // Report the bound over the longest tail, where the estimate is
            // strongest -- not the shortest, where it is merely unmeasurable.
            int n = reps;
            double s1, e1, r1, s2, e2, r2;
            fitTrend(startTemp, n, &s1, &e1, &r1);
            fitTrend(peakTemp, n, &s2, &e2, &r2);
            double tc = tCrit95(n - 2);
            printf("  [NOT CONVERGED] no tail is provably flat. Over all %d reps the drift is\n",
                   reps);
            printf("                  at least start %+.2fC / peak %+.2fC, and could be as much\n",
                   s1 * (n - 1), s2 * (n - 1));
            printf("                  as %+.2fC / %+.2fC (95%%) -- against scatter of %.2fC / %.2fC.\n",
                   (fabs(s1) + tc * e1) * (n - 1), (fabs(s2) + tc * e2) * (n - 1), r1, r2);
            printf("                  This duty cycle keeps accumulating heat; lengthen the rest.\n");
            printf("                  NOTE a shorter tail may LOOK flat here purely for lack of\n");
            printf("                  data -- absence of a detectable trend is not flatness.\n");
        } else {
            int n = reps - conv;
            double tc = tCrit95(n - 2);
            printf("  [CONVERGED] from rep %d: across reps %d-%d the drift is bounded at 95%% to\n",
                   conv + 1, conv + 1, reps);
            printf("              at most %.2fC (start) / %.2fC (peak), within %.0fx the\n",
                   (fabs(cS) + tc * cSe) * (n - 1), (fabs(pS) + tc * pSe) * (n - 1),
                   kDriftScatterMultiple);
            printf("              rep-to-rep scatter of %.2fC / %.2fC.\n", cR, pR);
            printf("              That scatter is the comparability you get, and it is the floor.\n");
            if (conv > 0) {
                printf("  -> discard the first %d rep%s, or pre-warm with that many throwaway runs.\n",
                       conv, conv == 1 ? "" : "s");
            } else {
                printf("  -> nothing to discard; this rest period is already long enough.\n");
            }
        }
    }

    free(ring);
    free(baseSamples);
    free(th);
    free(startTemp);
    free(peakTemp);
    return 0;
}

// Block until both coarse enums read nominal, so the run starts from a known
// state. Returns false if the cap expired first, in which case every
// downstream number is confounded by whatever the machine was already doing.
static bool waitForNominal(double maxSecs, double interval, const char *linkedName) {
    if (maxSecs <= 0) return true;  // caller opted out
    double start = monotonicNow();
    for (;;) {
        NSProcessInfoThermalState ps = [[NSProcessInfo processInfo] thermalState];
        uint64_t linked = readNotify(linkedName);
        double waited = monotonicNow() - start;
        if (ps == NSProcessInfoThermalStateNominal && linked == 0) {
            printf("--- nominal after %.0fs; starting baseline ---\n", waited);
            fflush(stdout);
            return true;
        }
        if (waited >= (double)maxSecs) {
            printf("--- still %s/%s after %.0fs cap; STARTING ANYWAY, RESULTS CONFOUNDED ---\n",
                   thermalStateWord(ps), pressureWord(linked), waited);
            fflush(stdout);
            return false;
        }
        printf("%6.0fs   wait  %d(%-8s) %llu(%-8s)  waiting for nominal start\n", waited,
               (int)ps, thermalStateWord(ps), linked, pressureWord(linked));
        fflush(stdout);
        sleepSecs(interval);
    }
}

static void usage(const char *me) {
    fprintf(stderr,
            "usage:\n"
            "  %s [baseline_secs] [load_secs] [steady_secs] [interval_secs] [threads]\n"
            "     [nominal_cap_secs] [max_cool_secs]\n"
            "        single load->cooldown run; gate evidence and the settling curve\n"
            "  %s --cycle N --load L --rest R [--interval I] [--threads T] [--baseline B]\n"
            "        repeated load/rest, to find the fastest safe gap between batch runs\n",
            me, me);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        // Cycle mode takes flags; the original single-shot interface stays
        // positional so existing invocations keep working.
        if (argc > 1 && strncmp(argv[1], "--", 2) == 0) {
            int reps = 5;
            double cLoad = 60, cRest = 45, cInterval = 2, cBase = 60;
            int cThreads = (int)[[NSProcessInfo processInfo] activeProcessorCount];
            for (int i = 1; i < argc; i++) {
                bool hasVal = i + 1 < argc;
                if (!strcmp(argv[i], "--cycle") && hasVal) reps = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--load") && hasVal) cLoad = atof(argv[++i]);
                else if (!strcmp(argv[i], "--rest") && hasVal) cRest = atof(argv[++i]);
                else if (!strcmp(argv[i], "--interval") && hasVal) cInterval = atof(argv[++i]);
                else if (!strcmp(argv[i], "--threads") && hasVal) cThreads = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--baseline") && hasVal) cBase = atof(argv[++i]);
                else {
                    fprintf(stderr, "unrecognized or incomplete option: %s\n\n", argv[i]);
                    usage(argv[0]);
                    return 2;
                }
            }
            if (cInterval < 0.02) cInterval = 0.02;
            if (reps < 1 || cLoad <= 0 || cThreads < 1) {
                fprintf(stderr, "--cycle and --threads must be >= 1, --load > 0\n");
                return 2;
            }
            return runCycle(reps, cLoad, cRest, cInterval, cThreads, cBase);
        }
        // 300s by default: the band is calibrated from windows slid over the
        // baseline, so it wants several times `steadySecs` of idle history.
        double baseSecs = argc > 1 ? atof(argv[1]) : 300;
        double loadSecs = argc > 2 ? atof(argv[2]) : 300;
        double steadySecs = argc > 3 ? atof(argv[3]) : 60;
        double interval = argc > 4 ? atof(argv[4]) : 2;
        int threads = argc > 5 ? atoi(argv[5])
                               : (int)[[NSProcessInfo processInfo] activeProcessorCount];
        double nominalCap = argc > 6 ? atof(argv[6]) : 900;
        double maxCoolSecs = argc > 7 ? atof(argv[7]) : 1800;
        // 20ms floor: below that the sampler's own cost starts to matter, and
        // IOHID event copies are not free. See the observer-effect note.
        if (interval < 0.02) interval = 0.02;
        if (steadySecs < interval) steadySecs = interval;

        dieTempInit();
        const char *linkedName = kOSThermalNotificationPressureLevelName;
        printf("linked kOSThermalNotificationPressureLevelName = %s\n", linkedName);
        printf("PMU tdie sensors: %d\n", gDieCount);
        printf("threads=%d baseline=%gs load=%gs steady=%gs interval=%gs nominal_cap=%gs "
               "max_cool=%gs\n",
               threads, baseSecs, loadSecs, steadySecs, interval, nominalCap, maxCoolSecs);
        printf("recovered = %gs smoothed mean back near baseline, threshold from noise\n",
               kRecoveredSmoothSecs);
        printf("steady    = die temp spread across the last %gs inside the p%.0f of the same\n",
               steadySecs, kBandPercentile * 100);
        printf("            window slid over the idle baseline (floor %.2fC, no ceiling)\n",
               kBandFloorC);
        printf("cooldown ends when recovered AND pressurelevel nominal, or at the cap\n");
        if (baseSecs < 3 * steadySecs) {
            printf("NOTE: baseline %gs is under 3x the %gs steady window, so it cannot calibrate\n",
                   baseSecs, steadySecs);
            printf("      a band itself -- it will use $THERMAL_PROBE_CALIB if that holds a\n");
            printf("      matching calibration, else fall back to the weaker sigma estimate.\n");
        }
        printf("gate DEFAULT_MAX_WAIT = %ds\n\n", kGateDefaultMaxWaitSecs);

        bool cleanStart = waitForNominal(nominalCap, interval, linkedName);

        printf("\n%7s %6s  %-11s %-11s %8s  %-9s %s\n", "elapsed", "phase", "ProcessInfo",
               "pressurelevel", "tdie_max", "pressure", "bogus");

        Track tProcInfo = {"ProcessInfo.thermalState", 0, -1, -1, false};
        Track tLinked = {"pressurelevel (linked)", 0, -1, -1, false};
        Track tUndoc = {"pressure (undocumented)", 0, -1, -1, false};
        Track tBogus = {"bogus control", 0, -1, -1, false};

        double baseSum = 0.0;
        int baseCount = 0;
        double baseline = NAN;
        double diePeak = NAN;
        double dieRecoveredAt = -1.0;
        double dieAtLinkedClear = NAN;
        double loadStoppedAt = NAN;

        double dieSteadyAt = -1.0;   // onset of the CURRENT steady stretch, or -1
        double dieFirstSteadyAt = -1.0;  // first onset ever, even if it broke
        int steadyBreaks = 0;            // how many times it fell back out
        bool capHit = false;
        // Derived from the baseline at load start; until then no cooldown check
        // runs, so the value is unused.
        double steadyBand = kBandFallbackC;
        double slopeBand = NAN;  // C/min; NAN disables the trend test
        double baseSpread = NAN, baseSigma = NAN, baseSlope = NAN, baseLag1 = NAN;
        Calib cal = {NAN, NAN, NAN, NAN, 0};
        int baseStatN = 0;  // baseCount clamped to what baseSamples actually holds
        const char *bandNote = NULL;
        // Opt-in noise cache: export THERMAL_PROBE_CALIB=<path> to let a short
        // baseline borrow an earlier long one's calibration on this host.
        const char *calibPath = getenv("THERMAL_PROBE_CALIB");
        const char *calibNote = NULL;
        // Also derived from baseline noise; see kRecoveredSmoothSecs.
        double recoveredWithin = kRecoveredFloorC;

        pthread_t *th = calloc((size_t)threads, sizeof(pthread_t));
        // Trailing window of cooldown die samples, oldest overwritten. Sized to
        // span at least steadySecs so a full window is steadySecs of history.
        // ceil, not integer-division trickery: both are doubles now.
        int winSamples = (int)ceil(steadySecs / interval) + 1;
        double *win = calloc((size_t)winSamples, sizeof(double));
        int winNext = 0, winFilled = 0;
        // Every baseline die sample, for the noise statistics.
        int baseCap = (int)(baseSecs / interval) + 2;
        double *baseSamples = calloc((size_t)baseCap, sizeof(double));
        double *calScratch = calloc((size_t)baseCap * 2, sizeof(double));
        // Whole cooldown trace, for the settling curve. The cap bounds it.
        int coolCap = (int)(maxCoolSecs / interval) + 4;
        double *coolT = calloc((size_t)coolCap, sizeof(double));
        double *coolV = calloc((size_t)coolCap, sizeof(double));
        int coolN = 0;
        if (!th || !win || !baseSamples || !calScratch || !coolT || !coolV) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }

        int smoothN = smoothSamples(kRecoveredSmoothSecs, interval);
        double start = monotonicNow();
        bool loadStarted = false, loadOver = false;
        // No fixed end: the cooldown decides when to stop, bounded by maxCoolSecs.
        for (double e = 0;; e += interval) {
            double elapsed = monotonicNow() - start;

            if (!loadStarted && e >= baseSecs) {
                baseline = baseCount > 0 ? baseSum / baseCount : NAN;
                baseStatN = baseCount < baseCap ? baseCount : baseCap;
                baseSpread = spreadOf(baseSamples, baseStatN);
                baseSigma = sigmaOf(baseSamples, baseStatN);
                baseSlope = slopeOf(baseSamples, baseStatN, interval);
                calibrate(baseSamples, baseStatN, winSamples, interval, calScratch, &cal);
                // A baseline long enough to calibrate is authoritative and
                // refreshes the cache; a short one borrows the last good
                // measurement from this host. Only the NOISE is cached -- the
                // baseline mean stays session-local, because ambient moves and
                // `recovered` is measured against it.
                if (cal.windows >= kMinCalibWindows) {
                    if (calibPath && calibSave(calibPath, winSamples, interval, baseSigma, &cal))
                        calibNote = "measured now; cache refreshed";
                } else if (calibPath) {
                    const char *why = NULL;
                    double cachedSigma = NAN;
                    Calib cached = {NAN, NAN, NAN, NAN, 0};
                    if (calibLoad(calibPath, winSamples, interval, &cachedSigma, &cached, &why)) {
                        cal = cached;
                        baseSigma = cachedSigma;
                        calibNote = "noise from cache; this baseline was too short";
                    } else {
                        calibNote = why;
                    }
                }
                steadyBand = deriveSteadyBand(&cal, baseSigma, baseStatN, &bandNote);
                if (cal.windows >= kMinCalibWindows) slopeBand = cal.slopeP90;
                if (!isnan(baseSigma)) {
                    // Divide by the EFFECTIVE sample count. Dividing by
                    // sqrt(smoothN) assumes independent samples; at lag-1 +0.68
                    // five of them are worth about one, so that version claimed
                    // roughly twice the resolution the data supports.
                    baseLag1 = lag1Of(baseSamples, baseStatN);
                    double nEff = effectiveN(smoothN, baseLag1);
                    double fromNoise = kRecoveredSigmas * baseSigma / sqrt(nEff);
                    recoveredWithin =
                        fromNoise > kRecoveredFloorC ? fromNoise : kRecoveredFloorC;
                }
                for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, burn, NULL);
                loadStarted = true;
                printf("--- baseline die temp %.2fC, noise sigma %.2fC (spread %.2fC) over %d "
                       "samples; steady band %.2fC%s%s ---\n",
                       baseline, baseSigma, baseSpread, baseStatN, steadyBand,
                       bandNote ? " -- " : "", bandNote ? bandNote : "");
                // Drift means the baseline was taken while the die was still
                // moving, so every figure derived from it is inflated. Say so
                // before the run continues on top of it.
                if (!isnan(baseSlope) && !isnan(baseSigma) && baseSigma > 0.0) {
                    double totalDrift = fabs(baseSlope) * (baseStatN * interval) / 60.0;
                    if (totalDrift > kDriftSigmas * baseSigma) {
                        printf("--- [DRIFT] baseline moved %.2fC (%.2f C/min) over its own "
                               "%.2fC sigma:\n",
                               totalDrift, baseSlope, baseSigma);
                        printf("---         it was still settling, so the band and the baseline "
                               "mean are\n");
                        printf("---         both biased. `nominal` does not mean settled. "
                               "Re-run from idle. ---\n");
                    }
                }
                printf("--- load started at %.0fs ---\n", elapsed);
                fflush(stdout);
            }
            if (loadStarted && !loadOver && e >= baseSecs + loadSecs) {
                gStop = true;
                loadOver = true;
                loadStoppedAt = elapsed;
                printf("--- load stopped at %.0fs; cooling ---\n", elapsed);
                fflush(stdout);
            }

            NSProcessInfoThermalState ps = [[NSProcessInfo processInfo] thermalState];
            uint64_t linked = readNotify(linkedName);
            uint64_t undoc = readNotify(kUndocumentedName);
            uint64_t bogus = readNotify(kBogusName);
            double die = dieTempMax();

            if (!loadStarted && !isnan(die)) {
                baseSum += die;
                if (baseCount < baseCap) baseSamples[baseCount] = die;
                baseCount++;
            }
            if (!isnan(die) && (isnan(diePeak) || die > diePeak)) diePeak = die;
            if (loadOver && !isnan(die)) {
                win[winNext] = die;
                winNext = (winNext + 1) % winSamples;
                if (winFilled < winSamples) winFilled++;
                if (coolN < coolCap) {
                    coolT[coolN] = elapsed;
                    coolV[coolN] = die;
                    coolN++;
                }
            }
            // Recovery is judged on the smoothed tail, so it must come after the
            // window push. A single sample carries the full noise -- on a host
            // with 3.79C of idle wander one dip means nothing.
            if (loadOver && !isnan(baseline) && dieRecoveredAt < 0 &&
                winFilled >= smoothN) {
                double smoothed = tailMean(win, winSamples, winFilled, winNext, smoothN);
                if (!isnan(smoothed) && smoothed <= baseline + recoveredWithin) {
                    dieRecoveredAt = elapsed;
                }
            }

            bool linkedWasClear = tLinked.clearedAt >= 0;
            trackSample(&tProcInfo, (uint64_t)ps, elapsed, loadOver);
            trackSample(&tLinked, linked, elapsed, loadOver);
            trackSample(&tUndoc, undoc, elapsed, loadOver);
            trackSample(&tBogus, bogus, elapsed, loadOver);
            if (!linkedWasClear && tLinked.clearedAt >= 0) dieAtLinkedClear = die;

            printf("%6.0fs %6s  %d(%-8s) %llu(%-8s) ", elapsed,
                   !loadStarted ? "base" : (loadOver ? "cool" : "load"), (int)ps,
                   thermalStateWord(ps), linked, pressureWord(linked));
            if (isnan(die)) printf("%8s  ", "n/a");
            else printf("%7.2fC  ", die);
            printf("%-9llu %llu\n", undoc, bogus);
            fflush(stdout);

            if (loadOver) {
                // `steady` is tracked for comparison but no longer gates the
                // exit -- an honest trend test needs a window too long to be
                // the fastest signal. `recovered` does the gating.
                bool steady = false;
                if (gDieCount > 0 && dieSteady(win, winSamples, winFilled, steadyBand)) {
                    // Range says flat; now ask whether it was actually moving.
                    double trend = winFilled >= winSamples
                                       ? slopeOf(win, winSamples, interval)
                                       : NAN;
                    steady = isnan(slopeBand) || isnan(trend) || fabs(trend) <= slopeBand;
                }
                if (!steady) {
                    // Fell back out -- the settle clock restarts rather than
                    // standing as a claim the run went on to contradict. Say so
                    // in the log, or a bare sequence of "die steady" lines with
                    // no retraction reads as though it held.
                    if (dieSteadyAt >= 0) {
                        steadyBreaks++;
                        printf("--- die left the steady band at %.0fs (%.0fs after load stop); "
                               "settle clock restarts ---\n",
                               elapsed, elapsed - loadStoppedAt);
                        fflush(stdout);
                    }
                    dieSteadyAt = -1.0;
                } else if (gDieCount > 0 && dieSteadyAt < 0) {
                    dieSteadyAt = elapsed;
                    if (dieFirstSteadyAt < 0) dieFirstSteadyAt = elapsed;
                    printf("--- die steady within %.2fC for %gs at %.0fs (%.0fs after load "
                           "stop) ---\n",
                           steadyBand, steadySecs, elapsed, elapsed - loadStoppedAt);
                    fflush(stdout);
                }
                // Without die sensors recovery is unknowable, so the enum alone
                // ends the cooldown (still capped).
                bool recovered = gDieCount > 0 ? dieRecoveredAt >= 0 : true;
                if (recovered && linked == 0) {
                    printf("--- cooldown done at %.0fs (%.0fs after load stop): die recovered, "
                           "pressurelevel nominal ---\n",
                           elapsed, elapsed - loadStoppedAt);
                    fflush(stdout);
                    break;
                }
                if (elapsed - loadStoppedAt >= (double)maxCoolSecs) {
                    capHit = true;
                    printf("--- cooldown hit its %gs cap at %.0fs; ending ---\n", maxCoolSecs,
                           elapsed);
                    fflush(stdout);
                    break;
                }
            }

            // Absolute deadline: see sampleFor.
            sleepSecs(start + e + interval - monotonicNow());
        }
        gStop = true;

        printf("\n=== summary ===\n");
        printTrack(&tProcInfo, false, loadStoppedAt);
        printTrack(&tLinked, true, loadStoppedAt);
        printTrack(&tUndoc, true, loadStoppedAt);
        printTrack(&tBogus, true, loadStoppedAt);
        if (gDieCount > 0) {
            printf("  %-28s baseline=%.2fC peak=%.2fC ", "die temp (PMU tdie max)", baseline,
                   diePeak);
            if (dieRecoveredAt >= 0)
                printf("recovered=%.0fs (%.0fs after load stop)\n", dieRecoveredAt,
                       dieRecoveredAt - loadStoppedAt);
            else printf("recovered=not yet\n");
            printf("  %-28s sigma=%.2fC spread=%.2fC trend=%+.2f C/min over %d idle samples\n",
                   "die temp noise (baseline)", baseSigma, baseSpread, baseSlope, baseStatN);
            // Autocorrelation decides whether a faster interval buys precision
            // or only time resolution. Printed because it is the number that
            // makes an oversampled mean look better than it is.
            if (!isnan(baseLag1)) {
                double corrSecs = baseLag1 > 0 && baseLag1 < 1 ? -interval / log(baseLag1) : NAN;
                printf("  %-28s lag-1 %+.2f at %gs spacing", "noise autocorrelation", baseLag1,
                       interval);
                if (!isnan(corrSecs)) printf(" -> correlation time ~%.1fs", corrSecs);
                printf("\n");
                printf("  %-28s %d samples over %gs are worth %.1f independent ones\n", "",
                       smoothN, kRecoveredSmoothSecs, effectiveN(smoothN, baseLag1));
            }
            // The calibration is the headline: it says what the steadiness test
            // would have read on data known to be idle.
            if (cal.windows > 0) {
                printf("  %-28s range med=%.2fC p%.0f=%.2fC | |slope| med=%.2f p%.0f=%.2f C/min"
                       " (%d windows)\n",
                       "idle window stats", cal.rangeMedian, kBandPercentile * 100, cal.rangeP90,
                       cal.slopeMedian, kBandPercentile * 100, cal.slopeP90, cal.windows);
            } else {
                printf("  %-28s baseline holds no full %d-sample window\n", "idle window stats",
                       winSamples);
            }
            if (calibNote) printf("  %-28s %s\n", "calibration source", calibNote);
            printf("  %-28s %.2fC", "steady band (derived)", steadyBand);
            // Only claim the calibration when the calibration produced it; a
            // floored or fallback band came from somewhere else.
            if (bandNote) printf("  [%s]", bandNote);
            else printf(" = p%.0f of idle window range", kBandPercentile * 100);
            printf("\n");
            printf("  %-28s ", "slope band (derived)");
            if (!isnan(slopeBand))
                printf("%.2f C/min = p%.0f of idle window |slope|\n", slopeBand,
                       kBandPercentile * 100);
            else printf("disabled -- no calibration, range test only\n");
            printf("  %-28s tested over a %d-sample window (%gs)\n", "", winSamples, steadySecs);
            printf("  %-28s within %.2fC of baseline on a %d-sample mean\n",
                   "recovered threshold", recoveredWithin, smoothN);
            printf("  %-28s ", "die temp steady");
            if (dieSteadyAt >= 0) {
                printf("held from %.0fs (%.0fs after load stop)", dieSteadyAt,
                       dieSteadyAt - loadStoppedAt);
                if (steadyBreaks > 0) printf(", after %d false start(s)", steadyBreaks);
                printf("\n");
            } else if (dieFirstSteadyAt >= 0) {
                // It did settle, transiently. Saying "never" here would be a
                // flat contradiction of the run log above it.
                printf("reached at %.0fs (%.0fs after load stop) but did NOT hold -- broke %d "
                       "time(s)\n",
                       dieFirstSteadyAt, dieFirstSteadyAt - loadStoppedAt, steadyBreaks);
            } else {
                printf("never settled within %.2fC for %gs\n", steadyBand, steadySecs);
            }
        }

        if (gDieCount > 0 && !isnan(loadStoppedAt)) {
            printSettlingCurve(coolT, coolV, coolN, loadStoppedAt, baseSigma, smoothN);
        }

        printf("\n=== verdict ===\n");
        if (!cleanStart) {
            printf("  [CONFOUNDED] the run did not start from nominal, so recovery times\n");
            printf("               include the tail of an earlier thermal event. Treat every\n");
            printf("               number below as an upper bound and re-run from idle.\n");
        }
        if (capHit) {
            printf("  [PARTIAL] the cooldown ended on its %gs cap, not on the exit condition:\n",
                   maxCoolSecs);
            printf("            die %s, pressurelevel %s. Raise max_cool_secs to see the rest.\n",
                   dieSteadyAt >= 0 ? "steady" : "still drifting",
                   (tLinked.clearedAt >= 0 || !tLinked.everNonZero) ? "nominal"
                                                                    : "still non-nominal");
        }
        if (tBogus.everNonZero) {
            printf("  [BAD]  the bogus control went non-zero -- results are untrustworthy.\n");
        }
        // Replaces the old a-priori band ceiling. A band is only "too wide" in
        // relation to the thermal swing it has to resolve, and that is not
        // known until the load has run -- clamping at a fixed temperature
        // before seeing the rise is what broke the Neo run.
        if (gDieCount > 0 && !isnan(diePeak) && !isnan(baseline) && diePeak - baseline > 0.0) {
            double rise = diePeak - baseline;
            if (steadyBand > 0.5 * rise) {
                printf("  [WEAK] the steady band (%.2fC) is %.0f%% of the load-induced rise\n",
                       steadyBand, 100.0 * steadyBand / rise);
                printf("         (%.2fC), so 'steady' resolves very little on this run. Either\n",
                       rise);
                printf("         the load was too small, or this host's idle noise is too large\n");
                printf("         for a die-trend gate to mean much.\n");
            }
        }
        if (!tLinked.everNonZero && !tProcInfo.everNonZero) {
            printf("  [INCONCLUSIVE] neither enum left nominal: this machine never got hot\n");
            printf("                 enough. Raise the load duration, or use a fanless host.\n");
            if (!isnan(diePeak) && !isnan(baseline)) {
                printf("                 die temp did rise %.1fC (%.2f -> %.2f), so the load\n",
                       diePeak - baseline, baseline, diePeak);
                printf("                 landed -- the enums simply never tripped.\n");
            }
            free(th);
            free(win);
            free(baseSamples);
            free(calScratch);
            free(coolT);
            free(coolV);
            return 0;
        }
        if (!tLinked.everNonZero && tProcInfo.everNonZero) {
            printf("  [FAIL] ProcessInfo reported pressure (max %llu) but pressurelevel stayed 0.\n",
                   tProcInfo.max);
            printf("         The notify read is not tracking; revert macos.rs to ProcessInfo.\n");
            free(th);
            free(win);
            free(baseSamples);
            free(calScratch);
            free(coolT);
            free(coolV);
            return 0;
        }

        printf("  [OK]   pressurelevel reached %llu (%s) -- the notify signal is live.\n",
               tLinked.max, pressureWord(tLinked.max));
        if (tLinked.clearedAt >= 0 && tProcInfo.clearedAt >= 0) {
            double delta = tProcInfo.clearedAt - tLinked.clearedAt;
            if (delta > (double)interval) {
                printf("  [OK]   pressurelevel cleared %.0fs before ProcessInfo -- Foundation does\n",
                       delta);
                printf("         lag on recovery after all; the gate change shortens cooldowns.\n");
            } else if (delta < -(double)interval) {
                printf("  [NOTE] ProcessInfo cleared %.0fs BEFORE pressurelevel -- the gate is now\n",
                       -delta);
                printf("         stricter, not faster.\n");
            } else {
                printf("  [NOTE] both cleared within one sample interval: lock-step, no latency\n");
                printf("         win from the notify switch (only the removed compiler spawn).\n");
            }
        }

        // Does this host even fit inside the gate's deadline?
        if (tLinked.clearedAt >= 0 && !isnan(loadStoppedAt)) {
            double sinceStop = tLinked.clearedAt - loadStoppedAt;
            if (sinceStop > (double)kGateDefaultMaxWaitSecs) {
                printf("  [CRITICAL] pressurelevel took %.0fs after load stop to clear, over the\n",
                       sinceStop);
                printf("             %ds DEFAULT_MAX_WAIT. The gate does not merely wait here --\n",
                       kGateDefaultMaxWaitSecs);
                printf("             it times out and fails the cell. Raise the deadline, or gate\n");
                printf("             on something that tracks the hardware.\n");
            } else {
                printf("  [OK]   pressurelevel cleared %.0fs after load stop, inside the %ds\n",
                       sinceStop, kGateDefaultMaxWaitSecs);
                printf("         DEFAULT_MAX_WAIT.\n");
            }
        } else if (tLinked.everNonZero && tLinked.clearedAt < 0) {
            printf("  [CRITICAL] pressurelevel never cleared before the run ended. With a %ds\n",
                   kGateDefaultMaxWaitSecs);
            printf("             DEFAULT_MAX_WAIT the gate would have timed out and failed the\n");
            printf("             cell. Raise max_cool_secs to find the real recovery time.\n");
        }

        // The over-wait this run exists to quantify.
        if (gDieCount <= 0) {
            printf("  [SKIP] die temp unavailable, so the over-wait question is unanswered.\n");
        } else if (dieRecoveredAt < 0) {
            printf("  [PARTIAL] die temp never returned within %.2fC of baseline (%.2fC) before\n",
                   recoveredWithin, baseline);
            printf("            the run ended -- raise max_cool_secs to measure the over-wait.\n");
        } else if (tLinked.clearedAt < 0) {
            printf("  [STRONG] die recovered %.0fs after load stop but pressurelevel never\n",
                   dieRecoveredAt - loadStoppedAt);
            printf("           cleared. The gate would still be waiting on a cool SoC.\n");
        } else {
            double overWait = tLinked.clearedAt - dieRecoveredAt;
            printf("         die recovered %.0fs after load stop; pressurelevel %.0fs after",
                   dieRecoveredAt - loadStoppedAt, tLinked.clearedAt - loadStoppedAt);
            if (!isnan(dieAtLinkedClear)) printf(" (die %.2fC then)", dieAtLinkedClear);
            printf(".\n");
            if (overWait > (double)interval) {
                printf("  [STRONG] the gate over-waits %.0fs past a recovered SoC -- %.0fx longer\n",
                       overWait, (tLinked.clearedAt - loadStoppedAt) /
                                     fmax(1.0, dieRecoveredAt - loadStoppedAt));
                printf("           than the hardware needed. That is what gating on PMU tdie\n");
                printf("           would recover per cooldown.\n");
                // dieAtLinkedClear is one raw reading, not a smoothed mean, so
                // it needs a single-sample allowance rather than the tighter
                // threshold calibrated for the 5-sample tail.
                double rawAllowance = recoveredWithin;
                if (!isnan(baseSigma) && kRecoveredSigmas * baseSigma > rawAllowance)
                    rawAllowance = kRecoveredSigmas * baseSigma;
                if (!isnan(dieAtLinkedClear) && !isnan(baseline) &&
                    dieAtLinkedClear <= baseline + rawAllowance) {
                    printf("  [NOTE] the die was already at baseline when the enum finally cleared,\n");
                    printf("         so the enum is on a timer, not a temperature threshold --\n");
                    printf("         cooling the machine harder will not clear it sooner.\n");
                }
            } else if (overWait < -(double)interval) {
                printf("  [NOTE] pressurelevel cleared %.0fs BEFORE the die recovered -- the enum\n",
                       -overWait);
                printf("         is the *looser* gate here, so switching to die temp would make\n");
                printf("         cooldowns longer, not shorter. Worth knowing before changing.\n");
            } else {
                printf("  [NOTE] the enum and the die agree within one sample interval. No\n");
                printf("         over-wait to recover; leave the gate on the notify level.\n");
            }
        }
        free(th);
        free(win);
        free(baseSamples);
        free(calScratch);
        free(coolT);
        free(coolV);
    }
    return 0;
}
