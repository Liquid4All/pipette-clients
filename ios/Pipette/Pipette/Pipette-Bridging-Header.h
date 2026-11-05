#ifndef PIPETTE_BRIDGING_HEADER_H
#define PIPETTE_BRIDGING_HEADER_H

#include <stdint.h>

// The ONNX Runtime GenAI Objective-C surface, published into Generated/OnnxGenai by
// ios/build-onnx-genai.sh.
//
// Foundation first: `ort_genai_objc.h` uses `NS_ENUM`, `NSInteger` and
// `NS_ASSUME_NONNULL_BEGIN` without importing anything that defines them (upstream relies
// on a CocoaPods umbrella header). Imported bare it parses as C and every declaration
// collapses to implicit `int`.
#import <Foundation/Foundation.h>
#import "ort_genai_objc.h"

// Not in the `llama` module: Native/PipetteLlama.c includes it as text, and the module
// would read its definitions twice.
#include "pipette_llama.h"

// Implemented in Native/PipetteThermal.m.

/// Max SoC die temperature in C, or -1 when unavailable. Gated by the
/// PIPETTE_PRIVATE_THERMAL build flag in the implementation.
double pipette_soc_temp(void);

/// Every IOHID temperature sensor as `Product` and `TemperatureLevel` (C), or nil
/// without the private read.
NSArray<NSDictionary<NSString *, id> *> *_Nullable pipette_temperature_sensors(void);

/// 1 when this build compiled in the private read above, 0 otherwise. The runtime
/// identity reports it, so a plan can require the gated build and a stock one refuses
/// the cell rather than measuring with a coarser gate.
int pipette_private_thermal_build(void);

// Native/PipetteClocks.m: IOReport CPU-cluster and GPU clock-state residency, for the log.
// Gated by PIPETTE_PRIVATE_THERMAL like the reads above.

/// A retained sample of the clock channels, or NULL without the private read.
const void *_Nullable pipette_clock_sample(void);

/// Releases a sample from `pipette_clock_sample`; NULL is fine.
void pipette_clock_release(const void *_Nullable sample);

/// Each channel's share of time per clock state between two samples, or nil.
NSString *_Nullable pipette_clock_residency(const void *_Nullable start, const void *_Nullable end);

#endif /* PIPETTE_BRIDGING_HEADER_H */
