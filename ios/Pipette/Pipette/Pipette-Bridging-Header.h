#ifndef PIPETTE_BRIDGING_HEADER_H
#define PIPETTE_BRIDGING_HEADER_H

#include <stdint.h>

// Native ObjC probe compiled into the app target (Native/PipetteThermal.m),
// formerly provided by the Rust crate's metal_counter.m. Exposed to Swift.
// (Metal allocation is read in Swift via MTLDevice.currentAllocatedSize.)

/// Max SoC die temperature in C, or -1 when unavailable. Gated by the
/// PIPETTE_PRIVATE_THERMAL build flag in the implementation.
double pipette_soc_temp(void);

/// 1 when this build compiled in the private read above, 0 otherwise. The runtime
/// identity reports it, so a plan can require the gated build and a stock one refuses
/// the cell rather than measuring with a coarser gate.
int pipette_private_thermal_build(void);

#endif /* PIPETTE_BRIDGING_HEADER_H */
