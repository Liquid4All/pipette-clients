import Foundation

/// Max SoC die temperature in °C, or nil. `pipette_soc_temp` (gated by
/// `PIPETTE_PRIVATE_THERMAL`) answers a non-positive sentinel when unavailable; it stops
/// here so no caller mistakes it for a temperature.
nonisolated func dieTempMaxC() -> Double? {
    let celsius = pipette_soc_temp()
    return celsius > 0 ? celsius : nil
}
