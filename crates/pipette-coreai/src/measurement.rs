//! Core AI's conversion from the sidecar's reported rate to a span.

/// The span, in milliseconds, that produced `tokens` at `tps` tokens per second.
pub fn time_ms_from_tps(metric: &str, tokens: u32, tps: f64) -> anyhow::Result<f64> {
    Ok(f64::from(tokens) / pipette_ops::measurement::positive_finite(metric, tps)? * 1000.0)
}
