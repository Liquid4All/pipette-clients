//! The `/proc` terms behind Android's and Linux's swap-aware peak
//! ([`pipette_memprobe::Peak::counting_swap`]), for the memory cells' rows.

use pipette_memprobe::Footprint;

/// The `/proc` terms behind the peak, for the row's `client_extras`.
pub fn extras(footprint: &Footprint) -> serde_json::Map<String, serde_json::Value> {
    serde_json::Map::from_iter([
        ("peak_rss_kib".to_owned(), footprint.peak_rss_kib.into()),
        (
            "peak_committed_kib".to_owned(),
            footprint.peak_committed_kib.into(),
        ),
        ("max_swap_kib".to_owned(), footprint.max_swap_kib.into()),
        ("major_faults".to_owned(), footprint.major_faults.into()),
    ])
}
