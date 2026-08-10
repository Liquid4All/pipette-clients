//! Which Linux board are we on — the piece `cfg!` can't give us, since the
//! same `aarch64-unknown-linux-gnu` binary runs on a Pi 5, a Jetson, or a
//! generic ARM box.

use std::sync::OnceLock;

/// Identified board / SoC class. Add a variant + detection only when a
/// readiness path needs to branch on it.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Board {
    /// Raspberry Pi 5 (BCM2712).
    RaspberryPi5,
    /// Unrecognized — the generic path.
    Other,
}

impl Board {
    /// Detected once per process. A failed read resolves to [`Board::Other`],
    /// so caching is what keeps every rep in a run gated by the same
    /// criteria — re-detecting per call would let one unreadable probe drop a
    /// single rep to the generic gate and average it in with the rest.
    pub(super) fn current() -> &'static Board {
        static CELL: OnceLock<Board> = OnceLock::new();
        CELL.get_or_init(|| detect_board(&read_nul_list("/proc/device-tree/compatible")))
    }
}

/// The device-tree `compatible` node is the kernel's canonical board/SoC
/// identifier and is always present on a booted Pi, so it alone decides:
/// Pi 5 is `raspberrypi,5-*` (board) on the `brcm,bcm2712` SoC. (cpuinfo
/// revision / model-string heuristics would only restate the same fact.)
fn detect_board(dt_compatible: &[String]) -> Board {
    let is_pi5 = dt_compatible
        .iter()
        .any(|c| c.starts_with("raspberrypi,5") || c == "brcm,bcm2712");
    if is_pi5 {
        Board::RaspberryPi5
    } else {
        Board::Other
    }
}

/// e.g. `["raspberrypi,5-model-b", "brcm,bcm2712"]`. A missing file yields
/// an empty list, which detection reads as [`Board::Other`].
fn read_nul_list(path: &str) -> Vec<String> {
    std::fs::read(path)
        .ok()
        .map(|bytes| {
            bytes
                .split(|&b| b == 0)
                .filter(|chunk| !chunk.is_empty())
                .map(|chunk| String::from_utf8_lossy(chunk).trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(&["raspberrypi,5-model-b", "brcm,bcm2712"], Board::RaspberryPi5)]
    #[case(&["brcm,bcm2712"], Board::RaspberryPi5)] // SoC compatible alone is sufficient
    #[case(&["raspberrypi,4-model-b"], Board::Other)] // other Pis fall to the generic path
    #[case(&[], Board::Other)] // missing device tree
    fn detect_board_from_compatible(#[case] compatible: &[&str], #[case] want: Board) {
        let dt: Vec<String> = compatible.iter().map(|s| s.to_string()).collect();
        assert_eq!(detect_board(&dt), want);
    }
}
