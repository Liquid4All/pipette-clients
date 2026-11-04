//! Which Linux board this is: `cfg!` cannot tell, since the same
//! `aarch64-unknown-linux-gnu` binary runs on a Pi 5, a Jetson, or a generic
//! ARM box.

use std::sync::OnceLock;

use crate::ReadinessPlatform;

/// Add a variant only when a readiness path needs to branch on it.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Board {
    /// Raspberry Pi 5 (BCM2712).
    RaspberryPi5,
    /// NVIDIA Jetson Orin (Tegra234).
    JetsonOrin,
    Other,
}

impl Board {
    /// Detected once per process: a failed read resolves to [`Board::Other`],
    /// so re-detecting could drop a single rep to the generic gate.
    pub(super) fn current() -> &'static Board {
        static CELL: OnceLock<Board> = OnceLock::new();
        CELL.get_or_init(|| detect_board(&read_nul_list("/proc/device-tree/compatible")))
    }

    pub(super) fn platform(&self) -> ReadinessPlatform {
        match self {
            Self::RaspberryPi5 => ReadinessPlatform::RaspberryPi5,
            Self::JetsonOrin => ReadinessPlatform::JetsonOrin,
            Self::Other => ReadinessPlatform::Linux,
        }
    }
}

/// The device-tree `compatible` node is the kernel's canonical board and SoC
/// identifier and always present on a booted Pi, so it alone decides: a Pi 5
/// is `raspberrypi,5-*` on the `brcm,bcm2712` SoC, a Jetson Orin the
/// `nvidia,tegra234` SoC. Xavier (`tegra194`) and Thor (`tegra264`) have other
/// throttle wiring and stay generic.
fn detect_board(dt_compatible: &[String]) -> Board {
    let is_pi5 = dt_compatible
        .iter()
        .any(|c| c.starts_with("raspberrypi,5") || c == "brcm,bcm2712");
    if is_pi5 {
        Board::RaspberryPi5
    } else if dt_compatible.iter().any(|c| c == "nvidia,tegra234") {
        Board::JetsonOrin
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
    #[case(&["nvidia,p3737-0000+p3701-0005", "nvidia,p3701-0005", "nvidia,tegra234"], Board::JetsonOrin)]
    #[case(&["nvidia,tegra194"], Board::Other)] // Xavier
    #[case(&["nvidia,tegra264"], Board::Other)] // Thor
    #[case(&[], Board::Other)] // missing device tree
    fn detect_board_from_compatible(#[case] compatible: &[&str], #[case] want: Board) {
        let dt: Vec<String> = compatible.iter().map(|s| s.to_string()).collect();
        assert_eq!(detect_board(&dt), want);
    }

    #[rstest]
    #[case(Board::RaspberryPi5, ReadinessPlatform::RaspberryPi5)]
    #[case(Board::JetsonOrin, ReadinessPlatform::JetsonOrin)]
    #[case(Board::Other, ReadinessPlatform::Linux)]
    fn each_board_names_its_gate(#[case] board: Board, #[case] want: ReadinessPlatform) {
        assert_eq!(board.platform(), want);
    }
}
