//! Cross-server flavor translation.
//!
//! A leaf module with no in-crate imports, so the modules that need the
//! mapping can reach it without depending on each other.

use pipette_plan_types::{SglangFlavor, VllmFlavor};

/// Lives here because plan-types keeps the flavor enums distinct. Total while
/// the variant sets match; a server-specific variant would make it fallible.
pub(crate) fn sglang_to_vllm_flavor(f: SglangFlavor) -> VllmFlavor {
    match f {
        SglangFlavor::NvidiaGpu => VllmFlavor::NvidiaGpu,
        SglangFlavor::AmdGpu => VllmFlavor::AmdGpu,
        SglangFlavor::Cpu => VllmFlavor::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(SglangFlavor::NvidiaGpu, VllmFlavor::NvidiaGpu)]
    #[case(SglangFlavor::AmdGpu, VllmFlavor::AmdGpu)]
    #[case(SglangFlavor::Cpu, VllmFlavor::Cpu)]
    fn sglang_flavor_translates_to_its_vllm_counterpart(
        #[case] sglang: SglangFlavor,
        #[case] vllm: VllmFlavor,
    ) {
        assert_eq!(sglang_to_vllm_flavor(sglang), vllm);
    }
}
