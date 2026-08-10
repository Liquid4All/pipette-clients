//! Cross-server flavor translation.
//!
//! Leaf module: plan-types in, plan-types out, no in-crate imports, so the
//! modules that need the mapping can reach it without depending on each other.
//!
//! Only the vllm↔sglang mapping lives here. Classifying a uv build tag into a
//! flavor is [`pipette_plan_types::flavor_from_uv_build`] — the installer needs
//! the same answer, and it sits upstream of this crate.

use pipette_plan_types::{SglangFlavor, VllmFlavor};

/// Translate a [`SglangFlavor`] to its [`VllmFlavor`] counterpart.
/// Lives in torch-oai because plan-types deliberately keeps the
/// flavor enums distinct; cross-server conversion is a torch-oai
/// concern. Total today because the variant sets match; if plan-types
/// adds a server-specific variant, this function becomes fallible and
/// the compiler flags the non-exhaustive match.
pub(crate) fn sglang_to_vllm_flavor(f: SglangFlavor) -> VllmFlavor {
    match f {
        SglangFlavor::NvidiaGpu => VllmFlavor::NvidiaGpu,
        SglangFlavor::AmdGpu => VllmFlavor::AmdGpu,
        SglangFlavor::Cpu => VllmFlavor::Cpu,
    }
}

/// Translate a [`VllmFlavor`] to its [`SglangFlavor`] counterpart.
/// See [`sglang_to_vllm_flavor`].
pub(crate) fn vllm_to_sglang_flavor(f: VllmFlavor) -> SglangFlavor {
    match f {
        VllmFlavor::NvidiaGpu => SglangFlavor::NvidiaGpu,
        VllmFlavor::AmdGpu => SglangFlavor::AmdGpu,
        VllmFlavor::Cpu => SglangFlavor::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(VllmFlavor::NvidiaGpu, SglangFlavor::NvidiaGpu)]
    #[case(VllmFlavor::AmdGpu, SglangFlavor::AmdGpu)]
    #[case(VllmFlavor::Cpu, SglangFlavor::Cpu)]
    fn flavor_translation_round_trips(#[case] vllm: VllmFlavor, #[case] sglang: SglangFlavor) {
        assert_eq!(vllm_to_sglang_flavor(vllm), sglang);
        assert_eq!(sglang_to_vllm_flavor(sglang), vllm);
    }
}
