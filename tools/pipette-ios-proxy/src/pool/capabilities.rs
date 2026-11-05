//! The capability flags a phone advertises: a deliberate second implementation of
//! `installedRuntimeCapabilities` and `clientVersionCapabilityFlag` in
//! `ios/Pipette/Pipette/Client/Worker/Profile.swift`, since each client owns the flags it
//! reports. The tests pin the two together; a proxied phone advertising a different set
//! would silently stop matching plans that pin it.
//!
//! Every level is advertised (general and versioned): matching is set containment,
//! so an extra level only widens what a client matches.
//!
//! Each iOS app runtime also gets its `runtime:descriptor:<digest>` build flag, which
//! the catalog's build check reads. `pipette_plan_types::runtime_build_flag` computes
//! it. The contract fixture `crates/pipette-contract/fixtures/runtime-build-flags.json`
//! verifies that the Swift port returns the same flag.

use pipette_plan_types::{Runtime, RuntimeBuildFlagError, runtime_build_flag};

/// What `ios/build-llama.sh` stamps when there is no git metadata.
const UNKNOWN_BUILD: &str = "unknown";

/// The flags one runtime contributes, as the app would report them, and its build flag.
pub(crate) fn flags_for(runtime: &Runtime) -> Result<Vec<String>, RuntimeBuildFlagError> {
    let mut flags: Vec<String> = runtime_build_flag(runtime)?
        .map(|flag| flag.into_inner())
        .into_iter()
        .collect();
    match runtime {
        // The descriptor carries one value (tag when pinned, else commit).
        Runtime::LlamacppIosPipette(rt) => {
            flags.push("runtime:llama_cpp".to_string());
            push_level(
                &mut flags,
                "runtime:llama_cpp",
                rt.source.repository_version.as_ref(),
            );
        }
        // One flag per package (canonical form strips whitespace, so a composite
        // would run together). `mlx-swift` is reported both bare, the form a plan
        // generated from a runtime spec carries, and named, the hand-written form.
        Runtime::MlxIosPipette(rt) => {
            flags.push("runtime:mlx".to_string());
            push_level(
                &mut flags,
                "runtime:mlx",
                rt.packages.mlx_swift.repository_version.as_ref(),
            );
            flags.extend(
                [
                    ("mlx-swift", &rt.packages.mlx_swift),
                    ("mlx-swift-lm", &rt.packages.mlx_swift_lm),
                    ("swift-transformers", &rt.packages.swift_transformers),
                ]
                .into_iter()
                .filter_map(|(package, source)| {
                    canonical_build(source.repository_version.as_ref())
                        .map(|build| format!("runtime:mlx:{package}={build}"))
                }),
            );
        }
        // Two packages, as the app reports them (`Client/Worker/Profile.swift`): the bare
        // `runtime:onnx_genai:<build>` names the genai ref, which is what
        // `runtime_capability_flags` derives for an `onnx_genai_ios_pipette` cell, and the
        // named forms pin each half of the stack.
        Runtime::OnnxGenaiIosPipette(rt) => {
            flags.push("runtime:onnx_genai".to_string());
            push_level(
                &mut flags,
                "runtime:onnx_genai",
                rt.packages.onnxruntime_genai.repository_version.as_ref(),
            );
            flags.extend(
                [
                    ("onnxruntime", &rt.packages.onnxruntime),
                    ("onnxruntime-genai", &rt.packages.onnxruntime_genai),
                ]
                .into_iter()
                .filter_map(|(package, source)| {
                    canonical_build(source.repository_version.as_ref())
                        .map(|build| format!("runtime:onnx_genai:{package}={build}"))
                }),
            );
        }
        // Ships with the OS; no build id to pin.
        Runtime::AppleFoundation(_) => flags.push("runtime:apple_foundation".to_string()),
        // Runtimes no iOS build can be. Listed rather than `_` so a new variant in
        // `pipette-plan-types` forces a decision here.
        Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_) => {}
    }
    Ok(flags)
}

fn push_level(flags: &mut Vec<String>, prefix: &str, build: &str) {
    if let Some(build) = canonical_build(build) {
        flags.push(format!("{prefix}:{build}"));
    }
}

/// `build` canonicalized for the wire (one non-canonical flag fails the whole
/// profile update with a 400), or `None` for blank and [`UNKNOWN_BUILD`], which
/// the Swift `canonicalBuild` drops too.
fn canonical_build(build: &str) -> Option<String> {
    let canonical: String = build
        .to_ascii_lowercase()
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    (!canonical.is_empty() && canonical != UNKNOWN_BUILD).then_some(canonical)
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{
        AppleCoreAiMacosPipette, AppleFoundation, LlamacppApkPipette, LlamacppApkPipetteFlavor,
        LlamacppIosPipette, LlamacppIosPipetteFlavor, MlxIosPipette, MlxIosPipetteFlavor,
        MlxSwiftStack, NonEmptyString, OnnxGenaiIosPipette, OnnxGenaiIosPipetteFlavor,
        OnnxGenaiStack, PrivateApis, RepositoryUrl, SourceRepository,
    };

    use super::*;
    use crate::releases::catalog::RUNTIME_BUILD_FLAG_PREFIX;

    fn source(url: &str, version: &str) -> anyhow::Result<SourceRepository> {
        Ok(SourceRepository {
            repository_url: RepositoryUrl::new(url),
            repository_version: NonEmptyString::try_new(version.to_owned())?,
        })
    }

    fn mlx(swift: &str, lm: &str, transformers: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::MlxIosPipette(MlxIosPipette {
            packages: MlxSwiftStack {
                mlx_swift: source("github.com/ml-explore/mlx-swift", swift)?,
                mlx_swift_lm: source("github.com/ml-explore/mlx-swift-lm", lm)?,
                swift_transformers: source(
                    "github.com/huggingface/swift-transformers",
                    transformers,
                )?,
            },
            flavor: MlxIosPipetteFlavor::IosArm64,
            private_apis: PrivateApis::default(),
        }))
    }

    fn llamacpp(version: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppIosPipette(LlamacppIosPipette {
            source: source("github.com/ggml-org/llama.cpp", version)?,
            flavor: LlamacppIosPipetteFlavor::IosArm64,
            private_apis: PrivateApis::default(),
        }))
    }

    fn onnx_genai(onnxruntime: &str, genai: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::OnnxGenaiIosPipette(OnnxGenaiIosPipette {
            packages: OnnxGenaiStack {
                onnxruntime: source("github.com/microsoft/onnxruntime", onnxruntime)?,
                onnxruntime_genai: source("github.com/microsoft/onnxruntime-genai", genai)?,
            },
            flavor: OnnxGenaiIosPipetteFlavor::IosArm64,
            private_apis: PrivateApis::default(),
        }))
    }

    fn apple_foundation() -> Runtime {
        Runtime::AppleFoundation(AppleFoundation {
            private_apis: PrivateApis::default(),
        })
    }

    /// The flags the app's `installedRuntimeCapabilities` mirror pins, sorted: every
    /// flag but the build flag.
    fn sorted(runtime: &Runtime) -> anyhow::Result<Vec<String>> {
        let mut flags: Vec<String> = flags_for(runtime)?
            .into_iter()
            .filter(|flag| !flag.starts_with(RUNTIME_BUILD_FLAG_PREFIX))
            .collect();
        flags.sort();
        Ok(flags)
    }

    fn build_flags(runtime: &Runtime) -> anyhow::Result<Vec<String>> {
        Ok(flags_for(runtime)?
            .into_iter()
            .filter(|flag| flag.starts_with(RUNTIME_BUILD_FLAG_PREFIX))
            .collect())
    }

    /// Each iOS app runtime reports one build flag, the shared function's.
    #[rstest]
    #[case::llamacpp(llamacpp("b10216"))]
    #[case::mlx(mlx("0.31.6", "f5f18ed9d", "1.3.3"))]
    #[case::onnx_genai(onnx_genai("v1.26.0", "v0.15.2"))]
    fn an_ios_app_runtime_reports_its_build_flag(
        #[case] runtime: anyhow::Result<Runtime>,
    ) -> anyhow::Result<()> {
        let runtime = runtime?;
        let expected = runtime_build_flag(&runtime)?
            .context("an iOS app runtime has a build flag")?
            .into_inner();
        assert_eq!(build_flags(&runtime)?, vec![expected]);
        Ok(())
    }

    /// The private APIs are part of the build's identity, so the stock and the
    /// private-API builds of one llama.cpp ref report different build flags.
    #[test]
    fn the_private_apis_change_the_build_flag() -> anyhow::Result<()> {
        let stock = llamacpp("b10216")?;
        let Runtime::LlamacppIosPipette(stock_rt) = &stock else {
            anyhow::bail!("llamacpp() builds an llamacpp_ios_pipette runtime");
        };
        let private = Runtime::LlamacppIosPipette(LlamacppIosPipette {
            private_apis: PrivateApis {
                thermal: true,
                memory: true,
            },
            ..stock_rt.clone()
        });
        assert_ne!(build_flags(&stock)?, build_flags(&private)?);
        Ok(())
    }

    /// Mirrors the Swift `advertisesEachMLXPackageSeparately` and
    /// `alsoAdvertisesTheBareMLXSwiftRefForRustDerivedPlans` for the same versions.
    #[test]
    fn ios_mlx_reports_one_flag_per_package_as_the_app_does() -> anyhow::Result<()> {
        assert_eq!(
            sorted(&mlx("0.31.6", "f5f18ed9d", "1.3.3")?)?,
            vec![
                "runtime:mlx".to_string(),
                "runtime:mlx:0.31.6".to_string(),
                "runtime:mlx:mlx-swift-lm=f5f18ed9d".to_string(),
                "runtime:mlx:mlx-swift=0.31.6".to_string(),
                "runtime:mlx:swift-transformers=1.3.3".to_string(),
            ]
        );
        Ok(())
    }

    /// Mirrors the Swift `neverConcatenatesTwoPackagePinsIntoOneFlag`.
    #[test]
    fn no_flag_packs_two_pins_together() -> anyhow::Result<()> {
        let flags = flags_for(&mlx("0.31.6", "f5f18ed9d", "1.3.3")?)?;
        assert!(
            flags.iter().all(|flag| flag.matches('=').count() <= 1),
            "a flag packs more than one pin: {flags:?}"
        );
        Ok(())
    }

    #[test]
    fn llamacpp_reports_the_general_and_versioned_level() -> anyhow::Result<()> {
        assert_eq!(
            sorted(&llamacpp("b10216")?)?,
            vec![
                "runtime:llama_cpp".to_string(),
                "runtime:llama_cpp:b10216".to_string(),
            ]
        );
        Ok(())
    }

    /// Mirrors the Swift `omitsTheVersionedFlagWhenTheBuildIdIsUnknownOrBlank`.
    #[test]
    fn the_unknown_placeholder_earns_no_versioned_flag() -> anyhow::Result<()> {
        assert_eq!(
            sorted(&llamacpp(UNKNOWN_BUILD)?)?,
            vec!["runtime:llama_cpp".to_string()]
        );
        Ok(())
    }

    /// It ships with the OS, so it has no build flag either.
    #[test]
    fn apple_foundation_reports_the_general_level_alone() -> anyhow::Result<()> {
        assert_eq!(
            flags_for(&apple_foundation())?,
            vec!["runtime:apple_foundation".to_string()]
        );
        Ok(())
    }

    #[test]
    fn onnx_genai_reports_each_package_as_the_app_does() -> anyhow::Result<()> {
        assert_eq!(
            sorted(&onnx_genai("v1.26.0", "v0.15.2")?)?,
            vec![
                "runtime:onnx_genai".to_string(),
                "runtime:onnx_genai:onnxruntime-genai=v0.15.2".to_string(),
                "runtime:onnx_genai:onnxruntime=v1.26.0".to_string(),
                "runtime:onnx_genai:v0.15.2".to_string(),
            ]
        );
        Ok(())
    }

    /// One non-canonical flag fails the whole profile update with a 400.
    #[test]
    fn every_derived_flag_is_canonical() -> anyhow::Result<()> {
        [
            mlx("0.31.6", "F5F18ED9D", "1.3.3")?,
            llamacpp("B10216")?,
            onnx_genai("V1.26.0", "v0.15.2")?,
            apple_foundation(),
        ]
        .iter()
        .map(flags_for)
        .collect::<Result<Vec<_>, _>>()?
        .concat()
        .into_iter()
        .try_for_each(|flag| {
            pipette_plan_types::CapabilityFlag::try_new(flag.clone())
                .map(|_| ())
                .with_context(|| format!("{flag} is not canonical"))
        })
    }

    /// The Android engine and the macOS Core AI engine are the nearest
    /// neighbours a careless arm would swallow.
    #[test]
    fn a_runtime_no_ios_build_can_be_contributes_nothing() -> anyhow::Result<()> {
        let android = Runtime::LlamacppApkPipette(LlamacppApkPipette {
            source: source("github.com/ggml-org/llama.cpp", "b10216")?,
            flavor: LlamacppApkPipetteFlavor::AndroidArm64V8,
        });
        let core_ai = Runtime::AppleCoreAiMacosPipette(AppleCoreAiMacosPipette::bundled());
        assert!(flags_for(&android)?.is_empty());
        assert!(flags_for(&core_ai)?.is_empty());
        Ok(())
    }
}
