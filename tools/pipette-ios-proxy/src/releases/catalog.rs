//! The versions a phone advertises in version mode, and the one a claim gets
//! (DESIGN section 9): the bucket's builds, newest upload first, cut short.

use std::collections::{BTreeSet, HashSet};
use std::time::Instant;

use anyhow::Context;

use pipette_http::HttpClient;
use pipette_plan_types::{Runtime, RuntimeBuildFlagError, runtime_build_flag};

use super::store::{Held, Records};
use super::{Build, BuildVersion, Listing, Reach, list_newest};
use crate::RUNTIME_BUILD_FLAG_PREFIX;
use crate::config::ReleasesBucket;
use crate::state::ReleasePaths;
use crate::timing;

/// How many of the newest builds a phone advertises, the default aside:
/// `PATCH /clients/me` takes 64 KB, and a `413` would stop every phone.
pub(crate) const ADVERTISED_BUILDS: usize = 20;

const FLAG_PREFIX: &str = "client_version:";
// Leave room in the 64 KiB PATCH body for device fields and client details.
const ADVERTISED_FLAGS_MAX_BYTES: usize = 48 * 1024;
const RUNTIME_DIGEST_HEX_BYTES: usize = 64;

#[derive(Debug, thiserror::Error)]
pub(crate) enum CatalogError {
    #[error("{bucket} publishes no app build this phone can install")]
    Empty { bucket: String },
    #[error(
        "this phone withdrew every build {bucket} publishes ({withdrawn}): each failed in a way \
         that would repeat, so a claim could only fail its job"
    )]
    Exhausted { bucket: String, withdrawn: String },
    #[error("{bucket} has no app build whose runtime flags fit the client profile")]
    TooLarge { bucket: String },
}

/// What a phone can install: the listing, minus what it has withdrawn.
#[derive(Debug)]
pub(crate) struct Catalog {
    default: Option<BuildVersion>,
    builds: Vec<Build>,
    /// Listed builds published under another digest than this Mac recorded:
    /// never offered, but their runtimes may come back under a new version.
    changed: Vec<Build>,
    /// Builds past the advertised ones, listed only by [`Catalog::relist_whole`]
    /// and never offered: another proxy may offer them.
    older: Vec<Build>,
    /// Whether the last listing read every build in the bucket.
    whole: bool,
    /// Whether a listing with an unreadable build was warned about.
    warned_partial: bool,
    /// Versions that failed in a way that would repeat on this phone, kept
    /// for the life of the process.
    withdrawn: BTreeSet<BuildVersion>,
    listed_at: Option<Instant>,
}

/// Why a claim got the build it did, as the log names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum ChosenBy {
    Default,
    Newest,
}

#[derive(Debug)]
pub(crate) struct Choice<'a> {
    pub(crate) build: &'a Build,
    pub(crate) by: ChosenBy,
}

#[derive(Debug, PartialEq)]
pub(crate) enum NoChoice {
    NoneAdmitted,
    RuntimeNotCarried { admitted: Vec<BuildVersion> },
}

impl Catalog {
    pub(crate) fn new(default: Option<BuildVersion>) -> Self {
        Self {
            default,
            builds: Vec::new(),
            changed: Vec::new(),
            older: Vec::new(),
            whole: false,
            warned_partial: false,
            withdrawn: BTreeSet::new(),
            listed_at: None,
        }
    }

    pub(crate) fn listed(
        http: &HttpClient,
        bucket: &ReleasesBucket,
        default: Option<&BuildVersion>,
        paths: &ReleasePaths,
    ) -> anyhow::Result<Self> {
        let mut catalog = Self::new(default.cloned());
        catalog
            .relist(http, bucket, paths)
            .context("listing the releases bucket")?;
        Ok(catalog)
    }

    /// List the bucket again. A failure leaves the previous listing in place,
    /// and the next one waits [`timing::CATALOG_REFRESH`] either way.
    pub(crate) fn relist(
        &mut self,
        http: &HttpClient,
        bucket: &ReleasesBucket,
        paths: &ReleasePaths,
    ) -> anyhow::Result<()> {
        self.list(http, bucket, paths, Reach::Newest)
    }

    /// As [`Catalog::relist`], reading every build past the advertised ones
    /// too, so [`Catalog::carried_instead`] can tell that none carries a
    /// runtime. Two more reads per older build, taken only when a claim's
    /// runtime is in no build it admits.
    pub(crate) fn relist_whole(
        &mut self,
        http: &HttpClient,
        bucket: &ReleasesBucket,
        paths: &ReleasePaths,
    ) -> anyhow::Result<()> {
        self.list(http, bucket, paths, Reach::Whole)
    }

    fn list(
        &mut self,
        http: &HttpClient,
        bucket: &ReleasesBucket,
        paths: &ReleasePaths,
        reach: Reach,
    ) -> anyhow::Result<()> {
        self.listed_at = Some(Instant::now());
        let records = Records::load(paths)?;
        let Listing {
            newest,
            older,
            whole,
        } = list_newest(
            http,
            bucket,
            ADVERTISED_BUILDS,
            self.default.as_ref(),
            reach,
        )?;
        let (changed, builds): (Vec<Build>, Vec<Build>) = newest
            .into_iter()
            .partition(|build| matches!(records.held(paths, build), Held::Changed { .. }));
        if builds.is_empty() {
            return Err(CatalogError::Empty {
                bucket: bucket.url.to_string(),
            }
            .into());
        }
        if let Some(default) = &self.default
            && !builds.iter().any(|build| &build.version == default)
        {
            if changed.iter().any(|build| &build.version == default) {
                tracing::warn!(
                    default = %default,
                    "[client_releases] default is published with another digest than recorded; \
                     unpinned jobs take the newest"
                );
            } else {
                tracing::warn!(
                    default = %default,
                    "[client_releases] default is not a build the bucket publishes; unpinned \
                     jobs take the newest"
                );
            }
        }
        if reach == Reach::Whole && !whole && !self.warned_partial {
            self.warned_partial = true;
            tracing::warn!(
                "[client_releases] a build in the bucket could not be read (warned above), so a \
                 job whose runtime no admitted build carries stays retriable: that build may \
                 carry it (logged once)"
            );
        }
        self.builds = builds;
        self.changed = changed;
        self.older = older;
        self.whole = whole;
        let available = self.builds.len();
        let offered = self.offered().count();
        if offered < available {
            tracing::warn!(
                available,
                offered,
                flag_budget_bytes = ADVERTISED_FLAGS_MAX_BYTES,
                "the client profile cannot fit every listed build; offering only builds that fit"
            );
        }
        Ok(())
    }

    pub(crate) fn due(&self) -> bool {
        self.listed_at
            .is_none_or(|at| at.elapsed() >= timing::CATALOG_REFRESH)
    }

    /// Whether a listing ever succeeded.
    pub(crate) fn has_listing(&self) -> bool {
        !self.builds.is_empty()
    }

    /// Every build this phone offers, newest upload first. Reserve space for the
    /// default build, then include newer builds while their flags fit the profile.
    fn offered(&self) -> impl Iterator<Item = &Build> {
        fn include<'a>(
            build: &'a Build,
            selected: &mut Vec<&'a Build>,
            seen: &mut HashSet<Runtime>,
            bytes: &mut usize,
        ) {
            let mut new_runtimes = HashSet::new();
            for runtime in build.readable_runtimes() {
                if !seen.contains(&runtime) {
                    new_runtimes.insert(runtime);
                }
            }
            let cost = FLAG_PREFIX.len()
                + build.version.as_str().len()
                + 3 // quotes and comma
                + new_runtimes.len()
                    * (RUNTIME_BUILD_FLAG_PREFIX.len() + RUNTIME_DIGEST_HEX_BYTES + 3);
            if *bytes + cost <= ADVERTISED_FLAGS_MAX_BYTES {
                selected.push(build);
                seen.extend(new_runtimes);
                *bytes += cost;
            }
        }

        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        let mut bytes = 2; // JSON array brackets.
        if let Some(default) = self.builds.iter().find(|build| {
            self.default.as_ref() == Some(&build.version)
                && !self.withdrawn.contains(&build.version)
        }) {
            include(default, &mut selected, &mut seen, &mut bytes);
        }
        for build in &self.builds {
            if self.withdrawn.contains(&build.version)
                || selected.iter().any(|chosen| std::ptr::eq(*chosen, build))
            {
                continue;
            }
            include(build, &mut selected, &mut seen, &mut bytes);
        }
        // Keep the catalog's newest-first order for choice among admitted builds.
        self.builds
            .iter()
            .filter(move |build| selected.iter().any(|chosen| std::ptr::eq(*chosen, *build)))
    }

    pub(crate) fn offers(&self, version: &str) -> Option<&Build> {
        self.offered()
            .find(|build| build.version.as_str() == version)
    }

    pub(crate) fn versions(&self) -> impl Iterator<Item = &BuildVersion> {
        self.offered().map(|build| &build.version)
    }

    /// `client_version:<V>` per offered build, and the build flag of each runtime that
    /// an offered build carries, sorted and unique: the server reindexes whenever the
    /// set changes. A runtime that this proxy cannot read gets no build flag because no
    /// claim can select it. The manifest reader already warned about the runtime.
    pub(crate) fn advertised(&self) -> Result<Vec<String>, RuntimeBuildFlagError> {
        let build_flags = self
            .offered()
            .flat_map(Build::readable_runtimes)
            .map(|runtime| runtime_build_flag(&runtime))
            .filter_map(Result::transpose)
            .map(|flag| flag.map(|flag| flag.into_inner()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut flags: Vec<String> = self
            .version_flags()
            .into_iter()
            .chain(build_flags)
            .collect();
        flags.sort();
        flags.dedup();
        Ok(flags)
    }

    pub(crate) fn version_flags(&self) -> Vec<String> {
        let mut flags: Vec<String> = self
            .versions()
            .map(|version| format!("{FLAG_PREFIX}{version}"))
            .collect();
        flags.sort();
        flags
    }

    /// The build a claim runs on: among the offered ones it admits (all of
    /// them when the server predates `client_versions`) whose manifest names
    /// `runtime` (any, without one), the default, else the newest.
    pub(crate) fn choose(
        &self,
        admitted: Option<&[String]>,
        runtime: Option<&Runtime>,
    ) -> Result<Choice<'_>, NoChoice> {
        let admits = |build: &&Build| {
            admitted.is_none_or(|admitted| {
                admitted
                    .iter()
                    .any(|version| version == build.version.as_str())
            })
        };
        let candidates: Vec<&Build> = self.offered().filter(admits).collect();
        if candidates.is_empty() {
            return Err(NoChoice::NoneAdmitted);
        }
        let carrying = || {
            candidates
                .iter()
                .copied()
                .filter(|build| runtime.is_none_or(|runtime| build.carries(runtime)))
        };
        if let Some(build) = carrying().find(|build| self.default.as_ref() == Some(&build.version))
        {
            return Ok(Choice {
                build,
                by: ChosenBy::Default,
            });
        }
        carrying()
            .next()
            .map(|build| Choice {
                build,
                by: ChosenBy::Newest,
            })
            .ok_or_else(|| NoChoice::RuntimeNotCarried {
                admitted: candidates
                    .iter()
                    .map(|build| build.version.clone())
                    .collect(),
            })
    }

    /// The distinct runtimes the `admitted` builds carry, newest upload first,
    /// once this listing proves no phone can run `runtime`: it read the whole
    /// bucket, every descriptor is readable, every admitted build is listed,
    /// and no build carries it. Builds outside `admitted` count, since a
    /// claim's `client_versions` holds only what this phone advertised.
    /// `None` otherwise.
    pub(crate) fn carried_instead(
        &self,
        admitted: &[String],
        runtime: &Runtime,
    ) -> Option<Vec<Runtime>> {
        let is_admitted = |build: &&Build| {
            admitted
                .iter()
                .any(|version| version == build.version.as_str())
        };
        let listed = || self.builds.iter().chain(&self.changed).chain(&self.older);
        if !self.whole
            || admitted.is_empty()
            || listed().filter(is_admitted).count() < admitted.len()
        {
            return None;
        }
        let mut carried = Vec::new();
        for build in listed() {
            let named = build
                .runtimes
                .iter()
                .map(|raw| serde_json::from_value::<Runtime>(raw.clone()).ok())
                .collect::<Option<Vec<_>>>()?;
            if named.contains(runtime) {
                return None;
            }
            if is_admitted(&build) {
                for one in named {
                    if !carried.contains(&one) {
                        carried.push(one);
                    }
                }
            }
        }
        Some(carried)
    }

    /// A phone offering nothing would fail every job it claims, and its empty
    /// capability set would leave the server holding the last one it reported.
    pub(crate) fn check_offered(&self, bucket: &ReleasesBucket) -> Result<(), CatalogError> {
        if self.offered().next().is_some() {
            return Ok(());
        }
        if self
            .builds
            .iter()
            .any(|build| !self.withdrawn.contains(&build.version))
        {
            return Err(CatalogError::TooLarge {
                bucket: bucket.url.to_string(),
            });
        }
        Err(CatalogError::Exhausted {
            bucket: bucket.url.to_string(),
            withdrawn: self
                .withdrawn
                .iter()
                .map(BuildVersion::as_str)
                .collect::<Vec<_>>()
                .join(", "),
        })
    }

    pub(crate) fn withdraw(&mut self, version: &BuildVersion) {
        if self.withdrawn.insert(version.clone()) {
            tracing::warn!(
                version = %version,
                "no longer advertising this version: it failed in a way that would repeat \
                 on this phone. A restart offers it again"
            );
        }
    }

    /// A catalog that listed `builds`, the whole bucket.
    #[cfg(test)]
    pub(crate) fn of(default: Option<BuildVersion>, builds: Vec<Build>) -> Self {
        Self {
            builds,
            whole: true,
            listed_at: Some(Instant::now()),
            ..Self::new(default)
        }
    }

    #[cfg(test)]
    pub(crate) fn expire(&mut self) {
        self.listed_at = None;
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::releases::s3;
    use crate::releases::testing::{FakeBucket, b10216, build, sha256_of};
    use crate::state::StatePaths;

    /// The llama.cpp build tagged `version`, so each fake build carries its own.
    fn llamacpp(version: &str) -> serde_json::Value {
        let mut descriptor = b10216();
        descriptor["repository_version"] = version.into();
        descriptor
    }

    fn runtime(version: &str) -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(llamacpp(version))?)
    }

    /// Newest first, as `list_builds` orders them.
    fn builds(versions: &[&str]) -> anyhow::Result<Vec<Build>> {
        versions
            .iter()
            .map(|version| {
                Ok(Build {
                    runtimes: vec![llamacpp(version)],
                    ..build(version, version, version.as_bytes())?
                })
            })
            .collect()
    }

    fn admitted(versions: &[&str]) -> Vec<String> {
        versions
            .iter()
            .map(|version| (*version).to_owned())
            .collect()
    }

    #[test]
    fn large_catalog_offers_only_builds_whose_flags_fit() -> anyhow::Result<()> {
        let mut listed = builds(&["a", "b", "c"])?;
        for build in &mut listed {
            build.runtimes = (0..250)
                .map(|index| llamacpp(&format!("{}-{index}", build.version)))
                .collect();
        }
        let catalog = Catalog::of(Some("c".parse()?), listed);

        assert_eq!(
            catalog
                .versions()
                .map(BuildVersion::as_str)
                .collect::<Vec<_>>(),
            ["a", "c"]
        );
        assert!(catalog.offers("b").is_none());
        assert!(matches!(
            catalog.choose(Some(&admitted(&["b"])), None),
            Err(NoChoice::NoneAdmitted)
        ));
        assert_eq!(
            catalog
                .choose(None, None)
                .ok()
                .map(|choice| choice.build.version.as_str()),
            Some("c")
        );
        assert!(serde_json::to_vec(&catalog.advertised()?)?.len() <= ADVERTISED_FLAGS_MAX_BYTES);
        Ok(())
    }

    #[test]
    fn shared_runtime_flags_leave_room_for_all_build_versions() -> anyhow::Result<()> {
        let mut listed = builds(&["a", "b", "c"])?;
        let runtimes = (0..250)
            .map(|index| llamacpp(&format!("shared-{index}")))
            .collect::<Vec<_>>();
        for build in &mut listed {
            build.runtimes = runtimes.clone();
        }
        let catalog = Catalog::of(None, listed);
        assert_eq!(
            catalog
                .versions()
                .map(BuildVersion::as_str)
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        Ok(())
    }

    #[rstest]
    #[case::older_server_takes_the_default(Some("b"), None, None, Ok(("b", ChosenBy::Default)))]
    #[case::older_server_without_a_default(None, None, None, Ok(("a", ChosenBy::Newest)))]
    #[case::pinned_to_one(Some("b"), Some(&["c"][..]), None, Ok(("c", ChosenBy::Newest)))]
    #[case::pinned_to_several_with_the_default(
        Some("c"),
        Some(&["b", "c"][..]),
        None,
        Ok(("c", ChosenBy::Default))
    )]
    #[case::pinned_to_several_without_the_default(
        Some("a"),
        Some(&["b", "c"][..]),
        None,
        Ok(("b", ChosenBy::Newest))
    )]
    #[case::none_admitted(Some("a"), Some(&[][..]), None, Err(NoChoice::NoneAdmitted))]
    #[case::only_unknown_versions(Some("a"), Some(&["z"][..]), None, Err(NoChoice::NoneAdmitted))]
    #[case::the_default_carries_it(Some("b"), None, Some("b"), Ok(("b", ChosenBy::Default)))]
    #[case::only_an_older_one_does(Some("a"), None, Some("c"), Ok(("c", ChosenBy::Newest)))]
    #[case::none_admitted_carries_it(
        None,
        Some(&["a", "b"][..]),
        Some("c"),
        Err(NoChoice::RuntimeNotCarried { admitted: vec!["a".parse()?, "b".parse()?] })
    )]
    #[case::none_admitted_with_a_runtime(
        None,
        Some(&[][..]),
        Some("a"),
        Err(NoChoice::NoneAdmitted)
    )]
    fn a_claim_gets_the_default_else_the_newest_it_admits_carrying_its_runtime(
        #[case] default: Option<&str>,
        #[case] admits: Option<&[&str]>,
        #[case] wants: Option<&str>,
        #[case] expected: Result<(&str, ChosenBy), NoChoice>,
    ) -> anyhow::Result<()> {
        let catalog = Catalog::of(
            default.map(str::parse).transpose()?,
            builds(&["a", "b", "c"])?,
        );
        let admits = admits.map(admitted);
        let wants = wants.map(runtime).transpose()?;
        let choice = catalog.choose(admits.as_deref(), wants.as_ref());
        assert_eq!(
            choice.map(|choice| (choice.build.version.as_str(), choice.by)),
            expected
        );
        Ok(())
    }

    /// A build the listing holds beside `a` to `d`, which the claim never
    /// admits.
    #[derive(Debug, Clone, Copy)]
    enum Beside {
        Nothing,
        /// `n`, uploaded since this phone advertised, carrying `n`.
        Newer,
        /// `e`, carrying `e`, published under another digest than recorded.
        Changed,
        /// `u`, naming a runtime this proxy cannot read.
        Unreadable,
        /// `o`, carrying `o`, past the advertised builds.
        Older,
        /// A build the listing could not read.
        Skipped,
    }

    /// Only a whole listing in which no build carries the runtime names what
    /// the admitted builds carry; a build the claim did not admit still
    /// counts. `c` carries what `a` does; `b` is withdrawn.
    #[rstest]
    #[case::none_carries_it(&["a", "c"][..], "z", Beside::Nothing, Some(&["a"][..]))]
    #[case::withdrawn_ones_count(&["b", "a"][..], "z", Beside::Nothing, Some(&["a", "b"][..]))]
    #[case::one_carries_it(&["a", "c"][..], "a", Beside::Nothing, None)]
    #[case::a_withdrawn_one_carries_it(&["a"][..], "b", Beside::Nothing, None)]
    #[case::one_not_admitted_carries_it(&["a", "c"][..], "d", Beside::Nothing, None)]
    #[case::a_newer_one_carries_it(&["a", "c"][..], "n", Beside::Newer, None)]
    #[case::a_changed_one_carries_it(&["a", "c"][..], "e", Beside::Changed, None)]
    #[case::a_changed_one_does_not(&["a"][..], "z", Beside::Changed, Some(&["a"][..]))]
    #[case::one_is_not_listed(&["a", "x"][..], "z", Beside::Nothing, None)]
    #[case::one_names_an_unreadable_runtime(&["a"][..], "z", Beside::Unreadable, None)]
    #[case::none_admitted(&[][..], "z", Beside::Nothing, None)]
    #[case::an_older_one_carries_it(&["a"][..], "o", Beside::Older, None)]
    #[case::an_older_one_does_not(&["a"][..], "z", Beside::Older, Some(&["a"][..]))]
    #[case::an_admitted_older_one(&["a", "o"][..], "z", Beside::Older, Some(&["a", "o"][..]))]
    #[case::one_was_skipped(&["a"][..], "z", Beside::Skipped, None)]
    fn what_admitted_builds_carry_instead(
        #[case] admits: &[&str],
        #[case] wants: &str,
        #[case] beside: Beside,
        #[case] expected: Option<&[&str]>,
    ) -> anyhow::Result<()> {
        let mut listed = match beside {
            Beside::Newer => builds(&["n", "a", "b", "c", "d"])?,
            Beside::Unreadable => builds(&["u", "a", "b", "c", "d"])?,
            Beside::Nothing | Beside::Changed | Beside::Older | Beside::Skipped => {
                builds(&["a", "b", "c", "d"])?
            }
        };
        for build in &mut listed {
            match build.version.as_str() {
                "c" => build.runtimes = vec![llamacpp("a")],
                "u" => build.runtimes = vec![serde_json::json!({ "type": "tomorrows_engine_ios" })],
                _ => {}
            }
        }
        let mut catalog = Catalog::of(None, listed);
        match beside {
            Beside::Changed => catalog.changed = builds(&["e"])?,
            Beside::Older => catalog.older = builds(&["o"])?,
            Beside::Skipped => catalog.whole = false,
            Beside::Nothing | Beside::Newer | Beside::Unreadable => {}
        }
        catalog.withdraw(&"b".parse()?);

        let instead = catalog.carried_instead(&admitted(admits), &runtime(wants)?);

        let expected = expected
            .map(|versions| versions.iter().map(|version| runtime(version)).collect())
            .transpose()?;
        assert_eq!(instead, expected);
        Ok(())
    }

    fn build_flag(version: &str) -> anyhow::Result<String> {
        Ok(runtime_build_flag(&runtime(version)?)?
            .ok_or_else(|| anyhow::anyhow!("an iOS app runtime has a build flag"))?
            .into_inner())
    }

    /// The versions, then one build flag per runtime that an offered build carries:
    /// a runtime two builds share appears once, a withdrawn build's runtime and one this
    /// proxy cannot read appear not at all.
    #[test]
    fn a_phone_advertises_the_build_flags_of_its_offered_builds() -> anyhow::Result<()> {
        let mut listed = builds(&["a", "b", "c"])?;
        for build in &mut listed {
            match build.version.as_str() {
                "a" => build
                    .runtimes
                    .push(serde_json::json!({ "type": "tomorrows_engine_ios" })),
                "b" => build.runtimes.push(llamacpp("a")),
                _ => {}
            }
        }
        let mut catalog = Catalog::of(None, listed);
        catalog.withdraw(&"c".parse()?);

        let mut expected = vec![
            "client_version:a".to_owned(),
            "client_version:b".to_owned(),
            build_flag("a")?,
            build_flag("b")?,
        ];
        expected.sort();
        assert_eq!(catalog.advertised()?, expected);
        assert!(!catalog.advertised()?.contains(&build_flag("c")?));
        Ok(())
    }

    #[test]
    fn a_withdrawn_version_is_offered_no_more() -> anyhow::Result<()> {
        let bucket = FakeBucket::config_at("https://releases.example")?;
        let mut catalog = Catalog::of(Some("a".parse()?), builds(&["a", "b"])?);
        assert_eq!(
            catalog.version_flags(),
            ["client_version:a", "client_version:b"]
        );

        catalog.withdraw(&"a".parse()?);

        assert_eq!(catalog.version_flags(), ["client_version:b"]);
        assert_eq!(
            catalog
                .choose(Some(&admitted(&["a", "b"])), None)
                .map(|choice| choice.build.version.as_str()),
            Ok("b")
        );
        assert_eq!(
            catalog.choose(Some(&admitted(&["a"])), None).map(|c| c.by),
            Err(NoChoice::NoneAdmitted)
        );
        assert_eq!(
            catalog
                .choose(None, None)
                .map(|choice| choice.build.version.as_str()),
            Ok("b")
        );
        assert!(catalog.offers("a").is_none());
        catalog.check_offered(&bucket)?;

        catalog.withdraw(&"b".parse()?);

        assert_eq!(catalog.version_flags(), [] as [&str; 0]);
        let error = catalog
            .check_offered(&bucket)
            .err()
            .context("nothing is left to offer")?;
        assert!(
            error.to_string().contains("withdrew every build"),
            "{error}"
        );
        assert!(error.to_string().contains("(a, b)"), "{error}");
        Ok(())
    }

    /// `pulled` is recorded with other bytes, so the store refuses it as changed.
    #[test]
    fn a_listing_leaves_out_changed_builds_and_survives_a_failure() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let paths = StatePaths::resolve(Some(state.path()))?.releases();
        let bucket = FakeBucket::start();
        let pulled = "2026.10.01-0000000-internal";
        let listing = bucket.publish_builds(
            &[("a", pulled, 1), ("b", "2026.10.02-e6a9434-internal", 2)],
            1000,
        );
        std::fs::write(
            &paths.records,
            serde_json::json!({
                "schema": 1,
                "versions": {
                    pulled: {
                        "sha256": sha256_of(b"other bytes"),
                        "key": "k",
                        "pulled_at": "2026-10-01T04:00:00Z",
                    },
                },
            })
            .to_string(),
        )?;
        let http = s3::http_client()?;
        let mut catalog = Catalog::new(None);

        catalog.relist(&http, &bucket.config()?, &paths)?;
        assert_eq!(
            catalog.version_flags(),
            ["client_version:2026.10.02-e6a9434-internal"]
        );
        assert!(!catalog.due());

        listing.into_iter().for_each(|mut page| page.delete());
        bucket.server.mock(|when, then| {
            when.path("/builds/");
            then.status(503);
        });
        let error = catalog
            .relist(&http, &bucket.config()?, &paths)
            .err()
            .context("a 503 listing must fail")?;
        assert!(
            format!("{error:#}").contains("listing the bucket"),
            "{error:#}"
        );
        assert_eq!(
            catalog.version_flags(),
            ["client_version:2026.10.02-e6a9434-internal"]
        );
        Ok(())
    }

    #[test]
    fn an_empty_bucket_is_an_error() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let paths = StatePaths::resolve(Some(state.path()))?.releases();
        let bucket = FakeBucket::start();
        bucket.list(&[], 1000);

        let error = Catalog::new(None)
            .relist(&s3::http_client()?, &bucket.config()?, &paths)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an empty bucket offers nothing"))?;
        assert!(
            matches!(
                error.downcast_ref::<CatalogError>(),
                Some(CatalogError::Empty { .. })
            ),
            "{error:#}"
        );
        Ok(())
    }
}
