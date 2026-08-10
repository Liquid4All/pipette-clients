//! Hardcoded capability-requirement rules (PIP-413).
//!
//! The hardware-policy layer that injects the capability requirements a plan
//! author shouldn't have to know, applied per cell at expansion time. A
//! scheduler-mode variant (`pipette_plan_types::SchedulerVariant`) declares the
//! eligibility it *cares* about (`requires`/`clients`); these rules add what the
//! hardware *demands* — e.g. an iOS Apple-Foundation run must land on one of a
//! curated list of supported iPhones — and reject contradictions before any job
//! body is generated.
//!
//! Resolution per (variant, runtime) starts from the variant's committed
//! `requires`, injects the runtime's policy, applies its `when` blocks, checks
//! the guardrails, and yields the cell's [`EffectiveRequirement`] — a flat
//! `requires` set plus zero or more `any_of` groups (the conjunctive-normal-form
//! the server matches, see `pipette-mgmt` `docs/plan-ingestion.md` §5).
//!
//! The rules are **code, not config**, keyed on [`RuntimeType`] via an
//! exhaustive `match` with no catch-all: a newly added runtime kind fails to
//! compile until it states a policy (an explicit empty one counts), so hardware
//! policy can never silently ship policy-free. A policy change (a new supported
//! device, a raised minimum) is then an ordinary reviewed edit to this file
//! rather than a config deploy.
//!
//! Device families are expressed against an **ordered vocabulary** ([`IPHONES`])
//! rather than as hand-copied lists: a rule names its floor and takes the tail
//! from it ([`Group::AtLeast`]), so two rules can never disagree about which
//! devices exist, and a new device is one append that every floor-based rule
//! picks up.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use pipette_plan_types::{
    CapabilityFlag, CapabilityFlagError, Runtime, RuntimeType, SchedulerPlan,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A capability-rule violation. Distinct variants so a caller — the PIP-414
/// `generate` command, or a test — can match on the rejection kind instead of
/// parsing an error string.
///
/// The `RuleTable*` variants are internal-consistency failures (a typo in this
/// file, not a bad plan); they are returned rather than panicking because the
/// workspace bans panicking constructs, and the `rules_table_is_well_formed`
/// test proves they cannot fire in practice.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityRuleError {
    /// The variant committed none of a `one_of` guardrail's members.
    #[error(
        "requires must commit to exactly one of {one_of:?} (e.g. one `os:`); found none. \
         Add the flag naming the platform this variant targets"
    )]
    OneOfNoCommit { one_of: Vec<String> },

    /// The variant committed more than one member of a `one_of` guardrail.
    #[error(
        "requires commits to {committed:?}, but {one_of:?} are mutually exclusive; \
         exactly one is allowed"
    )]
    OneOfMultipleCommit {
        committed: Vec<String>,
        one_of: Vec<String>,
    },

    /// Two different flags from one single-valued namespace in the flat set.
    #[error(
        "requires contains {first:?} and {second:?}, both in the mutually exclusive \
         `{namespace}:` namespace; a device has exactly one {namespace}"
    )]
    ExclusiveNamespaceConflict {
        namespace: String,
        first: String,
        second: String,
    },

    /// The flat set pins a single-valued flag that the injected family for that
    /// same namespace excludes — satisfiable by no client at all.
    #[error(
        "requires pins {pinned:?}, but this runtime's supported `{namespace}:` family is \
         {family:?}, which excludes it; no client could satisfy both"
    )]
    PinExcludedFromFamily {
        namespace: String,
        pinned: String,
        family: Vec<String>,
    },

    /// A rules-table literal is not a canonical capability flag.
    #[error("rules table: flag {raw:?} is not in canonical form")]
    RuleTableFlagNotCanonical {
        raw: String,
        #[source]
        source: CapabilityFlagError,
    },

    /// A rules-table `AtLeast` floor names a member its vocabulary lacks.
    #[error("rules table: floor {floor:?} is not a member of its ordered vocabulary")]
    RuleTableUnknownFloor { floor: String },

    /// A rules-table group resolved to nothing. An empty disjunction is
    /// satisfiable by no client, so it must be a typo rather than a policy.
    #[error("rules table: an any_of group is empty, which no client can satisfy")]
    RuleTableEmptyGroup,
}

// ---------------------------------------------------------------------------
// EffectiveRequirement
// ---------------------------------------------------------------------------

/// The resolved capability requirement of one cell: a conjunction of a flat
/// `requires` set (all must be present on an eligible client) and zero or more
/// `any_of` groups (each satisfied by at least one member). This is the form
/// `pipette-plan`'s forthcoming job-body generation (PIP-414, not yet
/// implemented) will write, and that the server matches against a client's
/// effective capabilities.
///
/// Both the flat set and each group are stored **sorted and deduplicated**, so
/// two cells with set-equal requirements compare equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveRequirement {
    /// Flat all-of requirements (sorted, deduped).
    pub requires: Vec<CapabilityFlag>,
    /// Disjunction groups (each sorted, deduped); a client must share at least
    /// one flag with every group. Injected by the rules — a plan author never
    /// writes `any_of` directly.
    pub any_of: Vec<Vec<CapabilityFlag>>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Resolve the [`EffectiveRequirement`] for a cell running on `runtime`, given
/// the variant's author-committed `requires`.
///
/// Depends only on the runtime kind, not the model: every runtime pairs with a
/// single model format (see `pipette_plan_types::is_compatible`), so hardware
/// policy is a function of the runtime alone.
pub fn resolve_effective_requirement(
    committed: &[CapabilityFlag],
    runtime: &Runtime,
) -> Result<EffectiveRequirement, CapabilityRuleError> {
    resolve_from_policy(committed, &policy_for(RuntimeType::of(runtime)))
}

/// Apply the capability-requirement rules to every cell of `plan`, rejecting
/// the whole plan if any (variant, runtime) violates a guardrail. Run at
/// generation, before any job body is written. Resolution depends only on the
/// runtime, so each variant is checked once per distinct runtime it declares.
///
/// Returns `anyhow` rather than [`CapabilityRuleError`]: this is the reporting
/// boundary, where the typed cause is wrapped with the variant/runtime context
/// an operator needs to find the offending block.
pub fn validate_capability_rules(plan: &SchedulerPlan) -> anyhow::Result<()> {
    use anyhow::Context;

    plan.variants
        .iter()
        .enumerate()
        .try_for_each(|(idx, variant)| {
            variant.runtimes.iter().try_for_each(|runtime| {
                resolve_effective_requirement(&variant.requires, runtime)
                    .map(|_| ())
                    .with_context(|| {
                        format!("variant {idx} ({} runtime)", runtime.headless_token())
                    })
            })
        })
}

// ---------------------------------------------------------------------------
// The rules table
// ---------------------------------------------------------------------------

/// A disjunction the rules inject: satisfied by at least one member.
enum Group {
    /// Every member of an ordered vocabulary from `floor` onward — the "X or
    /// newer" family. Because the vocabulary is shared, two rules with
    /// different floors cannot disagree about which members exist, and adding
    /// a new one extends every floor-based rule at once.
    AtLeast {
        list: &'static [&'static str],
        floor: &'static str,
    },
    /// An explicit, unordered set — for an attribute with no useful ordering
    /// (e.g. which SoCs count as Apple silicon).
    Exactly(&'static [&'static str]),
}

/// One runtime kind's hardware policy.
struct Policy {
    /// Flags injected into the flat `requires` unconditionally (all-of).
    requires: &'static [&'static str],
    /// Authoring guardrail: the **author-committed** flat `requires` must
    /// contain exactly one of these members — zero is a rejection, two a
    /// contradiction. Empty disables the guardrail. Used where the runtime
    /// spans more than one platform and the author must pick one (e.g. Apple
    /// Foundation on iOS *or* macOS). Deliberately counted against what the
    /// author wrote, not against injected flags, so a policy can never
    /// self-satisfy its own "you must choose" requirement.
    one_of: &'static [&'static str],
    /// Disjunction groups injected unconditionally.
    any_of: &'static [Group],
    /// Conditional injections applied when `if_present` is among the committed
    /// requirements (author-committed ∪ unconditionally-injected `requires`).
    /// These do not chain: a `when`-injected flag never triggers another `when`.
    when: &'static [When],
}

/// A conditional injection: when `if_present` is committed, add `require`
/// (all-of) and `any_of` (disjunction groups) to the effective requirement.
struct When {
    if_present: &'static str,
    require: &'static [&'static str],
    any_of: &'static [Group],
}

impl Policy {
    /// A runtime that demands nothing beyond what the author writes.
    const EMPTY: Policy = Policy {
        requires: &[],
        one_of: &[],
        any_of: &[],
        when: &[],
    };
}

// The rule for each runtime kind. The `match` has no `_` arm, so adding a
// `RuntimeType` variant will not compile until it declares a policy here.
fn policy_for(runtime: RuntimeType) -> Policy {
    use RuntimeType::*;
    match runtime {
        // Apple Foundation Models run on iOS or macOS, on Apple silicon; the
        // author commits to one platform and the device policy follows from it.
        AppleFoundation => Policy {
            requires: &[],
            one_of: &["os:ios", "os:macos"],
            any_of: &[],
            when: &[
                // iOS: iPhone 17 or newer.
                When {
                    if_present: "os:ios",
                    require: &[],
                    any_of: &[Group::AtLeast {
                        list: IPHONES,
                        floor: "device:iphone17",
                    }],
                },
                // macOS: Apple silicon. The macOS 26.0 floor is deferred — see
                // the OS-VERSION FLOORS note below.
                When {
                    if_present: "os:macos",
                    require: &[],
                    any_of: &[Group::Exactly(APPLE_SILICON)],
                },
            ],
        },
        // Desktop MLX (Python/uv): Apple silicon. macOS 14 floor deferred.
        MlxMacosPipette => Policy {
            requires: &["os:macos"],
            one_of: &[],
            any_of: &[Group::Exactly(APPLE_SILICON)],
            when: &[],
        },
        // In-process iOS MLX (mlx-swift): A12 Bionic or newer — the iPhone XS
        // is the oldest qualifying device. iOS 17.2 floor deferred.
        MlxIosPipette => Policy {
            requires: &["os:ios"],
            one_of: &[],
            any_of: &[Group::AtLeast {
                list: IPHONES,
                floor: "device:iphonexs",
            }],
            when: &[],
        },
        // Server-hosted runtimes: Linux only.
        //
        // TODO(PIP-444): vLLM/sglang also require an AVX-512 CPU *or* a GPU,
        // which is not expressible today: neither is a reserved device
        // attribute, and enumerating every capable chip is impractical. Add the
        // disjunction once clients publish coarse CPU-feature / accelerator
        // capability flags (a separate convention, out of scope here).
        DockerVllm | DockerSglang | UvVllm | UvSglang => Policy {
            requires: &["os:linux"],
            one_of: &[],
            any_of: &[],
            when: &[],
        },
        // OpenVINO: Linux or Windows. Windows is not incidental — Intel NPU
        // hardware is overwhelmingly Windows-hosted, which is why this is the
        // one venv-backed runtime that is not Linux-pinned.
        //
        // TODO(PIP-444): the real floor is an Intel client, and for
        // `device = "npu"` an Intel Core Ultra with an AI Boost NPU. Neither is
        // expressible — there is no CPU-vendor or accelerator capability
        // attribute — so an NPU cell scheduled onto an AMD box strands at
        // dispatch rather than at claim time. Same blocker as the vLLM/sglang
        // AVX-512 note above.
        UvOpenvino => Policy {
            requires: &[],
            one_of: &["os:linux", "os:windows"],
            any_of: &[],
            when: &[],
        },
        // Desktop llama.cpp fans out across macOS/Linux/Windows and every
        // flavor; there is no single hardware floor to inject, so the author's
        // own `requires` stand alone.
        LlamacppCliStockTools => Policy::EMPTY,
        // In-process mobile llama.cpp: pin the platform. No device floor —
        // llama.cpp runs on far older hardware than MLX or AFM, so any
        // restriction here would be invented rather than derived.
        LlamacppApkPipette => Policy {
            requires: &["os:android"],
            one_of: &[],
            any_of: &[],
            when: &[],
        },
        LlamacppIosPipette => Policy {
            requires: &["os:ios"],
            one_of: &[],
            any_of: &[],
            when: &[],
        },
    }
}

// ---------------------------------------------------------------------------
// Ordered vocabularies
//
// TODO(PIP-444): the membership below is representative, not authoritative —
// filling in the curated set is a one-line-per-entry reviewed edit, which is
// the point of keeping the rules in code. The *ordering* is the load-bearing
// part: `Group::AtLeast` floors index into these lists, so entries must stay
// oldest-first.
//
// OS-VERSION FLOORS (deferred, deliberately): several real policies have an
// OS-version minimum — AFM needs macOS 26.0+/iOS 26+, desktop MLX needs macOS
// 14+, iOS MLX needs iOS 17.2+. None is expressible today. The server
// normalizes `device_os_version` verbatim, so a device reports its full version
// (`os_version:26.1`); a rule requiring `os_version:26` would therefore match
// nothing, and enumerating every point release is unbounded. Injecting an
// unmatchable group is worse than injecting none — it would strand every job
// for that runtime — so the floors are omitted until clients publish coarse
// major-version flags alongside the full version (e.g. `os_version:26` *and*
// `os_version:26.1`), at which point each becomes a `Group::AtLeast` over an
// ordered version vocabulary added here.
// ---------------------------------------------------------------------------

/// Every iPhone generation pipette tracks, **oldest first**. Shared by every
/// device-family rule: `Group::AtLeast` names its floor and takes the tail, so
/// "iPhone 17 or newer" and "A12 or newer" are two floors over one list rather
/// than two lists that can drift apart.
const IPHONES: &[&str] = &[
    // A12 Bionic (2018) — the in-process MLX floor.
    "device:iphonexs",
    "device:iphonexsmax",
    "device:iphonexr",
    // A13
    "device:iphone11",
    "device:iphone11pro",
    "device:iphone11promax",
    // A14
    "device:iphone12mini",
    "device:iphone12",
    "device:iphone12pro",
    "device:iphone12promax",
    // A15
    "device:iphone13mini",
    "device:iphone13",
    "device:iphone13pro",
    "device:iphone13promax",
    "device:iphone14",
    "device:iphone14plus",
    // A16
    "device:iphone14pro",
    "device:iphone14promax",
    "device:iphone15",
    "device:iphone15plus",
    // A17 Pro
    "device:iphone15pro",
    "device:iphone15promax",
    // A18
    "device:iphone16",
    "device:iphone16plus",
    "device:iphone16pro",
    "device:iphone16promax",
    // A19 — the Apple Foundation Models floor.
    "device:iphone17",
    "device:iphone17pro",
    "device:iphone17promax",
    "device:iphone18",
    "device:iphone18pro",
];

/// Apple silicon Mac SoCs. Unordered on purpose: every M-series chip qualifies,
/// so there is no "or newer" floor to express — a new generation is an append
/// with no rule change.
///
/// Tiers are spelled out because a Mac reports `machdep.cpu.brand_string`
/// ("Apple M3 Max", never a bare "Apple M3"), which the server slugifies whole
/// and matches by exact equality — base names alone match only base-chip Macs.
/// The grid is filled rather than pruned to shipped SKUs: a tier that does not
/// exist matches nobody, while a missing one strands real hardware silently.
const APPLE_SILICON: &[&str] = &[
    "chip:applem1",
    "chip:applem1pro",
    "chip:applem1max",
    "chip:applem1ultra",
    "chip:applem2",
    "chip:applem2pro",
    "chip:applem2max",
    "chip:applem2ultra",
    "chip:applem3",
    "chip:applem3pro",
    "chip:applem3max",
    "chip:applem3ultra",
    "chip:applem4",
    "chip:applem4pro",
    "chip:applem4max",
    "chip:applem4ultra",
    "chip:applem5",
    "chip:applem5pro",
    "chip:applem5max",
    "chip:applem5ultra",
];

// ---------------------------------------------------------------------------
// Exclusive namespaces
// ---------------------------------------------------------------------------

/// Reserved, single-valued capability namespaces — a device has exactly one of
/// each, so two different flags sharing one of these in a variant's flat
/// `requires` is a contradiction. Mirrors `pipette-mgmt`'s
/// `RESERVED_CAPABILITY_NAMESPACES` (`src/client.rs`); `runtime:` and free-form
/// flags are excluded (a device may have many). The server re-checks at
/// ingestion (§6.2), so drift here only changes how early a bad plan is caught,
/// never soundness.
const EXCLUSIVE_NAMESPACES: &[&str] = &[
    "os",
    "os_version",
    "device",
    "chip",
    "form_factor",
    "ram_bytes",
    "gpu",
    "gpu_vram_bytes",
    "npu",
    "npu_vram_bytes",
];

/// The namespace of a flag — the token before its first `:`, or `None` for a
/// free-form flag with no namespace.
fn namespace_of(flag: &str) -> Option<&str> {
    flag.split_once(':').map(|(namespace, _)| namespace)
}

// ---------------------------------------------------------------------------
// Resolution engine
// ---------------------------------------------------------------------------

fn resolve_from_policy(
    committed: &[CapabilityFlag],
    policy: &Policy,
) -> Result<EffectiveRequirement, CapabilityRuleError> {
    // The `one_of` guardrail asks what the *author* chose, so it is counted
    // against the committed set alone — before any injection, so a policy can
    // never satisfy its own "you must choose a platform" requirement.
    check_one_of(committed, policy.one_of)?;

    // The flat requires after unconditional injection. This set drives the
    // `when.if` tests, and is fixed before any conditional injection since
    // `when` blocks do not chain.
    let mut flat = committed.to_vec();
    flat.extend(flags(policy.requires)?);
    let mut groups = resolve_groups(policy.any_of)?;

    // Collect the triggered blocks before folding them in: `triggered` borrows
    // `flat` (via the trigger set), and the fold mutates it.
    let trigger_forms: HashSet<&str> = flat.iter().map(CapabilityFlag::as_ref).collect();
    let triggered: Vec<&When> = policy
        .when
        .iter()
        .filter(|w| trigger_forms.contains(w.if_present))
        .collect();
    triggered
        .iter()
        .try_for_each(|w| -> Result<(), CapabilityRuleError> {
            flat.extend(flags(w.require)?);
            groups.extend(resolve_groups(w.any_of)?);
            Ok(())
        })?;

    check_no_exclusive_conflict(&flat)?;
    check_pins_within_families(&flat, &groups)?;

    Ok(EffectiveRequirement {
        requires: normalized(flat),
        any_of: groups.into_iter().map(normalized).collect(),
    })
}

/// Enforce the "commit to exactly one" guardrail over the author's own flags.
fn check_one_of(committed: &[CapabilityFlag], one_of: &[&str]) -> Result<(), CapabilityRuleError> {
    if one_of.is_empty() {
        return Ok(());
    }
    let chosen: Vec<String> = one_of
        .iter()
        .filter(|&&member| committed.iter().any(|f| f.as_ref() == member))
        .map(|&member| member.to_owned())
        .collect();
    let one_of_owned = || one_of.iter().map(|&m| m.to_owned()).collect::<Vec<_>>();
    match chosen.len() {
        1 => Ok(()),
        0 => Err(CapabilityRuleError::OneOfNoCommit {
            one_of: one_of_owned(),
        }),
        _ => Err(CapabilityRuleError::OneOfMultipleCommit {
            committed: chosen,
            one_of: one_of_owned(),
        }),
    }
}

/// Reject a flat `requires` set containing two *different* flags that share an
/// exclusive namespace. The same flag committed and injected is fine — it
/// dedups. `any_of` groups are exempt by construction (this only sees the flat
/// set), since a group is deliberately many flags from one namespace.
fn check_no_exclusive_conflict(flat: &[CapabilityFlag]) -> Result<(), CapabilityRuleError> {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    flat.iter().try_for_each(|f| {
        let flag = f.as_ref();
        let Some(namespace) = namespace_of(flag).filter(|ns| EXCLUSIVE_NAMESPACES.contains(ns))
        else {
            return Ok(());
        };
        match seen.insert(namespace, flag) {
            Some(previous) if previous != flag => {
                Err(CapabilityRuleError::ExclusiveNamespaceConflict {
                    namespace: namespace.to_owned(),
                    first: previous.to_owned(),
                    second: flag.to_owned(),
                })
            }
            _ => Ok(()),
        }
    })
}

/// Reject a flat pin that the injected family for the same single-valued
/// namespace excludes — e.g. `requires = ["os:ios", "device:iphone12"]` against
/// the injected iPhone-17-or-newer family. Each is individually well formed, but
/// their conjunction is satisfiable by no client (a device has exactly one
/// `device:`), so it would mint jobs nothing can ever lease.
fn check_pins_within_families(
    flat: &[CapabilityFlag],
    groups: &[Vec<CapabilityFlag>],
) -> Result<(), CapabilityRuleError> {
    groups.iter().try_for_each(|group| {
        let Some(namespace) = homogeneous_exclusive_namespace(group) else {
            return Ok(());
        };
        let pinned = flat
            .iter()
            .find(|f| namespace_of(f.as_ref()) == Some(namespace));
        match pinned {
            Some(pin) if !group.contains(pin) => Err(CapabilityRuleError::PinExcludedFromFamily {
                namespace: namespace.to_owned(),
                pinned: pin.as_ref().to_owned(),
                family: group.iter().map(|m| m.as_ref().to_owned()).collect(),
            }),
            _ => Ok(()),
        }
    })
}

/// The single exclusive namespace every member of `group` belongs to, or `None`
/// when the group is empty, spans namespaces, or is not an exclusive one. Only
/// a homogeneous exclusive group constrains a flat pin.
fn homogeneous_exclusive_namespace(group: &[CapabilityFlag]) -> Option<&'static str> {
    let first = namespace_of(group.first()?.as_ref())?;
    let namespace = EXCLUSIVE_NAMESPACES.iter().find(|ns| **ns == first)?;
    group
        .iter()
        .all(|m| namespace_of(m.as_ref()) == Some(*namespace))
        .then_some(*namespace)
}

/// Expand one injected [`Group`] to its member flags.
fn resolve_group(group: &Group) -> Result<Vec<CapabilityFlag>, CapabilityRuleError> {
    let members: &[&str] = match group {
        Group::Exactly(members) => members,
        Group::AtLeast { list, floor } => {
            let from = list.iter().position(|m| m == floor).ok_or_else(|| {
                CapabilityRuleError::RuleTableUnknownFloor {
                    floor: (*floor).to_owned(),
                }
            })?;
            &list[from..]
        }
    };
    if members.is_empty() {
        return Err(CapabilityRuleError::RuleTableEmptyGroup);
    }
    flags(members)
}

fn resolve_groups(groups: &[Group]) -> Result<Vec<Vec<CapabilityFlag>>, CapabilityRuleError> {
    groups.iter().map(resolve_group).collect()
}

/// Construct a canonical [`CapabilityFlag`] from a rules-table literal.
fn flag(raw: &str) -> Result<CapabilityFlag, CapabilityRuleError> {
    CapabilityFlag::try_new(raw.to_owned()).map_err(|source| {
        CapabilityRuleError::RuleTableFlagNotCanonical {
            raw: raw.to_owned(),
            source,
        }
    })
}

fn flags(raw: &[&str]) -> Result<Vec<CapabilityFlag>, CapabilityRuleError> {
    raw.iter().map(|s| flag(s)).collect()
}

/// Sort by canonical string and drop duplicates, so set-equal inputs produce
/// equal `Vec`s (the same normal form `Eligibility` uses).
fn normalized(mut flags: Vec<CapabilityFlag>) -> Vec<CapabilityFlag> {
    flags.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
    flags.dedup();
    flags
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;
    use strum::IntoEnumIterator;

    use super::*;

    /// The canonical string forms of an effective requirement's flat set.
    fn requires_of(er: &EffectiveRequirement) -> Vec<String> {
        er.requires.iter().map(|f| f.as_ref().to_owned()).collect()
    }

    /// Whether some `any_of` group of `er` is exactly `expected`, as a set.
    fn has_group(er: &EffectiveRequirement, expected: &[&str]) -> bool {
        let mut want: Vec<&str> = expected.to_vec();
        want.sort_unstable();
        want.dedup();
        er.any_of
            .iter()
            .any(|g| g.iter().map(CapabilityFlag::as_ref).collect::<Vec<_>>() == want)
    }

    /// The group of `er` drawn from `namespace`, as canonical strings.
    fn group_in(er: &EffectiveRequirement, namespace: &str) -> Vec<String> {
        er.any_of
            .iter()
            .find(|g| homogeneous_exclusive_namespace(g) == Some(namespace))
            .map(|g| g.iter().map(|f| f.as_ref().to_owned()).collect())
            .unwrap_or_default()
    }

    // ---- rule primitives (synthetic policies) ----------------------------

    #[test]
    fn requires_injects_all_of() -> anyhow::Result<()> {
        let policy = Policy {
            requires: &["os:linux"],
            ..Policy::EMPTY
        };
        let er = resolve_from_policy(&flags(&["runtime:vllm"])?, &policy)?;
        assert_eq!(requires_of(&er), vec!["os:linux", "runtime:vllm"]);
        assert!(er.any_of.is_empty());
        Ok(())
    }

    #[test]
    fn any_of_group_injected_unconditionally() -> anyhow::Result<()> {
        let policy = Policy {
            any_of: &[Group::Exactly(&["chip:a", "chip:b"])],
            ..Policy::EMPTY
        };
        let er = resolve_from_policy(&[], &policy)?;
        assert!(has_group(&er, &["chip:a", "chip:b"]));
        Ok(())
    }

    #[test]
    fn one_of_exactly_one_accepted() -> anyhow::Result<()> {
        let policy = Policy {
            one_of: &["os:ios", "os:macos"],
            ..Policy::EMPTY
        };
        let er = resolve_from_policy(&flags(&["os:ios"])?, &policy)?;
        assert_eq!(requires_of(&er), vec!["os:ios"]);
        Ok(())
    }

    #[rstest]
    #[case::zero_commit(&[], |e: &CapabilityRuleError| {
        matches!(e, CapabilityRuleError::OneOfNoCommit { .. })
    })]
    #[case::two_commits(&["os:ios", "os:macos"], |e: &CapabilityRuleError| {
        matches!(e, CapabilityRuleError::OneOfMultipleCommit { .. })
    })]
    fn one_of_rejects_bad_commit_counts(
        #[case] committed: &[&str],
        #[case] expected: fn(&CapabilityRuleError) -> bool,
    ) -> anyhow::Result<()> {
        let policy = Policy {
            one_of: &["os:ios", "os:macos"],
            ..Policy::EMPTY
        };
        let err = resolve_from_policy(&flags(committed)?, &policy)
            .err()
            .context("a bad one_of commit count must be rejected")?;
        assert!(expected(&err), "got {err:?}");
        Ok(())
    }

    #[test]
    fn one_of_counts_only_author_committed_flags() -> anyhow::Result<()> {
        // A policy that injects a `one_of` member must NOT thereby satisfy its
        // own guardrail: the author still has to choose. Guards the regression
        // where the count ran over the post-injection set.
        let policy = Policy {
            requires: &["os:macos"],
            one_of: &["os:ios", "os:macos"],
            ..Policy::EMPTY
        };
        let err = resolve_from_policy(&[], &policy)
            .err()
            .context("an injected one_of member must not satisfy the guardrail")?;
        assert!(
            matches!(err, CapabilityRuleError::OneOfNoCommit { .. }),
            "got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn when_injects_only_on_committed_if() -> anyhow::Result<()> {
        let policy = Policy {
            when: &[When {
                if_present: "os:ios",
                require: &["runtime:llama_cpp"],
                any_of: &[Group::Exactly(&["chip:a", "chip:b"])],
            }],
            ..Policy::EMPTY
        };
        let on = resolve_from_policy(&flags(&["os:ios"])?, &policy)?;
        assert!(requires_of(&on).contains(&"runtime:llama_cpp".to_owned()));
        assert!(has_group(&on, &["chip:a", "chip:b"]));

        let off = resolve_from_policy(&flags(&["os:android"])?, &policy)?;
        assert!(!requires_of(&off).contains(&"runtime:llama_cpp".to_owned()));
        assert!(off.any_of.is_empty());
        Ok(())
    }

    // ---- exclusive-namespace contradictions ------------------------------

    #[rstest]
    #[case::os(&["os:ios", "os:macos"], "os")]
    #[case::device(&["device:iphone17", "device:iphone18"], "device")]
    #[case::chip(&["chip:applem1", "chip:applem2"], "chip")]
    #[case::os_version(&["os_version:14", "os_version:15"], "os_version")]
    #[case::ram(&["ram_bytes:8", "ram_bytes:16"], "ram_bytes")]
    fn exclusive_namespace_contradiction_rejected(
        #[case] committed: &[&str],
        #[case] namespace: &str,
    ) -> anyhow::Result<()> {
        let err = resolve_from_policy(&flags(committed)?, &Policy::EMPTY)
            .err()
            .context("two flags in one exclusive namespace must be rejected")?;
        assert!(
            matches!(&err, CapabilityRuleError::ExclusiveNamespaceConflict { namespace: ns, .. } if ns == namespace),
            "got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn injected_flag_contradicting_committed_is_rejected() -> anyhow::Result<()> {
        // The acceptance criterion is about the *effective* requirements, so the
        // conflict must also be caught when one side is injected by the policy
        // rather than written by the author: MLX-iOS injects `os:ios`.
        let runtime: Runtime = toml::from_str(
            r#"type = "mlx_ios_pipette"
flavor = "ios-arm64"
packages = { mlx_swift = { version = "1" }, mlx_swift_lm = { version = "1" }, swift_transformers = { version = "1" } }"#,
        )?;
        let err = resolve_effective_requirement(&flags(&["os:android"])?, &runtime)
            .err()
            .context("committed os:android must conflict with injected os:ios")?;
        assert!(
            matches!(&err, CapabilityRuleError::ExclusiveNamespaceConflict { namespace, .. } if namespace == "os"),
            "got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn identical_committed_and_injected_flag_is_accepted() -> anyhow::Result<()> {
        // Restating a flag the policy also injects is not a contradiction; it
        // dedups to one.
        let policy = Policy {
            requires: &["os:linux"],
            ..Policy::EMPTY
        };
        let er = resolve_from_policy(&flags(&["os:linux"])?, &policy)?;
        assert_eq!(requires_of(&er), vec!["os:linux"]);
        Ok(())
    }

    #[test]
    fn non_reserved_namespace_allows_multiple() -> anyhow::Result<()> {
        // `runtime:` is not exclusive — a device may advertise several.
        let er = resolve_from_policy(
            &flags(&["runtime:llama_cpp", "runtime:mlx"])?,
            &Policy::EMPTY,
        )?;
        assert_eq!(er.requires.len(), 2);
        Ok(())
    }

    #[test]
    fn any_of_may_repeat_a_namespace() -> anyhow::Result<()> {
        // The exclusive check covers the flat set only; an injected device
        // family is deliberately many `device:` flags.
        let policy = Policy {
            any_of: &[Group::Exactly(&["device:a", "device:b", "device:c"])],
            ..Policy::EMPTY
        };
        let er = resolve_from_policy(&flags(&["os:ios"])?, &policy)?;
        assert!(has_group(&er, &["device:a", "device:b", "device:c"]));
        Ok(())
    }

    #[test]
    fn requirements_are_normalized() -> anyhow::Result<()> {
        let er = resolve_from_policy(
            &flags(&["runtime:b", "runtime:a", "runtime:a"])?,
            &Policy::EMPTY,
        )?;
        assert_eq!(requires_of(&er), vec!["runtime:a", "runtime:b"]);
        Ok(())
    }

    // ---- pins vs. injected families --------------------------------------

    #[test]
    fn pin_outside_injected_family_is_rejected() -> anyhow::Result<()> {
        // `os:ios` + an iPhone 12 pin against AFM's iPhone-17-or-newer family:
        // each is well formed, the conjunction is satisfiable by nobody.
        let err = resolve_effective_requirement(
            &flags(&["os:ios", "device:iphone12"])?,
            &Runtime::AppleFoundation(Default::default()),
        )
        .err()
        .context("a pin the injected family excludes must be rejected")?;
        assert!(
            matches!(
                &err,
                CapabilityRuleError::PinExcludedFromFamily { namespace, pinned, .. }
                    if namespace == "device" && pinned == "device:iphone12"
            ),
            "got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn pin_inside_injected_family_is_accepted() -> anyhow::Result<()> {
        // Narrowing to one supported device is legitimate — a benchmark pinned
        // to specific hardware. It must survive the family check.
        let er = resolve_effective_requirement(
            &flags(&["os:ios", "device:iphone17pro"])?,
            &Runtime::AppleFoundation(Default::default()),
        )?;
        assert!(requires_of(&er).contains(&"device:iphone17pro".to_owned()));
        Ok(())
    }

    // ---- ordered vocabulary ----------------------------------------------

    #[test]
    fn at_least_group_takes_the_suffix_from_its_floor() -> anyhow::Result<()> {
        let group = resolve_group(&Group::AtLeast {
            list: IPHONES,
            floor: "device:iphone17",
        })?;
        let forms: Vec<&str> = group.iter().map(CapabilityFlag::as_ref).collect();
        assert_eq!(
            forms.first(),
            Some(&"device:iphone17"),
            "the floor leads the family"
        );
        assert!(
            !forms.contains(&"device:iphone16"),
            "devices older than the floor are excluded: {forms:?}"
        );
        assert!(
            forms.contains(&"device:iphone18pro"),
            "everything newer is included: {forms:?}"
        );
        Ok(())
    }

    #[test]
    fn unknown_floor_is_a_rules_table_error() -> anyhow::Result<()> {
        let err = resolve_group(&Group::AtLeast {
            list: IPHONES,
            floor: "device:nosuchphone",
        })
        .err()
        .context("a floor outside its vocabulary must be an error")?;
        assert!(
            matches!(err, CapabilityRuleError::RuleTableUnknownFloor { .. }),
            "got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn empty_group_is_a_rules_table_error() -> anyhow::Result<()> {
        let err = resolve_group(&Group::Exactly(&[]))
            .err()
            .context("an empty any_of group must be an error")?;
        assert!(
            matches!(err, CapabilityRuleError::RuleTableEmptyGroup),
            "got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn one_vocabulary_keeps_two_floors_consistent() -> anyhow::Result<()> {
        // The structural payoff of the shared list: the MLX (A12) family is a
        // strict superset of the AFM (iPhone 17) family, and both end at the
        // same newest device. Two hand-copied lists could not guarantee this.
        let afm = resolve_group(&Group::AtLeast {
            list: IPHONES,
            floor: "device:iphone17",
        })?;
        let mlx = resolve_group(&Group::AtLeast {
            list: IPHONES,
            floor: "device:iphonexs",
        })?;
        assert!(
            afm.iter().all(|d| mlx.contains(d)),
            "every AFM-capable device must also be MLX-capable"
        );
        assert_eq!(
            afm.last().map(CapabilityFlag::as_ref),
            mlx.last().map(CapabilityFlag::as_ref),
            "both families must extend to the newest known device"
        );
        Ok(())
    }

    // ---- real Apple-Foundation policy (the acceptance cases) -------------

    #[test]
    fn afm_ios_variant_gains_supported_iphone_family() -> anyhow::Result<()> {
        let er = resolve_effective_requirement(
            &flags(&["os:ios"])?,
            &Runtime::AppleFoundation(Default::default()),
        )?;
        let devices = group_in(&er, "device");
        assert!(
            devices.contains(&"device:iphone17".to_owned()),
            "expected the iPhone-17-or-newer family, got {devices:?}"
        );
        assert!(
            !devices.contains(&"device:iphone12".to_owned()),
            "family must not reach below the floor: {devices:?}"
        );
        Ok(())
    }

    #[test]
    fn afm_macos_variant_has_no_iphone_family() -> anyhow::Result<()> {
        let er = resolve_effective_requirement(
            &flags(&["os:macos"])?,
            &Runtime::AppleFoundation(Default::default()),
        )?;
        assert!(
            group_in(&er, "device").is_empty(),
            "a macOS AFM variant must not gain an iPhone family"
        );
        // It does gain the Apple-silicon group.
        assert!(has_group(&er, APPLE_SILICON));
        Ok(())
    }

    #[rstest]
    #[case::android(&["os:android"])]
    #[case::no_os(&[])]
    #[case::two_os(&["os:ios", "os:macos"])]
    fn afm_bad_os_commitment_rejected(#[case] committed: &[&str]) -> anyhow::Result<()> {
        // `os:android` and the empty set both commit to none of {ios, macos};
        // committing to both is the double-commit case. All three are the
        // `one_of` guardrail rejecting the variant before any output.
        let err = resolve_effective_requirement(
            &flags(committed)?,
            &Runtime::AppleFoundation(Default::default()),
        )
        .err()
        .context("a bad AFM OS commitment must be rejected")?;
        assert!(
            matches!(
                err,
                CapabilityRuleError::OneOfNoCommit { .. }
                    | CapabilityRuleError::OneOfMultipleCommit { .. }
            ),
            "got {err:?}"
        );
        Ok(())
    }

    // ---- other real policies ---------------------------------------------

    #[test]
    fn linux_server_runtime_injects_os_linux() -> anyhow::Result<()> {
        let runtime: Runtime = toml::from_str(
            r#"type = "uv_vllm"
server_version = "0.10.0"
build = "cu121"
python_version = "3.12"
source = { type = "pip_requirements_text", contents = "vllm==0.10.0" }"#,
        )?;
        let er = resolve_effective_requirement(&flags(&["runtime:vllm"])?, &runtime)?;
        assert!(requires_of(&er).contains(&"os:linux".to_owned()));
        Ok(())
    }

    #[test]
    fn mlx_macos_runtime_injects_os_and_apple_silicon() -> anyhow::Result<()> {
        let runtime: Runtime = toml::from_str(
            r#"type = "mlx_macos_pipette"
version = "0.20.0"
flavor = "macos-arm64"
source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" }"#,
        )?;
        let er = resolve_effective_requirement(&[], &runtime)?;
        assert!(requires_of(&er).contains(&"os:macos".to_owned()));
        assert!(has_group(&er, APPLE_SILICON));
        Ok(())
    }

    #[test]
    fn mlx_ios_runtime_family_reaches_back_to_a12() -> anyhow::Result<()> {
        let runtime: Runtime = toml::from_str(
            r#"type = "mlx_ios_pipette"
flavor = "ios-arm64"
packages = { mlx_swift = { version = "1" }, mlx_swift_lm = { version = "1" }, swift_transformers = { version = "1" } }"#,
        )?;
        let er = resolve_effective_requirement(&[], &runtime)?;
        assert!(requires_of(&er).contains(&"os:ios".to_owned()));
        let devices = group_in(&er, "device");
        assert!(
            devices.contains(&"device:iphonexs".to_owned()),
            "the A12 floor must be included: {devices:?}"
        );
        Ok(())
    }

    #[test]
    fn desktop_llamacpp_injects_nothing() -> anyhow::Result<()> {
        let runtime: Runtime = toml::from_str(
            r#"type = "llamacpp_cli_stock_tools"
source = "github_release"
version = "b5000"
flavor = "macos-arm64""#,
        )?;
        let er = resolve_effective_requirement(&flags(&["os:windows"])?, &runtime)?;
        assert_eq!(requires_of(&er), vec!["os:windows"]);
        assert!(er.any_of.is_empty());
        Ok(())
    }

    // ---- plan-level validation over a parsed SchedulerPlan ---------------

    #[test]
    fn validate_accepts_a_well_formed_afm_plan() -> anyhow::Result<()> {
        let plan = SchedulerPlan::parse(
            r#"benchmarks = ["decode_throughput_512_100"]
[[variants]]
requires = ["os:ios"]
models   = [{ type = "apple_foundation_text" }]
runtimes = [{ type = "apple_foundation" }]"#,
        )?;
        validate_capability_rules(&plan)?;
        Ok(())
    }

    #[test]
    fn validate_rejects_afm_variant_with_no_os_and_names_the_variant() -> anyhow::Result<()> {
        // A clients-only AFM variant parses under PIP-399 (eligibility via
        // `clients`), but the rules still demand an OS commitment: a job body
        // that pins a device must also state the hardware it targets.
        let plan = SchedulerPlan::parse(
            r#"benchmarks = ["decode_throughput_512_100"]
[[variants]]
clients  = ["ev1_abc"]
models   = [{ type = "apple_foundation_text" }]
runtimes = [{ type = "apple_foundation" }]"#,
        )?;
        let err = validate_capability_rules(&plan)
            .err()
            .context("an AFM variant with no OS must be rejected at generation")?;
        // The reporting boundary adds operator context...
        assert!(
            err.to_string().contains("variant 0"),
            "missing variant context: {err}"
        );
        // ...over a typed cause the caller can still match on.
        assert!(
            matches!(
                err.downcast_ref::<CapabilityRuleError>(),
                Some(CapabilityRuleError::OneOfNoCommit { .. })
            ),
            "expected a typed OneOfNoCommit cause, got {err:?}"
        );
        Ok(())
    }

    // ---- drift guard: the whole table stays well formed ------------------

    #[test]
    fn rules_table_is_well_formed() -> anyhow::Result<()> {
        // Iterates the enum itself (via strum), so a newly added `RuntimeType`
        // is covered automatically — a hand-written list here could go stale
        // precisely when a new runtime's literals most need checking.
        RuntimeType::iter().try_for_each(|rt| -> anyhow::Result<()> {
            let policy = policy_for(rt);
            flags(policy.requires)?;
            flags(policy.one_of)?;
            resolve_groups(policy.any_of)?;
            policy.when.iter().try_for_each(|w| -> anyhow::Result<()> {
                flag(w.if_present)?;
                flags(w.require)?;
                resolve_groups(w.any_of)?;
                Ok(())
            })?;

            // Every `one_of` member must live in an exclusive namespace. That
            // is what makes checking `one_of` before `when` fires safe: a
            // `when`-injected duplicate would otherwise slip past the
            // exactly-one count, but `check_no_exclusive_conflict` catches it.
            policy.one_of.iter().try_for_each(|member| {
                let namespace =
                    namespace_of(member).context("a one_of member must be namespaced")?;
                anyhow::ensure!(
                    EXCLUSIVE_NAMESPACES.contains(&namespace),
                    "one_of member {member:?} is not in an exclusive namespace, so a \
                     `when` injection could defeat the exactly-one guardrail"
                );
                Ok(())
            })
        })
    }

    #[test]
    fn ordered_vocabularies_have_no_duplicates() -> anyhow::Result<()> {
        // `Group::AtLeast` resolves a floor by first match, so a duplicate
        // entry would silently make the family's lower bound ambiguous.
        let mut seen = HashSet::new();
        IPHONES.iter().try_for_each(|entry| {
            anyhow::ensure!(seen.insert(*entry), "duplicate vocabulary entry {entry:?}");
            Ok(())
        })
    }

    /// `pipette-mgmt`'s `client::slugify`, mirrored: drop whitespace, lowercase.
    fn slugify(value: &str) -> String {
        value
            .chars()
            .filter(|c| !c.is_whitespace())
            .flat_map(char::to_lowercase)
            .collect()
    }

    /// Cases are brand strings rather than flags, so this fails if the family
    /// drifts from what a Mac actually reports.
    ///
    /// It regressed once: with `applem1`..`applem4` alone, an end-to-end ingest
    /// against a live M3 Max indexed zero eligible markers.
    #[rstest]
    #[case::m1_base("Apple M1")]
    #[case::m1_pro("Apple M1 Pro")]
    #[case::m1_max("Apple M1 Max")]
    #[case::m1_ultra("Apple M1 Ultra")]
    #[case::m2_base("Apple M2")]
    #[case::m2_pro("Apple M2 Pro")]
    #[case::m2_max("Apple M2 Max")]
    #[case::m2_ultra("Apple M2 Ultra")]
    #[case::m3_base("Apple M3")]
    #[case::m3_pro("Apple M3 Pro")]
    #[case::m3_max("Apple M3 Max")]
    #[case::m3_ultra("Apple M3 Ultra")]
    #[case::m4_base("Apple M4")]
    #[case::m4_pro("Apple M4 Pro")]
    #[case::m4_max("Apple M4 Max")]
    // Read off `boston-mbp-m5-1` (Mac17,6), a live fleet box.
    #[case::m5_max("Apple M5 Max")]
    fn real_macs_are_covered_by_the_apple_silicon_family(#[case] brand_string: &str) {
        let flag = format!("chip:{}", slugify(brand_string));
        assert!(
            APPLE_SILICON.contains(&flag.as_str()),
            "{brand_string:?} reports {flag:?}, which no MLX-on-macOS job would match"
        );
    }

    /// Every member has to be canonical, or the server rejects the job at
    /// ingestion — the flags are written into `any_of` verbatim.
    #[test]
    fn apple_silicon_members_are_canonical_and_unique() -> anyhow::Result<()> {
        let mut seen = HashSet::new();
        APPLE_SILICON.iter().try_for_each(|entry| {
            anyhow::ensure!(seen.insert(*entry), "duplicate vocabulary entry {entry:?}");
            CapabilityFlag::try_new((*entry).to_owned())
                .with_context(|| format!("{entry:?} is not a canonical capability flag"))?;
            Ok(())
        })
    }
}
