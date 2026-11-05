use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Context;

use pipette_plan_types::{HF_TOKEN_ENV, Plan, RunnableCell, TransportConfig};

use super::{
    host_semaphore::HostSemaphore,
    probe::{
        RejoinCallbacks, RejoinPhase, RejoinPolicy, RejoinStatus, probe_device, wait_for_device,
    },
    state_io::{append_local_state, ensure_plans_dir, load_state_index},
};
use crate::{
    runner::Shard,
    shell::RemoteExecRequest,
    state::{AttemptStatus, CellState, StateEvent, StateIndex, StateSummary},
    transport::Transport,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct RunOptions {
    /// Cap the number of cells to run this invocation.
    pub limit: Option<usize>,
    /// Re-queue cells that most recently failed.
    pub include_failed: bool,
    /// Re-queue cells that already succeeded. A re-run cell gains another
    /// attempt rather than replacing the recorded one.
    pub include_done: bool,
    /// JSON events on stdout, one per line, instead of summary lines.
    /// stderr logging is unchanged.
    pub json: bool,
    /// Run cells whose pinned transport is no longer in the plan, or no
    /// longer in their `allowed_clients`. Off by default to keep the
    /// on-device sample checkpoint; on restarts the cell from sample zero.
    pub reassign_stranded: bool,
    /// Ignore recorded pins and route by `allowed_clients` only, so
    /// stragglers that all last ran on one device fan out again. Cells
    /// restart from sample zero.
    pub ignore_pins: bool,
    /// `adb -P <port>` for every adb transport. `None` keeps each
    /// transport's own `port` (or the adb default).
    pub adb_port: Option<u16>,
    /// Forwarded as `PIPETTE_READINESS_MAX_WAIT_SECS` so off-box runners
    /// honor it; local runners inherit the driver env anyway.
    pub readiness_max_wait_secs: Option<u64>,
    /// Forwarded as `PIPETTE_READINESS_SKIP_THERMAL`: remote runners waive
    /// the thermal criterion but keep the load one.
    pub readiness_skip_thermal: bool,
    /// Select what an Android runner's gate does about a held touch. Forward
    /// the mode as `PIPETTE_READINESS_HELD_TOUCH`.
    pub readiness_held_touch: Option<pipette_readiness::HeldTouch>,
    /// How long a worker waits on a transport that stopped answering before
    /// giving up on it.
    pub rejoin: RejoinPolicy,
    /// Run only this shard of the ordered matrix. Sharding is by position
    /// before the state filter, so the partition stays stable as cells
    /// complete and independent invocations (one per SLURM job) can split it.
    pub shard: Option<Shard>,
}

/// Env forwarded to every remote invocation. The HF token comes from the
/// plan's gated/private models (the remote CLI injects it back into the
/// model definition), not from the driver env.
fn resolve_forwarded_env(
    plan: &Plan,
    readiness_max_wait_secs: Option<u64>,
    readiness_skip_thermal: bool,
    readiness_held_touch: Option<pipette_readiness::HeldTouch>,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut env = match plan.auth_token() {
        Some(token) => vec![(HF_TOKEN_ENV.to_string(), token.as_ref().to_string())],
        None => Vec::new(),
    };
    if let Some(secs) = readiness_max_wait_secs {
        env.push((
            pipette_readiness::MAX_WAIT_ENV.to_string(),
            secs.to_string(),
        ));
    }
    // Absent means "enforce", the runner's default, so an unset flag leaves
    // the remote env untouched.
    if readiness_skip_thermal {
        env.push((
            pipette_readiness::SKIP_THERMAL_ENV.to_string(),
            "1".to_string(),
        ));
    }
    if let Some(held_touch) = readiness_held_touch {
        env.push((
            "PIPETTE_READINESS_HELD_TOUCH".to_string(),
            held_touch.to_string(),
        ));
    }
    Ok(env)
}

#[derive(Debug, PartialEq, Eq)]
enum RouteOutcome {
    /// Also covers a cell with exactly one allowed client.
    Pinned(String),
    /// Sorted, so two cells with the same allowed set share one queue.
    Restricted(Vec<String>),
    Unassigned,
    Stranded(StrandedReason),
}

#[derive(Debug, PartialEq, Eq)]
enum StrandedReason {
    /// Pinned to a transport that is not in this run.
    PinnedAbsent(String),
    /// Pinned to a present transport that `allowed_clients` excludes.
    PinnedExcluded { label: String, allowed: Vec<String> },
    /// None of `allowed_clients` is in this run.
    NoOverlap { allowed: Vec<String> },
}

impl StrandedReason {
    fn bucket(&self) -> String {
        match self {
            Self::PinnedAbsent(label) => format!("absent:{label}"),
            Self::PinnedExcluded { label, .. } => format!("excluded-from-allowed:{label}"),
            Self::NoOverlap { .. } => "no-overlap".to_string(),
        }
    }

    fn eprintln_line(&self, count: usize) -> String {
        match self {
            Self::PinnedAbsent(label) => format!(
                "[stranded] {count} cell(s) pinned to {label}, which is not in this run's \
                 transports; skipping (use --reassign-stranded to run them on another \
                 device, losing their checkpoint)"
            ),
            Self::PinnedExcluded { label, allowed } => format!(
                "[stranded] {count} cell(s) pinned to {label} but their allowed_clients = \
                 {allowed:?} excludes that transport; skipping (use --reassign-stranded to \
                 reroute them, losing their checkpoint)"
            ),
            Self::NoOverlap { allowed } => format!(
                "[stranded] {count} cell(s) have allowed_clients = {allowed:?} with no \
                 overlap with this run's transports; skipping"
            ),
        }
    }
}

fn route_cell(
    allowed_clients: &[&str],
    state_pinned_label: Option<&str>,
    name_to_label: &HashMap<String, String>,
    present_labels: &HashSet<String>,
    reassign_stranded: bool,
) -> RouteOutcome {
    let mut allowed_labels: Vec<String> = allowed_clients
        .iter()
        .filter_map(|name| name_to_label.get(*name).cloned())
        .collect();
    allowed_labels.sort();
    let allowed_was_nonempty = !allowed_clients.is_empty();
    let allowed_has_overlap = !allowed_labels.is_empty();

    if allowed_was_nonempty && !allowed_has_overlap {
        return RouteOutcome::Stranded(StrandedReason::NoOverlap {
            allowed: allowed_clients.iter().map(|s| s.to_string()).collect(),
        });
    }

    if let Some(pinned) = state_pinned_label {
        if !present_labels.contains(pinned) {
            if reassign_stranded {
                return route_unpinned(allowed_labels);
            }
            return RouteOutcome::Stranded(StrandedReason::PinnedAbsent(pinned.to_string()));
        }
        if allowed_has_overlap && !allowed_labels.iter().any(|l| l == pinned) {
            if reassign_stranded {
                return route_unpinned(allowed_labels);
            }
            return RouteOutcome::Stranded(StrandedReason::PinnedExcluded {
                label: pinned.to_string(),
                allowed: allowed_labels,
            });
        }
        return RouteOutcome::Pinned(pinned.to_string());
    }

    route_unpinned(allowed_labels)
}

fn route_unpinned(mut allowed_labels: Vec<String>) -> RouteOutcome {
    match allowed_labels.len() {
        0 => RouteOutcome::Unassigned,
        1 => RouteOutcome::Pinned(allowed_labels.remove(0)),
        _ => RouteOutcome::Restricted(allowed_labels),
    }
}

fn parent_dir_of_binary(binary_path: &str) -> String {
    if binary_path.contains('\\') {
        match binary_path.rsplit_once('\\') {
            Some((parent, _)) if !parent.is_empty() => parent.to_string(),
            _ => ".".to_string(),
        }
    } else {
        match binary_path.rsplit_once('/') {
            Some((parent, _)) if !parent.is_empty() => parent.to_string(),
            _ => ".".to_string(),
        }
    }
}

/// `parallelism` workers per `[[transports]]` entry, each with its config
/// so it builds argv with its own `binary_path`/`work_dir`.
fn build_workers(plan: &Plan, adb_port: Option<u16>) -> Vec<(Transport, &TransportConfig)> {
    plan.transports
        .iter()
        .flat_map(|cfg| {
            (0..cfg.parallelism().get())
                .map(move |_| (Transport::from_config_with_adb_port(cfg, adb_port), cfg))
        })
        .collect()
}

/// Per-`physical_id` [`HostSemaphore`] capacity: the max `parallelism` of
/// any transport reaching that box, so co-located `parallelism = 1`
/// transports still serialize.
pub(super) fn physical_id_capacities(plan: &Plan) -> HashMap<String, NonZeroUsize> {
    plan.transports
        .iter()
        .fold(HashMap::new(), |mut caps, cfg| {
            let p = cfg.parallelism();
            caps.entry(cfg.physical_id())
                .and_modify(|c: &mut NonZeroUsize| *c = (*c).max(p))
                .or_insert(p);
            caps
        })
}

type CellQueue = Arc<Mutex<VecDeque<RunnableItem>>>;

/// `(cell_idx, attempt)`. `cell_idx` indexes the worker's own `work`.
type RunnableItem = (usize, usize);

fn start_event(
    plan_id: &str,
    total: usize,
    transport_labels: &[String],
    stranded: &BTreeMap<String, usize>,
) -> serde_json::Value {
    let stranded_obj: serde_json::Map<String, serde_json::Value> = stranded
        .iter()
        .map(|(label, count)| (label.clone(), serde_json::json!(count)))
        .collect();
    serde_json::json!({
        "event": "start",
        "plan_id": plan_id,
        "total": total,
        "transports": transport_labels,
        "stranded": stranded_obj,
    })
}

fn cell_event(
    plan_id: &str,
    transport_label: &str,
    cell: &RunnableCell,
    attempt: usize,
    exit_code: i32,
) -> serde_json::Value {
    serde_json::json!({
        "event": "cell",
        "plan_id": plan_id,
        "transport": transport_label,
        "attempt": attempt,
        "status": if exit_code == 0 { "success" } else { "failed" },
        "exit_code": exit_code,
        "benchmark": cell.benchmark.as_ref(),
        "model": cell.model.to_string(),
        "runtime": cell.runtime.to_string(),
    })
}

/// A worker waiting on a box that stopped answering, one line per turn, so
/// a `--json` stream does not go silent for minutes unexplained. Every key
/// is always present, `null` where the phase has nothing for it.
fn device_event(plan_id: &str, status: &RejoinStatus<'_>) -> serde_json::Value {
    let retry_in_secs = match &status.phase {
        RejoinPhase::Waiting { retry_in } => Some(retry_in.as_secs()),
        RejoinPhase::Refused { .. }
        | RejoinPhase::Unreachable { .. }
        | RejoinPhase::Online
        | RejoinPhase::GaveUp => None,
    };
    let exit_code = match &status.phase {
        RejoinPhase::Refused { exit_code } => Some(*exit_code),
        RejoinPhase::Waiting { .. }
        | RejoinPhase::Unreachable { .. }
        | RejoinPhase::Online
        | RejoinPhase::GaveUp => None,
    };
    let error = match &status.phase {
        RejoinPhase::Unreachable { error } => Some(error.as_str()),
        RejoinPhase::Waiting { .. }
        | RejoinPhase::Refused { .. }
        | RejoinPhase::Online
        | RejoinPhase::GaveUp => None,
    };
    serde_json::json!({
        "event": "device",
        "plan_id": plan_id,
        "transport": status.label,
        "state": status.phase.as_str(),
        "retry": status.retry,
        "max_retries": status.policy.max_retries,
        "retry_in_secs": retry_in_secs,
        "exit_code": exit_code,
        "error": error,
    })
}

fn end_event(plan_id: &str, summary: &StateSummary) -> serde_json::Value {
    serde_json::json!({
        "event": "end",
        "plan_id": plan_id,
        "done": summary.done,
        "failed": summary.failed,
        "missing": summary.missing,
    })
}

/// A total order, model first, so a worker finishes one model's sweep
/// before the next and runs are reproducible (`runnable_cells` is a
/// `HashSet`).
pub(crate) fn order_cells(mut cells: Vec<RunnableCell>) -> Vec<RunnableCell> {
    cells.sort_by_cached_key(cell_order_key);
    cells
}

fn cell_order_key(c: &RunnableCell) -> (String, String, String, String, Vec<String>, Vec<String>) {
    let mut clients: Vec<String> = c
        .allowed_clients
        .iter()
        .map(|c| c.as_ref().to_string())
        .collect();
    clients.sort();
    (
        c.model.to_string(),
        c.model_flags
            .as_ref()
            .and_then(|f| f.canonical_string())
            .unwrap_or_default(),
        c.benchmark.as_ref().to_string(),
        c.runtime.to_string(),
        // Canonical, beside `model_flags` above: ordering decides which cells a
        // `--limit` run reaches, so it should not shuffle when the wire form changes.
        c.runtime_flags_canonical_string()
            .into_iter()
            .collect::<Vec<_>>(),
        clients,
    )
}

/// The cells this invocation should execute, in matrix order. The index
/// is the position in the full ordered matrix, so shards stay stable as
/// cells complete.
fn select_runnable(
    cells: &[RunnableCell],
    state: &StateIndex,
    opts: &RunOptions,
) -> Vec<RunnableItem> {
    cells
        .iter()
        .enumerate()
        .filter(|(idx, _)| opts.shard.is_none_or(|s| s.contains(*idx)))
        .filter_map(|(idx, cell)| {
            let attempts = state.attempts_for(cell);
            match state.state_for(cell) {
                CellState::Done if !opts.include_done => None,
                // Continues the recorded attempt history, unlike the Failed arm below.
                CellState::Done => Some((idx, attempts + 1)),
                CellState::Failed if !opts.include_failed => None,
                CellState::Failed => Some((idx, 1)),
                CellState::Missing => Some((idx, attempts + 1)),
            }
        })
        .collect()
}

pub fn run_matrix(plans_dir: &Path, plan: &Plan, opts: RunOptions) -> anyhow::Result<()> {
    // Destructured so a new `RunOptions` field must be acknowledged here; the
    // `_` fields are read off `opts` by `select_runnable`.
    let RunOptions {
        limit,
        include_failed: _,
        include_done: _,
        json,
        reassign_stranded,
        ignore_pins,
        adb_port,
        readiness_max_wait_secs,
        readiness_skip_thermal,
        readiness_held_touch,
        rejoin,
        shard: _,
    } = opts;
    let forwarded_env = resolve_forwarded_env(
        plan,
        readiness_max_wait_secs,
        readiness_skip_thermal,
        readiness_held_touch,
    )?;
    let cells: Vec<RunnableCell> = order_cells(plan.runnable_cells()?.into_iter().collect());
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    let workers = build_workers(plan, adb_port);
    if workers.is_empty() {
        anyhow::bail!("plan has no transports configured");
    }

    // Keyed on `client_id`, not the target: two transports reaching one box
    // with different binaries need separate queues, or a cell lands on the
    // wrong binary.
    let name_to_label: HashMap<String, String> = plan
        .transports
        .iter()
        .map(|cfg| (cfg.client_id().to_string(), cfg.client_id().to_string()))
        .collect();
    let present_labels: HashSet<String> = name_to_label.values().cloned().collect();

    let runnable = select_runnable(&cells, &state, &opts);

    let total_before_limit = runnable.len();
    let total = match limit {
        Some(limit) => limit.min(total_before_limit),
        None => total_before_limit,
    };
    let runnable = runnable.into_iter().take(total);

    let mut by_transport: BTreeMap<String, VecDeque<RunnableItem>> = present_labels
        .iter()
        .map(|l| (l.clone(), VecDeque::new()))
        .collect();
    let mut restricted: BTreeMap<Vec<String>, VecDeque<RunnableItem>> = BTreeMap::new();
    let mut unassigned: VecDeque<RunnableItem> = VecDeque::new();
    let mut stranded: BTreeMap<String, (StrandedReason, usize)> = BTreeMap::new();

    for (idx, attempt) in runnable {
        let cell = cells
            .get(idx)
            .context("select_runnable indexes into the same cells")?;
        let allowed: Vec<&str> = cell.allowed_clients.iter().map(|c| c.as_ref()).collect();
        let pinned = if ignore_pins {
            None
        } else {
            state.pinned_transport_for(cell)
        };
        match route_cell(
            &allowed,
            pinned,
            &name_to_label,
            &present_labels,
            reassign_stranded,
        ) {
            RouteOutcome::Pinned(label) => {
                by_transport
                    .get_mut(&label)
                    .context("pre-seeded queue missing for present label")?
                    .push_back((idx, attempt));
            }
            RouteOutcome::Restricted(labels) => {
                restricted
                    .entry(labels)
                    .or_default()
                    .push_back((idx, attempt));
            }
            RouteOutcome::Unassigned => unassigned.push_back((idx, attempt)),
            RouteOutcome::Stranded(reason) => {
                let key = reason.bucket();
                let entry = stranded.entry(key).or_insert((reason, 0));
                entry.1 += 1;
            }
        }
    }

    let runnable_count = by_transport.values().map(|q| q.len()).sum::<usize>()
        + restricted.values().map(|q| q.len()).sum::<usize>()
        + unassigned.len();

    let stranded_counts: BTreeMap<String, usize> =
        stranded.iter().map(|(k, (_, n))| (k.clone(), *n)).collect();
    let stranded_total: usize = stranded_counts.values().sum();

    if json {
        let labels: Vec<String> = workers.iter().map(|(t, _)| t.target_label()).collect();
        println!(
            "{}",
            start_event(&plan.plan_id, runnable_count, &labels, &stranded_counts)
        );
    } else if runnable_count == 0 && stranded.is_empty() {
        println!("nothing to run");
    } else {
        if runnable_count > 0 {
            println!(
                "{} cells to run across {} worker(s)",
                runnable_count,
                workers.len()
            );
            for (label, q) in &by_transport {
                if !q.is_empty() {
                    println!("  pinned to {label}: {}", q.len());
                }
            }
            for (labels, q) in &restricted {
                if !q.is_empty() {
                    println!("  restricted to {labels:?}: {}", q.len());
                }
            }
            if !unassigned.is_empty() {
                println!("  unassigned: {}", unassigned.len());
            }
        } else {
            println!("0 runnable, {stranded_total} stranded; see stderr");
        }
        for (reason, count) in stranded.values() {
            eprintln!("{}", reason.eprintln_line(*count));
        }
        if let Some(limit) = limit
            && runnable_count < limit
            && stranded_total > 0
        {
            eprintln!(
                "[stranded] --limit {limit} included {stranded_total} stranded \
                     cell(s); only {runnable_count} will actually run this invocation"
            );
        }
    }

    if runnable_count > 0 {
        ensure_plans_dir(plans_dir, &plan.plan_id)?;
        let by_transport: HashMap<String, CellQueue> = by_transport
            .into_iter()
            .map(|(k, v)| (k, Arc::new(Mutex::new(v))))
            .collect();
        let restricted: BTreeMap<Vec<String>, CellQueue> = restricted
            .into_iter()
            .map(|(k, v)| (k, Arc::new(Mutex::new(v))))
            .collect();
        let unassigned: CellQueue = Arc::new(Mutex::new(unassigned));
        let host_locks: HashMap<String, Arc<HostSemaphore>> = physical_id_capacities(plan)
            .into_iter()
            .map(|(key, cap)| (key, Arc::new(HostSemaphore::new(cap))))
            .collect();
        // `sync` pulls benchmark definitions as well as pushing results, so
        // running it first is what makes a cell benchmark the current
        // definition rather than whatever that box last pulled.
        run_sync_pass(plan, &forwarded_env, adb_port, SyncPhase::Initial)?;
        execute_queue(
            plan,
            plans_dir,
            workers,
            by_transport,
            restricted,
            unassigned,
            host_locks,
            &cells,
            runnable_count,
            &forwarded_env,
            json,
            rejoin,
        )?;
        run_sync_pass(plan, &forwarded_env, adb_port, SyncPhase::Final)?;
    }

    let final_state = load_state_index(plans_dir, &plan.plan_id)?;
    let summary = final_state.summary_for(&cells);
    if json {
        println!("{}", end_event(&plan.plan_id, &summary));
    } else if runnable_count > 0 {
        println!("done: {}", summary.done);
        println!("failed: {}", summary.failed);
        println!("missing: {}", summary.missing);
    }
    Ok(())
}

/// Which queue an item came from, so a worker whose probe failed returns
/// it there and keeps eligibility correct.
#[derive(Debug, Clone)]
enum Origin {
    Pinned,
    Restricted(CellQueue),
    Unassigned,
}

/// Pinned first, then `eligible_restricted` in the caller's order, then the
/// shared pool.
fn pop_next(
    pinned: &CellQueue,
    eligible_restricted: &[CellQueue],
    unassigned: &CellQueue,
) -> Option<(RunnableItem, Origin)> {
    if let Some(item) = pinned.lock().unwrap_or_else(|p| p.into_inner()).pop_front() {
        return Some((item, Origin::Pinned));
    }
    for q in eligible_restricted {
        if let Some(item) = q.lock().unwrap_or_else(|p| p.into_inner()).pop_front() {
            return Some((item, Origin::Restricted(Arc::clone(q))));
        }
    }
    unassigned
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .pop_front()
        .map(|item| (item, Origin::Unassigned))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the queue worker's whole context"
)]
fn execute_queue(
    plan: &Plan,
    plans_dir: &Path,
    workers: Vec<(Transport, &TransportConfig)>,
    by_transport: HashMap<String, CellQueue>,
    restricted: BTreeMap<Vec<String>, CellQueue>,
    unassigned: CellQueue,
    host_locks: HashMap<String, Arc<HostSemaphore>>,
    cells: &[RunnableCell],
    total: usize,
    forwarded_env: &[(String, String)],
    json: bool,
    rejoin: RejoinPolicy,
) -> anyhow::Result<()> {
    let state_lock = Arc::new(Mutex::new(()));
    let counter = Arc::new(Mutex::new(0usize));
    let by_transport = Arc::new(by_transport);
    let forwarded_env = Arc::new(forwarded_env.to_vec());

    let handles: Vec<_> = workers
        .into_iter()
        .map(|(transport, transport_cfg)| {
            let work: Vec<(RunnableCell, Vec<String>)> = cells
                .iter()
                .map(|cell| {
                    let mut argv = cell.build_argv(transport_cfg)?;
                    if transport_cfg.appends_sync_flag() {
                        argv.push("--sync".to_string());
                    }
                    Ok((cell.clone(), argv))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let state_lock = Arc::clone(&state_lock);
            let counter = Arc::clone(&counter);
            let plan_id = plan.plan_id.clone();
            let binary_path = transport.binary_path().to_string();
            let plans_dir = plans_dir.to_path_buf();
            let max_consecutive = plan.retry.max_consecutive_failures;
            // `route_key` is persisted as the cell's `transport_label` so
            // pinning round-trips; `label` is only for display.
            let route_key = transport_cfg.client_id().to_string();
            let label = transport.target_label();
            let pinned_queue = by_transport
                .get(&route_key)
                .cloned()
                .unwrap_or_else(|| Arc::new(Mutex::new(VecDeque::new())));
            // Narrowest first, so cells with few permitted transports don't
            // starve behind broader ones.
            let eligible_restricted: Vec<CellQueue> = {
                let mut by_size: Vec<(&Vec<String>, &CellQueue)> = restricted
                    .iter()
                    .filter(|(keys, _)| keys.contains(&route_key))
                    .collect();
                by_size.sort_by(|(a, _), (b, _)| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
                by_size.into_iter().map(|(_, q)| Arc::clone(q)).collect()
            };
            let host_lock = host_locks
                .get(&transport_cfg.physical_id())
                .cloned()
                .context("host_lock present for every transport's physical_id")?;
            let unassigned = Arc::clone(&unassigned);
            let forwarded_env = Arc::clone(&forwarded_env);

            Ok(std::thread::spawn(move || {
                let prefix = Some(label.as_str());
                let mut consecutive_failures = 0usize;
                // Without `--json` the status writes the stderr line itself.
                // `--json` takes the narration over to add the `device` event
                // beside that same line, which it still writes.
                let report = |status: &RejoinStatus<'_>| {
                    eprintln!("{status}");
                    println!("{}", device_event(&plan_id, status));
                };
                let rejoin_callbacks = if json {
                    RejoinCallbacks::new().with_status(&report)
                } else {
                    RejoinCallbacks::new()
                };

                while let Some(((idx, attempt), origin)) =
                    pop_next(&pinned_queue, &eligible_restricted, &unassigned)
                {
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "queue items are `enumerate()` indices into the `cells` that `work` was built from"
                    )]
                    let (cell, argv) = &work[idx];

                    if !probe_device(&transport, &label) {
                        let return_queue: &CellQueue = match &origin {
                            Origin::Pinned => &pinned_queue,
                            Origin::Restricted(q) => q,
                            Origin::Unassigned => &unassigned,
                        };
                        push_back(return_queue, (idx, attempt));
                        if wait_for_device(&transport, &label, rejoin, &rejoin_callbacks) {
                            // The box may have missed the initial sync, so it
                            // could otherwise benchmark stale definitions.
                            let _host_guard = host_lock.acquire();
                            sync_transport(&transport, &forwarded_env, SyncPhase::Rejoin);
                            continue;
                        }
                        break;
                    }

                    let n = {
                        let mut c = counter.lock().unwrap_or_else(|p| p.into_inner());
                        *c += 1;
                        *c
                    };

                    eprintln!(
                        "[{label}] [{n}/{total}] benchmark: {} | model: {}",
                        cell.benchmark.as_ref(),
                        cell.model
                    );

                    if !write_event(
                        &state_lock,
                        &plans_dir,
                        &plan_id,
                        cell,
                        AttemptStatus::Started,
                        attempt,
                        &route_key,
                        &label,
                        None,
                        json,
                        false,
                    ) {
                        break;
                    }

                    let request = RemoteExecRequest {
                        argv: argv.clone(),
                        env: (*forwarded_env).clone(),
                        cwd: Some(parent_dir_of_binary(&binary_path)),
                        job_name: Some(format!("{}/{}", cell.benchmark.as_ref(), cell.model)),
                    };
                    transport
                        .preview_exec(&request)
                        .lines()
                        .for_each(|line| eprintln!("[{label}] {line}"));

                    let output = {
                        let _host_guard = host_lock.acquire();
                        transport.exec(request, prefix)
                    };
                    let output = match output {
                        Ok(o) => o,
                        Err(e) => {
                            eprintln!("[{label}] transport error: {e:#}");
                            push_back(&pinned_queue, (idx, attempt));
                            if wait_for_device(&transport, &label, rejoin, &rejoin_callbacks) {
                                let _host_guard = host_lock.acquire();
                                sync_transport(&transport, &forwarded_env, SyncPhase::Rejoin);
                                continue;
                            }
                            break;
                        }
                    };

                    let status = if output.status == 0 {
                        AttemptStatus::Success
                    } else {
                        AttemptStatus::Failed
                    };
                    let exit_code = if output.status == 0 {
                        None
                    } else {
                        Some(output.status)
                    };
                    if !write_event(
                        &state_lock,
                        &plans_dir,
                        &plan_id,
                        cell,
                        status,
                        attempt,
                        &route_key,
                        &label,
                        exit_code,
                        json,
                        true,
                    ) {
                        break;
                    }

                    if output.status != 0 {
                        consecutive_failures += 1;
                        eprintln!(
                            "[{label}] cell failed (exit {}): benchmark={} model={} runtime={}",
                            output.status,
                            cell.benchmark.as_ref(),
                            cell.model,
                            cell.runtime,
                        );
                        if max_consecutive > 0 && consecutive_failures >= max_consecutive {
                            eprintln!(
                                "[{label}] stopping: {consecutive_failures} consecutive failures"
                            );
                            break;
                        }
                    } else {
                        consecutive_failures = 0;
                    }
                }
            }))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    handles.into_iter().for_each(|handle| {
        if let Err(panic) = handle.join() {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            eprintln!("worker panicked: {msg}");
        }
    });
    Ok(())
}

fn push_back(queue: &CellQueue, item: RunnableItem) {
    if let Ok(mut q) = queue.lock() {
        q.push_back(item);
    }
}

/// Returns `false` if the write failed; the worker then stops. Started
/// events pass `emit_json_cell = false` so the `--json` stream keeps only
/// finished cells.
#[expect(
    clippy::too_many_arguments,
    reason = "one event carries every field it reports"
)]
fn write_event(
    state_lock: &Mutex<()>,
    plans_dir: &Path,
    plan_id: &str,
    cell: &RunnableCell,
    status: AttemptStatus,
    attempt: usize,
    route_key: &str,
    display_label: &str,
    exit_code: Option<i32>,
    json: bool,
    emit_json_cell: bool,
) -> bool {
    let mut event = match StateEvent::new(plan_id, cell, status, attempt, route_key) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[{display_label}] state error: {e:#}");
            return false;
        }
    };
    event.exit_code = exit_code;

    let _guard = state_lock.lock().unwrap_or_else(|p| p.into_inner());
    let json_line = match event.to_json_line() {
        Ok(line) => line,
        Err(e) => {
            eprintln!("[{display_label}] failed to serialize state event: {e:#}");
            return false;
        }
    };
    if let Err(e) = append_local_state(plans_dir, plan_id, &json_line) {
        eprintln!("[{display_label}] failed to write state: {e:#}");
        return false;
    }
    if json && emit_json_cell {
        println!(
            "{}",
            cell_event(
                plan_id,
                display_label,
                cell,
                attempt,
                exit_code.unwrap_or(0)
            )
        );
    }
    true
}

/// One target per `[[transports]]` entry, minus both iOS transports. iOS
/// uploads each cell inline (`submit=1`), and a sync there would launch the
/// app without the `headlessrun` marker: it opens the UI and never exits.
fn sync_targets(plan: &Plan, adb_port: Option<u16>) -> Vec<(Transport, &TransportConfig)> {
    plan.transports
        .iter()
        .map(|cfg| (Transport::from_config_with_adb_port(cfg, adb_port), cfg))
        .filter(|(transport, _)| {
            !matches!(
                transport,
                Transport::Ios { .. } | Transport::IosOverSsh { .. }
            )
        })
        .collect()
}

/// When in a run a sync happens, which decides whether an unreachable box
/// is worth waiting on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SyncPhase {
    /// Before the first cell: pulls definitions, flushes results a previous
    /// run left queued, then runs `storage gc`. Probe-gated, since every
    /// cell waits behind this pass.
    Initial,
    /// A box that stopped answering has come back mid-run: pulls the
    /// definitions it may have missed before it resumes.
    Rejoin,
    /// After the last cell. Attempted even against a flaky box: nothing
    /// waits on it, and results are at stake.
    Final,
}

impl SyncPhase {
    fn as_str(self) -> &'static str {
        match self {
            SyncPhase::Initial => "initial",
            SyncPhase::Rejoin => "rejoin",
            SyncPhase::Final => "final",
        }
    }
}

/// One `sync` per `[[transports]]` entry, whatever its `parallelism`.
/// Transports sync concurrently under the cell run's per-`physical_id`
/// budget: a box that tolerates one cell at a time tolerates one sync. A
/// pass never overlaps the cell run, so its slots never stack with the run's.
fn run_sync_pass(
    plan: &Plan,
    forwarded_env: &[(String, String)],
    adb_port: Option<u16>,
    phase: SyncPhase,
) -> anyhow::Result<()> {
    let host_locks: HashMap<String, HostSemaphore> = physical_id_capacities(plan)
        .into_iter()
        .map(|(key, cap)| (key, HostSemaphore::new(cap)))
        .collect();
    // Resolved before any thread starts, so a gap in the capacity map fails
    // the pass instead of leaving a box unthrottled.
    let jobs = sync_targets(plan, adb_port)
        .into_iter()
        .map(|(transport, cfg)| {
            let lock = host_locks
                .get(&cfg.physical_id())
                .context("host lock present for every transport's physical_id")?;
            Ok((transport, lock))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    std::thread::scope(|scope| {
        jobs.into_iter().for_each(|(transport, lock)| {
            scope.spawn(move || {
                let _guard = lock.acquire();
                // Probe, don't wait: that box's worker already waits on it
                // and re-syncs on `SyncPhase::Rejoin`.
                let label = transport.target_label();
                if phase == SyncPhase::Initial && !probe_device(&transport, &label) {
                    eprintln!(
                        "[{label}] skipping the initial sync; syncing instead when it rejoins"
                    );
                    return;
                }
                sync_transport(&transport, forwarded_env, phase);
            });
        });
    });
    Ok(())
}

/// On the initial pass, `storage gc` runs after `sync`: a queued result pins
/// the artifacts it names, so syncing first frees them and the box starts
/// with headroom instead of hitting its quota mid-run.
fn sync_transport(transport: &Transport, forwarded_env: &[(String, String)], phase: SyncPhase) {
    let label = transport.target_label();
    run_client_command(
        transport,
        &label,
        forwarded_env,
        &["sync"],
        &format!("{} sync", phase.as_str()),
    );
    if phase == SyncPhase::Initial {
        run_client_command(
            transport,
            &label,
            forwarded_env,
            &["storage", "gc"],
            "storage gc",
        );
    }
}

fn client_argv(binary_path: &str, work_dir: &str, args: &[&str]) -> Vec<String> {
    [binary_path, "--work-dir", work_dir]
        .into_iter()
        .chain(args.iter().copied())
        .map(str::to_string)
        .collect()
}

/// Failures are reported, not propagated: one unreachable box must not
/// strand the rest of the fleet.
fn run_client_command(
    transport: &Transport,
    label: &str,
    forwarded_env: &[(String, String)],
    args: &[&str],
    what: &str,
) {
    let binary_path = transport.binary_path().to_string();
    let request = RemoteExecRequest {
        argv: client_argv(&binary_path, transport.work_dir(), args),
        env: forwarded_env.to_vec(),
        cwd: Some(parent_dir_of_binary(&binary_path)),
        job_name: Some(args.join("-")),
    };
    eprintln!("[{label}] {what}...");
    match transport.exec(request, Some(label)) {
        Ok(o) if o.status != 0 => {
            eprintln!("[{label}] warning: {what} exit code {}", o.status);
        }
        Err(e) => {
            eprintln!("[{label}] warning: {what} failed: {e:#}");
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;

    use pipette_plan_types::{
        BenchmarkId, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, LlamaCppFlavor,
        LlamacppCliStockTools, LlamacppCliStockToolsRelease, LlamacppCliStockToolsSource, Model,
        NonEmptyString, RepositoryUrl, Runtime,
    };

    use super::*;

    fn sample_cell() -> anyhow::Result<RunnableCell> {
        let model = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("org".to_string()).context("org")?,
                    repo_name: HfRepoName::try_new("repo".to_string()).context("repo_name")?,
                    revision: None,
                    auth_token: None,
                },
                path: pipette_plan_types::RepoSubpath::try_new("file.gguf".to_string())
                    .context("filename")?,
                sha256: None,
            },
        });
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("rt".to_string()).context("version")?,
                asset: None,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        Ok(RunnableCell {
            benchmark: BenchmarkId::try_new("bench".to_string()).context("benchmark")?,
            model,
            runtime,
            allowed_clients: Vec::new(),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
        })
    }

    fn cell_named(repo: &str, benchmark: &str) -> anyhow::Result<RunnableCell> {
        let mut c = sample_cell()?;
        c.model = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("org".to_string()).context("org")?,
                    repo_name: HfRepoName::try_new(repo.to_string()).context("repo_name")?,
                    revision: None,
                    auth_token: None,
                },
                path: pipette_plan_types::RepoSubpath::try_new("file.gguf".to_string())
                    .context("filename")?,
                sha256: None,
            },
        });
        c.benchmark = BenchmarkId::try_new(benchmark.to_string()).context("benchmark")?;
        Ok(c)
    }

    fn state_over(entries: &[(&RunnableCell, AttemptStatus)]) -> anyhow::Result<StateIndex> {
        let raw: String = entries
            .iter()
            .map(|(cell, status)| StateEvent::new("p", cell, *status, 1, "local")?.to_json_line())
            .collect::<anyhow::Result<String>>()?;
        StateIndex::load(Some(&raw), "p")
    }

    fn include_opts(include_failed: bool, include_done: bool) -> RunOptions {
        RunOptions {
            include_failed,
            include_done,
            ..RunOptions::default()
        }
    }

    #[test]
    fn include_done_selects_the_cells_the_default_pass_skips() -> anyhow::Result<()> {
        let cells = vec![
            cell_named("done", "b1")?,
            cell_named("failed", "b1")?,
            cell_named("missing", "b1")?,
        ];
        let state = state_over(&[
            (&cells[0], AttemptStatus::Success),
            (&cells[1], AttemptStatus::Failed),
        ])?;

        assert_eq!(
            select_runnable(&cells, &state, &include_opts(false, false)),
            vec![(2, 1)],
            "default pass runs only the missing cell"
        );
        assert_eq!(
            select_runnable(&cells, &state, &include_opts(true, false)),
            vec![(1, 1), (2, 1)],
            "--include-failed adds the failed cell but not the done one"
        );
        assert_eq!(
            select_runnable(&cells, &state, &include_opts(false, true)),
            vec![(0, 2), (2, 1)],
            "--include-done adds the done cell but not the failed one"
        );
        assert_eq!(
            select_runnable(&cells, &state, &include_opts(true, true)),
            vec![(0, 2), (1, 1), (2, 1)],
            "both flags run the whole matrix"
        );
        Ok(())
    }

    #[test]
    fn include_done_still_runs_only_its_own_shard() -> anyhow::Result<()> {
        let cells = vec![
            cell_named("done", "b1")?,
            cell_named("also-done", "b1")?,
            cell_named("done-too", "b1")?,
        ];
        let state = state_over(&[
            (&cells[0], AttemptStatus::Success),
            (&cells[1], AttemptStatus::Success),
            (&cells[2], AttemptStatus::Success),
        ])?;
        let opts = RunOptions {
            shard: Some(Shard { index: 0, count: 2 }),
            ..include_opts(false, true)
        };
        assert_eq!(select_runnable(&cells, &state, &opts), vec![(0, 2), (2, 2)]);
        Ok(())
    }

    #[test]
    fn order_cells_groups_by_model_and_is_deterministic() -> anyhow::Result<()> {
        let build = || -> anyhow::Result<Vec<RunnableCell>> {
            Ok(vec![
                cell_named("beta", "b2")?,
                cell_named("alpha", "b2")?,
                cell_named("beta", "b1")?,
                cell_named("alpha", "b1")?,
            ])
        };
        let ordered = order_cells(build()?);
        let keys: Vec<(String, String)> = ordered
            .iter()
            .map(|c| (c.model.to_string(), c.benchmark.as_ref().to_string()))
            .collect();

        let mut want = keys.clone();
        want.sort();
        assert_eq!(keys, want, "cells must be in model-first sorted order");

        assert_eq!(keys[0].0, keys[1].0);
        assert_eq!(keys[2].0, keys[3].0);
        assert_ne!(keys[1].0, keys[2].0);

        let mut reversed = build()?;
        reversed.reverse();
        assert_eq!(order_cells(reversed), ordered);
        Ok(())
    }

    #[test]
    fn start_event_shape() -> anyhow::Result<()> {
        let ev = start_event(
            "my-plan",
            42,
            &["adb:ABC".into(), "ssh:host".into()],
            &BTreeMap::new(),
        );
        assert_eq!(ev["event"], "start");
        assert_eq!(ev["plan_id"], "my-plan");
        assert_eq!(ev["total"], 42);
        assert_eq!(ev["transports"][0], "adb:ABC");
        assert_eq!(ev["transports"][1], "ssh:host");
        assert!(ev["stranded"].is_object());
        assert_eq!(
            ev["stranded"]
                .as_object()
                .context("stranded should be an object")?
                .len(),
            0
        );
        Ok(())
    }

    #[test]
    fn start_event_includes_stranded_counts() {
        let mut stranded = BTreeMap::new();
        stranded.insert("adb:OLD".to_string(), 4);
        stranded.insert("ssh:gone".to_string(), 1);
        let ev = start_event("my-plan", 6, &["adb:NEW".into()], &stranded);
        assert_eq!(ev["total"], 6);
        assert_eq!(ev["stranded"]["adb:OLD"], 4);
        assert_eq!(ev["stranded"]["ssh:gone"], 1);
    }

    #[test]
    fn cell_event_shape_success() -> anyhow::Result<()> {
        let cell = sample_cell()?;
        let ev = cell_event("my-plan", "adb:ABC", &cell, 1, 0);
        assert_eq!(ev["event"], "cell");
        assert_eq!(ev["plan_id"], "my-plan");
        assert_eq!(ev["transport"], "adb:ABC");
        assert_eq!(ev["attempt"], 1);
        assert_eq!(ev["status"], "success");
        assert_eq!(ev["exit_code"], 0);
        assert_eq!(ev["benchmark"], "bench");
        assert_eq!(ev["model"], "org/repo:file.gguf");
        assert_eq!(
            ev["runtime"],
            "github.com/ggml-org/llama.cpp@rt:macos-arm64"
        );
        assert!(ev["mmproj"].is_null());
        Ok(())
    }

    #[test]
    fn cell_event_shape_failure() -> anyhow::Result<()> {
        let cell = sample_cell()?;
        let ev = cell_event("my-plan", "ssh:host", &cell, 2, 137);
        assert_eq!(ev["status"], "failed");
        assert_eq!(ev["exit_code"], 137);
        assert_eq!(ev["attempt"], 2);
        Ok(())
    }

    #[rstest]
    #[case::waiting(
        RejoinPhase::Waiting { retry_in: Duration::from_secs(30) },
        "waiting",
        serde_json::json!(30),
        serde_json::Value::Null,
        serde_json::Value::Null,
    )]
    #[case::refused(
        RejoinPhase::Refused { exit_code: 127 },
        "refused",
        serde_json::Value::Null,
        serde_json::json!(127),
        serde_json::Value::Null,
    )]
    #[case::unreachable(
        RejoinPhase::Unreachable { error: "no route to host".to_string() },
        "unreachable",
        serde_json::Value::Null,
        serde_json::Value::Null,
        serde_json::json!("no route to host"),
    )]
    #[case::online(
        RejoinPhase::Online,
        "online",
        serde_json::Value::Null,
        serde_json::Value::Null,
        serde_json::Value::Null
    )]
    #[case::gave_up(
        RejoinPhase::GaveUp,
        "gave_up",
        serde_json::Value::Null,
        serde_json::Value::Null,
        serde_json::Value::Null
    )]
    fn device_event_shape(
        #[case] phase: RejoinPhase,
        #[case] want_state: &str,
        #[case] want_retry_in_secs: serde_json::Value,
        #[case] want_exit_code: serde_json::Value,
        #[case] want_error: serde_json::Value,
    ) {
        let policy = RejoinPolicy {
            retry_delay: Duration::from_secs(30),
            max_retries: 10,
        };
        let ev = device_event(
            "my-plan",
            &RejoinStatus {
                label: "adb:ABC",
                phase,
                retry: 3,
                policy,
            },
        );
        assert_eq!(ev["event"], "device");
        assert_eq!(ev["plan_id"], "my-plan");
        assert_eq!(ev["transport"], "adb:ABC");
        assert_eq!(ev["state"], want_state);
        assert_eq!(ev["retry"], 3);
        assert_eq!(ev["max_retries"], 10);
        assert_eq!(ev["retry_in_secs"], want_retry_in_secs);
        assert_eq!(ev["exit_code"], want_exit_code);
        assert_eq!(ev["error"], want_error);
    }

    #[test]
    fn end_event_shape() {
        let summary = StateSummary {
            total: 10,
            done: 7,
            failed: 1,
            missing: 2,
        };
        let ev = end_event("my-plan", &summary);
        assert_eq!(ev["event"], "end");
        assert_eq!(ev["plan_id"], "my-plan");
        assert_eq!(ev["done"], 7);
        assert_eq!(ev["failed"], 1);
        assert_eq!(ev["missing"], 2);
    }

    fn plan_with_auth(token: Option<&str>) -> anyhow::Result<Plan> {
        let auth_attr = match token {
            Some(t) => format!(", auth_token = \"{t}\""),
            None => String::new(),
        };
        let toml = format!(
            r#"
plan_id          = "x"
benchmarks       = ["b"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf"{auth_attr} }}]
runtimes = [{{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b1", flavor = "macos-arm64" }}]
"#
        );
        Ok(Plan::parse(&toml)?)
    }

    #[rstest]
    #[case::empty(None, None, false, None, &[])]
    #[case::max_wait(
        None,
        Some(1800),
        false,
        None,
        &[("PIPETTE_READINESS_MAX_WAIT_SECS", "1800")]
    )]
    #[case::skip_thermal(
        None,
        None,
        true,
        None,
        &[("PIPETTE_READINESS_SKIP_THERMAL", "1")]
    )]
    #[case::held_touch(
        None,
        None,
        false,
        Some(pipette_readiness::HeldTouch::Fail),
        &[("PIPETTE_READINESS_HELD_TOUCH", "fail")]
    )]
    #[case::held_touch_off(
        None,
        None,
        false,
        Some(pipette_readiness::HeldTouch::Off),
        &[("PIPETTE_READINESS_HELD_TOUCH", "off")]
    )]
    #[case::auth_token(
        Some("hf_test_xxx"),
        None,
        false,
        None,
        &[(HF_TOKEN_ENV, "hf_test_xxx")]
    )]
    fn forwarded_env_cases(
        #[case] token: Option<&str>,
        #[case] max_wait_secs: Option<u64>,
        #[case] skip_thermal: bool,
        #[case] held_touch: Option<pipette_readiness::HeldTouch>,
        #[case] want: &[(&str, &str)],
    ) -> anyhow::Result<()> {
        let plan = plan_with_auth(token)?;
        let env = resolve_forwarded_env(&plan, max_wait_secs, skip_thermal, held_touch)?;
        let want: Vec<(String, String)> = want
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect();
        assert_eq!(env, want);
        Ok(())
    }

    #[rstest::rstest]
    fn pop_next_priority_order() -> anyhow::Result<()> {
        let pinned: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(0, 1)])));
        let restricted_narrow: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(1, 1)])));
        let restricted_wide: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(2, 1)])));
        let unassigned: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(3, 1)])));
        let eligible = [Arc::clone(&restricted_narrow), Arc::clone(&restricted_wide)];

        let (first, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected first pop"))?;
        assert_eq!(first, (0, 1));
        assert!(matches!(origin, Origin::Pinned));

        let (second, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected second pop"))?;
        assert_eq!(second, (1, 1));
        assert!(matches!(origin, Origin::Restricted(_)));

        let (third, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected third pop"))?;
        assert_eq!(third, (2, 1));
        assert!(matches!(origin, Origin::Restricted(_)));

        let (fourth, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected fourth pop"))?;
        assert_eq!(fourth, (3, 1));
        assert!(matches!(origin, Origin::Unassigned));

        assert!(pop_next(&pinned, &eligible, &unassigned).is_none());
        Ok(())
    }

    fn name_label_map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(n, l)| (n.to_string(), l.to_string()))
            .collect()
    }

    #[rstest::rstest]
    #[case::unrestricted_unpinned(
        &[],
        None,
        false,
        RouteOutcome::Unassigned,
    )]
    #[case::unrestricted_pinned_present(
        &[],
        Some("adb:A"),
        false,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    #[case::unrestricted_pinned_absent(
        &[],
        Some("adb:GONE"),
        false,
        RouteOutcome::Stranded(StrandedReason::PinnedAbsent("adb:GONE".to_string())),
    )]
    #[case::single_allowed_unpinned(
        &["nameA"],
        None,
        false,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    #[case::multi_allowed_unpinned(
        &["nameB", "nameA"],
        None,
        false,
        RouteOutcome::Restricted(vec!["adb:A".to_string(), "adb:B".to_string()]),
    )]
    #[case::pinned_within_allowed(
        &["nameA", "nameB"],
        Some("adb:A"),
        false,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    #[case::pinned_excluded_default(
        &["nameB"],
        Some("adb:A"),
        false,
        RouteOutcome::Stranded(StrandedReason::PinnedExcluded {
            label: "adb:A".to_string(),
            allowed: vec!["adb:B".to_string()],
        }),
    )]
    #[case::pinned_excluded_reassign(
        &["nameB"],
        Some("adb:A"),
        true,
        RouteOutcome::Pinned("adb:B".to_string()),
    )]
    #[case::no_overlap(
        &["nameC"],
        None,
        false,
        RouteOutcome::Stranded(StrandedReason::NoOverlap {
            allowed: vec!["nameC".to_string()],
        }),
    )]
    // Still stranded by default, to keep the on-device checkpoint.
    #[case::pinned_absent_with_overlap(
        &["nameA"],
        Some("adb:GONE"),
        false,
        RouteOutcome::Stranded(StrandedReason::PinnedAbsent("adb:GONE".to_string())),
    )]
    #[case::pinned_absent_reassign_routes_by_allowed(
        &["nameA"],
        Some("adb:GONE"),
        true,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    fn route_cell_cases(
        #[case] allowed: &[&str],
        #[case] pinned: Option<&str>,
        #[case] reassign: bool,
        #[case] expect: RouteOutcome,
    ) {
        let map = name_label_map(&[("nameA", "adb:A"), ("nameB", "adb:B")]);
        let present: HashSet<String> = map.values().cloned().collect();
        assert_eq!(
            route_cell(allowed, pinned, &map, &present, reassign),
            expect
        );
    }

    #[test]
    fn physical_id_capacities_takes_max_across_colocated_transports() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["b1"]

[[transports]]
client_id   = "t-llamacpp"
type        = "ssh"
host        = "shared-host"
binary_path = "/bin/pipette-llamacpp"
work_dir    = "/tmp"
shell       = "posix"
parallelism = 2

[[transports]]
client_id   = "t-mlx"
type        = "ssh"
host        = "shared-host"
binary_path = "/bin/pipette-mlx"
work_dir    = "/tmp"
shell       = "posix"

[[variants]]
clients  = ["t-llamacpp"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let caps = physical_id_capacities(&plan);
        let cap = caps
            .get("shared-host")
            .ok_or_else(|| anyhow::anyhow!("missing capacity entry for shared-host"))?;
        assert_eq!(cap.get(), 2);
        Ok(())
    }

    // The iOS transports sit between the kept ones, so this also checks each
    // kept transport stays paired with its own config.
    #[test]
    fn final_sync_skips_ios_transports() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["b1"]

[[transports]]
client_id   = "t-llamacpp"
type        = "ssh"
host        = "mac-studio"
binary_path = "/bin/pipette-llamacpp"
work_dir    = "/tmp"
shell       = "posix"

[[transports]]
client_id   = "t-iphone"
type        = "ios"
device_udid = "UDID-123"

[[transports]]
client_id   = "t-iphone-remote"
type        = "ios_over_ssh"
host        = "mac-mini"
device_udid = "UDID-456"

[[transports]]
client_id   = "t-pixel"
type        = "adb"
serial      = "PIXEL9"
binary_path = "/data/local/tmp/pipette-llamacpp"
work_dir    = "/data/local/tmp"
shell       = "posix"

[[variants]]
clients  = ["t-llamacpp"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let targets = sync_targets(&plan, None);
        assert_eq!(
            targets
                .iter()
                .map(|(transport, cfg)| (transport.target_label(), cfg.client_id().to_string()))
                .collect::<Vec<_>>(),
            vec![
                ("ssh:mac-studio".to_string(), "t-llamacpp".to_string()),
                ("adb:PIXEL9".to_string(), "t-pixel".to_string()),
            ]
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::sync(&["sync"], vec!["/bin/pipette", "--work-dir", "/tmp/w", "sync"])]
    #[case::storage_gc(
        &["storage", "gc"],
        vec!["/bin/pipette", "--work-dir", "/tmp/w", "storage", "gc"]
    )]
    fn a_client_command_names_the_work_dir_first(
        #[case] args: &[&str],
        #[case] expected: Vec<&str>,
    ) {
        assert_eq!(client_argv("/bin/pipette", "/tmp/w", args), expected);
    }
}
