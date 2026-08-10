use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    num::NonZeroUsize,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Context;

use pipette_plan_types::{Plan, RunnableCell, TransportConfig, HF_TOKEN_ENV};

use super::{
    host_semaphore::HostSemaphore,
    probe::{probe_device, wait_for_device},
    state_io::{append_local_state, ensure_plans_dir, load_state_index},
};
use crate::{
    runner::Shard,
    shell::RemoteExecRequest,
    state::{AttemptStatus, CellState, StateEvent, StateSummary},
    transport::Transport,
};

/// Options controlling a `run_matrix` invocation.
#[derive(Debug, Default, Clone, Copy)]
pub struct RunOptions {
    /// Cap the number of cells to run this invocation.
    pub limit: Option<usize>,
    /// Re-queue cells that most recently failed.
    pub include_failed: bool,
    /// Emit structured JSON events to stdout (one per line) instead
    /// of human-readable summary lines. stderr logging is unchanged.
    pub json: bool,
    /// Run cells whose pinned transport is no longer in the plan's
    /// `[[transports]]` list. Off by default — preserves the
    /// on-device sample checkpoint by waiting for the original
    /// device to come back. On loses that checkpoint and starts the
    /// cell from sample zero on whichever worker grabs it.
    pub reassign_stranded: bool,
    /// Ignore the state-recorded pin entirely and route every runnable
    /// cell by `allowed_clients` only. Lets stragglers that all last
    /// ran on one device fan back out across every allowed worker in
    /// parallel, at the cost of the on-device sample checkpoint (cells
    /// restart from sample zero on whichever worker grabs them).
    pub ignore_pins: bool,
    /// Override the adb server port (`adb -P <port>`) for every adb
    /// transport. `None` falls back to each transport's own `port`
    /// value (or the adb default).
    pub adb_port: Option<u16>,
    /// Forward `PIPETTE_READINESS_MAX_WAIT_SECS` to every cell so remote
    /// runners (ssh/adb/slurm) honor the readiness deadline override.
    /// `None` forwards nothing — local runners inherit the driver env
    /// regardless, so this only matters for off-box transports.
    pub readiness_max_wait_secs: Option<u64>,
    /// Forward `PIPETTE_READINESS_SKIP_THERMAL` to every cell so remote
    /// runners waive the thermal criterion (keeping the load criterion).
    /// `false` forwards nothing.
    pub readiness_skip_thermal: bool,
    /// Run only the cells in this shard of the ordered matrix. `None`
    /// runs every runnable cell. Sharding is by position in the full
    /// ordered matrix (before the state filter), so the partition is
    /// stable as cells complete — letting work be split across
    /// independent invocations (e.g. one per SLURM job).
    pub shard: Option<Shard>,
}

/// Resolve the env vars the runner forwards to every remote
/// invocation. Today: `PIPETTE_HF_TOKEN`, set from the access token the plan
/// carries on its gated/private models; the remote CLI injects it back into the
/// model definition. Empty when no model carries a token — the token travels in
/// the plan, not the driver env, so a missing driver env var is no longer an error.
fn resolve_forwarded_env(
    plan: &Plan,
    readiness_max_wait_secs: Option<u64>,
    readiness_skip_thermal: bool,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut env = match plan.auth_token() {
        Some(token) => vec![(HF_TOKEN_ENV.to_string(), token.as_ref().to_string())],
        None => Vec::new(),
    };
    // Forwarded so a remote runner (ssh/adb/slurm) honors the readiness
    // deadline override; the runner's own CLI validates it and its
    // `wait_until_ready()` reads it. Local runners would inherit it anyway.
    if let Some(secs) = readiness_max_wait_secs {
        env.push((
            pipette_readiness::MAX_WAIT_ENV.to_string(),
            secs.to_string(),
        ));
    }
    // Only forwarded when set: absent means "enforce", which is also what the
    // runner defaults to, so an unset flag leaves the remote env untouched.
    if readiness_skip_thermal {
        env.push((
            pipette_readiness::SKIP_THERMAL_ENV.to_string(),
            "1".to_string(),
        ));
    }
    Ok(env)
}

/// Where a runnable cell should go in this run.
#[derive(Debug, PartialEq, Eq)]
enum RouteOutcome {
    /// Pinned to a specific transport (by label). Used for both
    /// state-resume pinning and the degenerate "allowed_clients has
    /// length 1" case.
    Pinned(String),
    /// Cell allows 2+ transports — drained by any of those workers.
    /// The Vec is sorted so two cells with the same allowed set share
    /// one queue.
    Restricted(Vec<String>),
    /// No state pinning and no allowed_clients restriction — any
    /// worker can pick it up.
    Unassigned,
    /// Cell can't run in this invocation; the reason is reported on
    /// stderr (and surfaces under the existing `stranded` total).
    Stranded(StrandedReason),
}

#[derive(Debug, PartialEq, Eq)]
enum StrandedReason {
    /// State-pinned to a transport that's not in this run.
    PinnedAbsent(String),
    /// State-pinned to a present transport, but the cell's
    /// `allowed_clients` excludes it.
    PinnedExcluded { label: String, allowed: Vec<String> },
    /// Cell's `allowed_clients` references only transports not in
    /// this run.
    NoOverlap { allowed: Vec<String> },
}

impl StrandedReason {
    /// Bucket key for aggregated counts in the stranded summary.
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

/// Route a single cell given the run's transport set, the cell's
/// state-pinning history, and the `--reassign-stranded` flag.
///
/// Pure function so it can be exhaustively unit-tested away from the
/// rest of the worker plumbing.
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

    // Cell declared allowed_clients but none are in this run.
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

/// One `Transport` per `[[transports]]` entry — used for the final
/// `sync` pass, which should run once per target even if the worker
/// pool for that target is larger than one.
fn build_transports(plan: &Plan, adb_port: Option<u16>) -> Vec<Transport> {
    plan.transports
        .iter()
        .map(|cfg| Transport::from_config_with_adb_port(cfg, adb_port))
        .collect()
}

/// Expand each `[[transports]]` entry to `parallelism` workers, each
/// paired with the source `TransportConfig` so the worker can build
/// per-cell argv using its own `binary_path`/`work_dir`. Workers race
/// through their pinned queue (and the shared unassigned pool);
/// multiple workers for the same transport share one pinned queue
/// keyed by that target's label.
fn build_workers(plan: &Plan, adb_port: Option<u16>) -> Vec<(Transport, &TransportConfig)> {
    plan.transports
        .iter()
        .flat_map(|cfg| {
            (0..cfg.parallelism().get())
                .map(move |_| (Transport::from_config_with_adb_port(cfg, adb_port), cfg))
        })
        .collect()
}

/// Per-`physical_id` concurrency budget derived from the plan: the
/// max `parallelism` of any transport reaching that box. The runner
/// uses this to size the [`HostSemaphore`] for each box; co-located
/// transports with `parallelism = 1` still serialize per-box, while
/// a transport that opts into `parallelism = N` actually gets N
/// concurrent slots on its `physical_id`.
fn physical_id_capacities(plan: &Plan) -> HashMap<String, NonZeroUsize> {
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

/// One queue's worth of work — used both for per-transport pinned
/// queues and the shared unassigned pool.
type CellQueue = Arc<Mutex<VecDeque<RunnableItem>>>;

/// `(cell_idx, attempt-number-for-this-run)`. The cell is looked up
/// in the worker's own `cells_owned`; argv is looked up in the
/// worker's own `argvs` (built from the worker's `TransportConfig`).
type RunnableItem = (usize, usize);

// ---------------------------------------------------------------------------
// JSON event constructors
// ---------------------------------------------------------------------------

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

fn end_event(plan_id: &str, summary: &StateSummary) -> serde_json::Value {
    serde_json::json!({
        "event": "end",
        "plan_id": plan_id,
        "done": summary.done,
        "failed": summary.failed,
        "missing": summary.missing,
    })
}

/// Run the matrix plan: spawn one worker per transport, execute
/// cells, record state.
/// Order cells for execution: **model axis first**, then benchmark,
/// runtime, runtime flags, and (as a final tiebreaker) the allowed-client
/// set — a total, deterministic order. Grouping every cell of a model
/// together means a worker finishes one model's benchmark sweep before
/// starting the next, and the order is reproducible run-to-run.
/// `runnable_cells` returns a `HashSet` whose iteration order is otherwise
/// arbitrary, which made successive runs interleave models unpredictably.
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

pub fn run_matrix(plans_dir: &Path, plan: &Plan, opts: RunOptions) -> anyhow::Result<()> {
    let RunOptions {
        limit,
        include_failed,
        json,
        reassign_stranded,
        ignore_pins,
        adb_port,
        readiness_max_wait_secs,
        readiness_skip_thermal,
        shard,
    } = opts;
    // Validate before any setup work: a missing HF_TOKEN should fail
    // immediately, not after state files have been read or worker
    // handles built.
    let forwarded_env =
        resolve_forwarded_env(plan, readiness_max_wait_secs, readiness_skip_thermal)?;
    let cells: Vec<RunnableCell> = order_cells(plan.runnable_cells()?.into_iter().collect());
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    let workers = build_workers(plan, adb_port);
    if workers.is_empty() {
        anyhow::bail!("plan has no transports configured");
    }

    // Routing is keyed on `transport.name` (unique by TOML schema) so
    // two transports that reach the same physical box under different
    // binaries get separate queues — otherwise a cell tagged for
    // `macstudio-mlx-1` could land on the `macstudio-1` worker (running
    // `pipette-llamacpp`) and fail with the wrong-binary symptom.
    // `target_label()` stays as the human-readable display string.
    let name_to_label: HashMap<String, String> = plan
        .transports
        .iter()
        .map(|cfg| (cfg.client_id().to_string(), cfg.client_id().to_string()))
        .collect();
    let present_labels: HashSet<String> = name_to_label.values().cloned().collect();

    let runnable: Vec<_> = cells
        .iter()
        .enumerate()
        .filter_map(|(idx, cell)| {
            // `idx` is the cell's position in the full ordered matrix, so
            // the shard partition is stable regardless of which cells have
            // already completed.
            if !shard.is_none_or(|s| s.contains(idx)) {
                return None;
            }
            let attempts = state.attempts_for(cell);
            match state.state_for(cell) {
                CellState::Done => None,
                CellState::Failed if !include_failed => None,
                CellState::Failed => Some((idx, 1)),
                CellState::Missing => Some((idx, attempts + 1)),
            }
        })
        .collect();

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
        let cell = &cells[idx];
        let allowed: Vec<&str> = cell.allowed_clients.iter().map(|c| c.as_ref()).collect();
        // `--ignore-pins` drops the recorded pin so stragglers route by
        // `allowed_clients` alone and fan back out across every worker.
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
        if let Some(limit) = limit {
            if runnable_count < limit && stranded_total > 0 {
                eprintln!(
                    "[stranded] --limit {limit} included {stranded_total} stranded \
                     cell(s); only {runnable_count} will actually run this invocation"
                );
            }
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
        // One counting semaphore per `physical_id`. Co-located
        // transports (e.g. llamacpp + mlx on the same Mac Studio)
        // share a contention budget so the box never runs more
        // concurrent benchmarks than the user explicitly opted into.
        let host_locks: HashMap<String, Arc<HostSemaphore>> = physical_id_capacities(plan)
            .into_iter()
            .map(|(key, cap)| (key, Arc::new(HostSemaphore::new(cap))))
            .collect();
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
        )?;
        run_final_sync(plan, &forwarded_env, adb_port);
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

/// Which queue an item came from — used so a worker that has to
/// re-queue (e.g. probe failed before claiming) can push back to the
/// same queue rather than implicitly stealing from the shared pool.
#[derive(Debug, Clone)]
enum Origin {
    Pinned,
    /// The specific restricted queue this item was popped from.
    /// Cloned `Arc` so a probe-failed worker can return it to the
    /// same queue and keep eligibility correct.
    Restricted(CellQueue),
    Unassigned,
}

/// Pop next item for this worker: pinned → eligible restricted queues
/// (narrowest first to avoid starving cells with a single allowed
/// transport) → shared unassigned pool.
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

#[allow(clippy::too_many_arguments)]
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
) -> anyhow::Result<()> {
    let state_lock = Arc::new(Mutex::new(()));
    let counter = Arc::new(Mutex::new(0usize));
    let by_transport = Arc::new(by_transport);
    let forwarded_env = Arc::new(forwarded_env.to_vec());

    let handles: Vec<_> = workers
        .into_iter()
        .map(|(transport, transport_cfg)| {
            let argvs: Vec<Vec<String>> = cells
                .iter()
                .map(|cell| {
                    let mut argv = cell.build_argv(transport_cfg)?;
                    if transport_cfg.appends_sync_flag() {
                        argv.push("--sync".to_string());
                    }
                    Ok(argv)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let cells_owned: Vec<RunnableCell> = cells.to_vec();
            let state_lock = Arc::clone(&state_lock);
            let counter = Arc::clone(&counter);
            let plan_id = plan.plan_id.clone();
            let binary_path = transport.binary_path().to_string();
            let plans_dir = plans_dir.to_path_buf();
            let max_consecutive = plan.retry.max_consecutive_failures;
            // `route_key` (transport.name) identifies the worker's queue
            // and is persisted as the cell's `transport_label` in state —
            // route_cell uses the same key so pinning round-trips.
            // `label` is the human-readable target string used in stderr
            // and JSON `transport` fields.
            let route_key = transport_cfg.client_id().to_string();
            let label = transport.target_label();
            let pinned_queue = by_transport
                .get(&route_key)
                .cloned()
                .unwrap_or_else(|| Arc::new(Mutex::new(VecDeque::new())));
            // Restricted queues this worker is allowed to drain,
            // narrowest-first so cells with a single permitted
            // transport don't starve behind broader-allowed cells.
            let eligible_restricted: Vec<CellQueue> = {
                let mut by_size: Vec<(&Vec<String>, &CellQueue)> = restricted
                    .iter()
                    .filter(|(keys, _)| keys.contains(&route_key))
                    .collect();
                // Narrowest first; tie-break on the keys themselves
                // for a deterministic drain order across runs.
                by_size.sort_by(|(a, _), (b, _)| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
                by_size.into_iter().map(|(_, q)| Arc::clone(q)).collect()
            };
            // Host-lock slot: shared with every other worker whose
            // transport reaches the same physical box. Acquired
            // around `transport.exec` so co-located workers don't
            // run more concurrent benchmarks than the plan's
            // host-slot budget for that box allows.
            let host_lock = host_locks
                .get(&transport_cfg.physical_id())
                .cloned()
                .context("host_lock present for every transport's physical_id")?;
            let unassigned = Arc::clone(&unassigned);
            let forwarded_env = Arc::clone(&forwarded_env);

            Ok(std::thread::spawn(move || {
                let prefix = Some(label.as_str());
                let mut consecutive_failures = 0usize;

                while let Some(((idx, attempt), origin)) =
                    pop_next(&pinned_queue, &eligible_restricted, &unassigned)
                {
                    let cell = &cells_owned[idx];
                    let argv = &argvs[idx];

                    if !probe_device(&transport, &label) {
                        let return_queue: &CellQueue = match &origin {
                            Origin::Pinned => &pinned_queue,
                            Origin::Restricted(q) => q,
                            Origin::Unassigned => &unassigned,
                        };
                        push_back(return_queue, (idx, attempt));
                        if wait_for_device(&transport, &label) {
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
                            if wait_for_device(&transport, &label) {
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

/// Append a state event under the shared `state_lock`. Returns
/// `false` if writing failed — the worker should break out of its
/// loop on `false`. `emit_json_cell` controls whether a Success/Failed
/// also produces a `--json` cell line on stdout (Started events
/// stay out of the JSON event stream to avoid breaking existing
/// consumers).
///
/// `route_key` is the transport name (unique per plan) and is what
/// gets persisted as the cell's `transport_label` in state, so
/// `pinned_transport_for` round-trips against the route map.
/// `display_label` is the human-readable target string used for
/// stderr error prefixes and the JSON `cell` event's `transport`
/// field.
#[allow(clippy::too_many_arguments)]
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

fn run_final_sync(plan: &Plan, forwarded_env: &[(String, String)], adb_port: Option<u16>) {
    for transport in build_transports(plan, adb_port) {
        // iOS has no sync binary — each cell already uploaded inline
        // (`submit=1`), so there is nothing to flush at the end. Both iOS transports:
        // the argv below is a *binary* invocation, and an iOS transport turns argv into
        // app arguments, so this would launch the app with `xcrun --work-dir '' sync` —
        // no `headlessrun` marker, so it opens the UI and never terminates, and the
        // runner waits on it forever.
        if matches!(
            transport,
            Transport::Ios { .. } | Transport::IosOverSsh { .. }
        ) {
            continue;
        }
        let label = transport.target_label();
        let binary_path = transport.binary_path().to_string();
        let work_dir = transport.work_dir().to_string();
        let sync_request = RemoteExecRequest {
            argv: vec![
                binary_path.clone(),
                "--work-dir".to_string(),
                work_dir,
                "sync".to_string(),
            ],
            env: forwarded_env.to_vec(),
            cwd: Some(parent_dir_of_binary(&binary_path)),
            job_name: Some("sync".to_string()),
        };
        eprintln!("[{label}] final sync...");
        match transport.exec(sync_request, Some(label.as_str())) {
            Ok(o) if o.status != 0 => {
                eprintln!("[{label}] warning: final sync exit code {}", o.status);
            }
            Err(e) => {
                eprintln!("[{label}] warning: final sync failed: {e:#}");
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        BenchmarkId, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, LlamaCppFlavor,
        LlamacppCliStockTools, LlamacppCliStockToolsSource, Model, NonEmptyString, RepositoryUrl,
        Runtime, SourceRepository,
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
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("rt".to_string()).context("version")?,
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

    #[test]
    fn order_cells_groups_by_model_and_is_deterministic() -> anyhow::Result<()> {
        // Two models × two benchmarks, fed in a jumbled order.
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

        // Model axis first: total ascending order by (model, benchmark).
        let mut want = keys.clone();
        want.sort();
        assert_eq!(keys, want, "cells must be in model-first sorted order");

        // Each model's benchmarks are contiguous (no interleaving across models).
        assert_eq!(keys[0].0, keys[1].0);
        assert_eq!(keys[2].0, keys[3].0);
        assert_ne!(keys[1].0, keys[2].0);

        // Deterministic regardless of the input order.
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
runtimes = [{{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }}]
"#
        );
        Plan::parse(&toml)
    }

    #[test]
    fn forwarded_env_empty_when_no_auth_required() -> anyhow::Result<()> {
        let plan = plan_with_auth(None)?;
        let env = resolve_forwarded_env(&plan, None, false)?;
        assert!(env.is_empty());
        Ok(())
    }

    #[test]
    fn forwarded_env_includes_readiness_override_when_set() -> anyhow::Result<()> {
        let plan = plan_with_auth(None)?;
        let env = resolve_forwarded_env(&plan, Some(1800), false)?;
        assert_eq!(
            env,
            vec![(
                "PIPETTE_READINESS_MAX_WAIT_SECS".to_string(),
                "1800".to_string()
            )]
        );
        Ok(())
    }

    /// The thermal skip forwards only when asked for, so an unset flag leaves
    /// a remote runner's environment untouched and enforcing.
    #[test]
    fn forwarded_env_includes_thermal_skip_only_when_set() -> anyhow::Result<()> {
        let plan = plan_with_auth(None)?;
        assert!(resolve_forwarded_env(&plan, None, false)?.is_empty());
        assert_eq!(
            resolve_forwarded_env(&plan, None, true)?,
            vec![(
                "PIPETTE_READINESS_SKIP_THERMAL".to_string(),
                "1".to_string()
            )]
        );
        Ok(())
    }

    #[test]
    fn forwarded_env_forwards_plan_token_as_pipette_hf_token() -> anyhow::Result<()> {
        let plan = plan_with_auth(Some("hf_test_xxx"))?;
        let env = resolve_forwarded_env(&plan, None, false)?;
        assert_eq!(
            env,
            vec![(HF_TOKEN_ENV.to_string(), "hf_test_xxx".to_string())]
        );
        Ok(())
    }

    #[rstest::rstest]
    fn pop_next_priority_order() -> anyhow::Result<()> {
        let pinned: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(0, 1)])));
        let restricted_narrow: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(1, 1)])));
        let restricted_wide: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(2, 1)])));
        let unassigned: CellQueue = Arc::new(Mutex::new(VecDeque::from(vec![(3, 1)])));
        let eligible = [Arc::clone(&restricted_narrow), Arc::clone(&restricted_wide)];

        // 1. pinned wins.
        let (first, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected first pop"))?;
        assert_eq!(first, (0, 1));
        assert!(matches!(origin, Origin::Pinned));

        // 2. narrowest restricted next (caller pre-sorted the slice).
        let (second, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected second pop"))?;
        assert_eq!(second, (1, 1));
        assert!(matches!(origin, Origin::Restricted(_)));

        // 3. wider restricted next.
        let (third, origin) = pop_next(&pinned, &eligible, &unassigned)
            .ok_or_else(|| anyhow::anyhow!("expected third pop"))?;
        assert_eq!(third, (2, 1));
        assert!(matches!(origin, Origin::Restricted(_)));

        // 4. unassigned last.
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
    // Unrestricted + unpinned → unassigned.
    #[case::unrestricted_unpinned(
        &[],
        None,
        false,
        RouteOutcome::Unassigned,
    )]
    // Unrestricted + pinned to present → pinned to that label.
    #[case::unrestricted_pinned_present(
        &[],
        Some("adb:A"),
        false,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    // Unrestricted + pinned to absent → stranded (PinnedAbsent).
    #[case::unrestricted_pinned_absent(
        &[],
        Some("adb:GONE"),
        false,
        RouteOutcome::Stranded(StrandedReason::PinnedAbsent("adb:GONE".to_string())),
    )]
    // Single-allowed + unpinned → routed to that one transport's label.
    #[case::single_allowed_unpinned(
        &["nameA"],
        None,
        false,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    // Multi-allowed + unpinned → restricted queue keyed by sorted labels.
    #[case::multi_allowed_unpinned(
        &["nameB", "nameA"],
        None,
        false,
        RouteOutcome::Restricted(vec!["adb:A".to_string(), "adb:B".to_string()]),
    )]
    // Allowed-set overlaps run, pinned to a label in the overlap → keep pinning.
    #[case::pinned_within_allowed(
        &["nameA", "nameB"],
        Some("adb:A"),
        false,
        RouteOutcome::Pinned("adb:A".to_string()),
    )]
    // Pinned to label, allowed excludes it → stranded (PinnedExcluded), default.
    #[case::pinned_excluded_default(
        &["nameB"],
        Some("adb:A"),
        false,
        RouteOutcome::Stranded(StrandedReason::PinnedExcluded {
            label: "adb:A".to_string(),
            allowed: vec!["adb:B".to_string()],
        }),
    )]
    // Same case + --reassign-stranded → drops pinning, routes by allowed.
    #[case::pinned_excluded_reassign(
        &["nameB"],
        Some("adb:A"),
        true,
        RouteOutcome::Pinned("adb:B".to_string()),
    )]
    // Allowed-set has no overlap with the run → stranded (NoOverlap).
    #[case::no_overlap(
        &["nameC"],
        None,
        false,
        RouteOutcome::Stranded(StrandedReason::NoOverlap {
            allowed: vec!["nameC".to_string()],
        }),
    )]
    // Pinned to absent transport WITH allowed_clients that overlaps → still
    // PinnedAbsent by default (preserves on-device checkpoint preference).
    #[case::pinned_absent_with_overlap(
        &["nameA"],
        Some("adb:GONE"),
        false,
        RouteOutcome::Stranded(StrandedReason::PinnedAbsent("adb:GONE".to_string())),
    )]
    // Same situation + --reassign-stranded → drop pinning, route by allowed.
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
        // Two SSH transports reach the same host (so the same
        // physical_id) but ask for different parallelism. The
        // box's effective capacity should be the max.
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
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let caps = physical_id_capacities(&plan);
        let cap = caps
            .get("shared-host")
            .ok_or_else(|| anyhow::anyhow!("missing capacity entry for shared-host"))?;
        assert_eq!(cap.get(), 2);
        Ok(())
    }
}
