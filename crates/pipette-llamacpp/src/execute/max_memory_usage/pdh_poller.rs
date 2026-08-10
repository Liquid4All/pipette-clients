//! Windows PDH `\GPU Process Memory(pid_<PID>_*)\Total Committed`
//! poller — populates `max_gpu_bytes` for every Windows GPU flavor
//! (Vulkan / HIP / SYCL / D3D12 — PDH is GPU-API-agnostic, surfacing
//! whatever the WDDM driver reports).
//!
//! Architecturally a peer of `pipette_memprobe_metal::host::
//! PhysFootprintPoller`: spawn a thread, sample at 20 ms, track the
//! running maximum, stop on signal, return the peak.
//!
//! Reads through PDH (`pdh.dll`) rather than the in-process API
//! (`IDXGIAdapter3::QueryVideoMemoryInfo` / `D3DKMTQueryVideoMemoryInfo`)
//! because the latter only return data for the *calling* process; we
//! need per-PID data for a child process from the parent.
//!
//! ## Counter choice: `Total Committed`
//!
//! PDH exposes five counters per (PID × physical-adapter): Dedicated
//! Usage, Shared Usage, Local Usage, Non Local Usage, and Total
//! Committed. Empirical measurement on Strix Halo (UMA) shows:
//!
//! - `Total Committed` ≡ `Local Usage` on this driver; both track
//!   the per-tick joint sum of memory the GPU process holds.
//! - Summing `Dedicated Usage` peak + `Shared Usage` peak
//!   independently *over-reports* by ~3% because the two counters
//!   peak at different moments — the right reading is the per-tick
//!   sum's max, which is what Total Committed already tracks
//!   internally.
//! - `Non Local Usage` is 0 on UMA (no separate non-local pool).
//!
//! We poll `Total Committed` as a single counter. On UMA it equals
//! `Dedicated + Shared` at the joint peak moment; on discrete it
//! equals VRAM + system-RAM-mapped-to-GPU at the joint peak.
//!
//! ### Why we don't pick the lower-bound counter
//!
//! `Dedicated Usage` alone systematically under-reports by the size
//! of host-coherent buffers (`Vulkan_Host`-class allocations) — on
//! Strix Halo this was a ~10-12% shortfall. The previous in-process
//! Vulkan probe captured both Vulkan_Device and Vulkan_Host
//! (everything that went through `vkAllocateMemory`), and Total
//! Committed is the closest PDH analogue.
//!
//! ### UMA double-counting note
//!
//! On AMD UMA, `Shared Usage` (the system-memory portion of Total
//! Committed) overlaps with PSAPI's `PeakWorkingSetSize` — both
//! count the same `Vulkan_Host` mirror pages once. The wire schema
//! reports each peak independently per the methodology; on AMD UMA
//! the two can overlap (a property of the platform, not a transmitted
//! flag). Summing `max_host_bytes + max_gpu_bytes` to estimate
//! device-fit is a discrete-GPU pattern; on UMA, refer to the
//! runtime's announced buffer breakdown in `extras.json` for
//! partitioned totals.
//!
//! ## Counter discovery
//!
//! The instance name PDH assigns to each process×physical-GPU pair is
//! `pid_<PID>_luid_<X>_<Y>_phys_<N>` (variants across driver
//! versions). We open the counter with the wildcard
//! `\GPU Process Memory(pid_<PID>_*)\Total Committed` and sum across
//! matching instances per sample — covers multi-adapter systems
//! where a process might hold memory on more than one GPU.
//!
//! When the child has just started, no instance exists yet
//! (`PDH_NO_DATA`). We treat that as "no data this tick" and keep
//! polling. The first allocation registers the instance within one
//! poll cycle.

use std::{
    ffi::c_void,
    sync::mpsc::{self, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

use windows_sys::Win32::{
    Foundation::ERROR_SUCCESS,
    System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
        PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE,
    },
};

/// PDH query / counter handles. windows-sys 0.61 types `PDH_HQUERY`
/// and `PDH_HCOUNTER` as `*mut c_void` (HANDLE-shaped). Wrapping for
/// readability at call sites.
type PdhQuery = *mut c_void;
type PdhCounter = *mut c_void;

/// PDH return codes we treat specially. windows-sys does not expose
/// most of these as constants — they're defined in pdhmsg.h on the
/// platform SDK side. Listed here with their MSDN numeric values.
const PDH_MORE_DATA: u32 = 0x800007D2;
const PDH_NO_DATA: u32 = 0x800007D5;
const PDH_CALC_NEGATIVE_DENOMINATOR: u32 = 0x800007D6;

/// Polling cadence matches Mac's `phys_footprint` poller. Inference
/// allocations are persistent (model + KV + compute scratch all live
/// until process exit), so the peak is steady-state — 20 ms gives
/// hundreds of samples after the workload reaches it.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Handle for a PDH GPU-memory poller thread. Mirrors the actor shape
/// of `pipette_memprobe_metal::host::PhysFootprintPoller`: drop the
/// handle to cancel without retrieving the peak (`#[must_use]` catches
/// silent drops), or call [`Self::stop_and_join`] to stop the thread
/// and read the running maximum.
#[must_use = "PdhGpuMemoryPoller::stop_and_join must be called to retrieve the peak; \
              dropping the handle silently discards the measurement"]
pub(super) struct PdhGpuMemoryPoller {
    stop: mpsc::Sender<()>,
    handle: JoinHandle<u64>,
}

impl PdhGpuMemoryPoller {
    /// Signal the poller to stop, join the thread, and return the
    /// peak `Total Committed` (in bytes) summed across all GPU
    /// adapters the child used.
    ///
    /// Returns `Err` if the poller thread panicked. The peak feeds
    /// `max_gpu_bytes` on the wire, so a panicked poller must fail
    /// the bench rather than silently report 0.
    pub(super) fn stop_and_join(self) -> anyhow::Result<u64> {
        let _ = self.stop.send(());
        self.handle
            .join()
            .map_err(|_| anyhow::anyhow!("PDH GPU-memory poller thread panicked"))
    }
}

/// Spawn a PDH polling thread for the given child PID. Returns
/// immediately; the caller should [`PdhGpuMemoryPoller::stop_and_join`]
/// after the child is reaped.
///
/// Adds one counter to the query:
/// `\GPU Process Memory(pid_<PID>_*)\Total Committed`. PDH resolves
/// the wildcard against live counter state every sample, so instances
/// becoming available later (after the child's first allocation) are
/// handled automatically; the same wildcard also covers multi-adapter
/// systems by summing every matching instance.
pub(super) fn spawn_pdh_gpu_memory_poller(pid: u32) -> PdhGpuMemoryPoller {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || -> u64 {
        // Open a PDH query handle and add the wildcard counter. If
        // any of these fail (e.g. PDH service unavailable on a
        // minimal Server Core install), log and return 0 — the
        // parent can still report a useful max_host_bytes via PSAPI,
        // and max_gpu_bytes simply ends up as 0 (not null — null
        // would require an Option<u64> through this path).
        //
        // SAFETY: all PDH calls are FFI to a stable Win32 API.
        // We CloseQuery on every exit path.
        let mut query: PdhQuery = std::ptr::null_mut();
        let open_rc = unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut query) };
        if open_rc != ERROR_SUCCESS {
            log::warn!("PdhOpenQueryW failed: code 0x{open_rc:08X}; max_gpu_bytes will be 0");
            return 0;
        }

        let counter = match add_counter(query, pid, "Total Committed") {
            Some(c) => c,
            None => {
                unsafe {
                    let _ = PdhCloseQuery(query);
                }
                return 0;
            }
        };

        let mut peak: u64 = 0;
        // Re-used between polls to avoid reallocating on every tick.
        // 64 KiB easily covers a hundred-adapter machine.
        let mut buffer: Vec<u8> = Vec::with_capacity(65_536);
        loop {
            sample_once(query, counter, &mut buffer, &mut peak);
            // Combined sleep + stop-check.
            match stop_rx.recv_timeout(POLL_INTERVAL) {
                Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => continue,
            }
        }

        unsafe {
            let _ = PdhCloseQuery(query);
        }
        peak
    });
    PdhGpuMemoryPoller {
        stop: stop_tx,
        handle,
    }
}

/// Add a `\GPU Process Memory(pid_<PID>_*)\<counter_name>` counter to
/// `query`. Returns the counter handle, or `None` after logging if
/// PDH refused. We treat add-counter failure as a non-fatal "no GPU
/// data" outcome; the caller's poller will just return 0 as the peak.
fn add_counter(query: PdhQuery, pid: u32, counter_name: &str) -> Option<PdhCounter> {
    let path = wide_string(&format!(
        "\\GPU Process Memory(pid_{pid}_*)\\{counter_name}"
    ));
    let mut counter: PdhCounter = std::ptr::null_mut();
    let rc = unsafe { PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) };
    if rc != ERROR_SUCCESS {
        log::warn!(
            "PdhAddEnglishCounterW({counter_name}) failed for pid {pid}: \
             code 0x{rc:08X}; max_gpu_bytes will be 0"
        );
        return None;
    }
    Some(counter)
}

/// One polling iteration: collect, read the counter's formatted
/// array, sum across instances, update the running peak. All errors
/// are logged at trace and swallowed — the next tick gets another
/// chance.
fn sample_once(query: PdhQuery, counter: PdhCounter, buffer: &mut Vec<u8>, peak: &mut u64) {
    let collect_rc = unsafe { PdhCollectQueryData(query) };
    if collect_rc != ERROR_SUCCESS {
        // PDH_NO_DATA is the common case before the child has
        // allocated any GPU memory and registered with WDDM.
        if collect_rc as u32 != PDH_NO_DATA {
            log::trace!("PdhCollectQueryData: code 0x{collect_rc:08X}");
        }
        return;
    }

    let total = sum_counter_instances(counter, buffer);
    if total > *peak {
        *peak = total;
    }
}

/// Sum `PDH_FMT_LARGE` values across every instance of `counter`
/// returned by `PdhGetFormattedCounterArrayW`. Returns 0 on any PDH
/// error (instance not registered yet, transient unavailability).
/// `buffer` is reused across calls to avoid reallocation on every
/// poll tick.
fn sum_counter_instances(counter: PdhCounter, buffer: &mut Vec<u8>) -> u64 {
    // Two-step pattern: first call with size=0 to learn required
    // buffer size, then call again with the buffer.
    let mut buf_size: u32 = 0;
    let mut item_count: u32 = 0;
    let size_rc = unsafe {
        PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_LARGE,
            &mut buf_size,
            &mut item_count,
            std::ptr::null_mut(),
        )
    };
    if size_rc as u32 != PDH_MORE_DATA {
        // PDH_NO_DATA — instance not ready yet; skip this tick.
        if size_rc != ERROR_SUCCESS && size_rc as u32 != PDH_NO_DATA {
            log::trace!("PdhGetFormattedCounterArrayW(size): code 0x{size_rc:08X}");
        }
        return 0;
    }
    if buf_size == 0 || item_count == 0 {
        return 0;
    }
    if buffer.len() < buf_size as usize {
        buffer.resize(buf_size as usize, 0);
    }
    let fill_rc = unsafe {
        PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_LARGE,
            &mut buf_size,
            &mut item_count,
            buffer.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W,
        )
    };
    if fill_rc != ERROR_SUCCESS {
        log::trace!("PdhGetFormattedCounterArrayW(fill): code 0x{fill_rc:08X}");
        return 0;
    }

    // Walk the array. Each item has a `largeValue` field (i64) for
    // PDH_FMT_LARGE-formatted counters. We ignore instance names —
    // the wildcard restricts to pid_<PID>_*, so every returned item
    // belongs to our child.
    let items: *const PDH_FMT_COUNTERVALUE_ITEM_W =
        buffer.as_ptr() as *const PDH_FMT_COUNTERVALUE_ITEM_W;
    let mut total: u64 = 0;
    for i in 0..item_count as isize {
        // SAFETY: PDH guarantees `item_count` valid items at `items`
        // when fill_rc == ERROR_SUCCESS.
        let item = unsafe { &*items.offset(i) };
        if item.FmtValue.CStatus != 0 && item.FmtValue.CStatus != PDH_CALC_NEGATIVE_DENOMINATOR {
            continue;
        }
        // SAFETY: the largeValue arm of the union is the one valid
        // for PDH_FMT_LARGE. Driver-reported values are non-negative
        // GPU-memory bytes, so casting i64 → u64 is sound.
        let v = unsafe { item.FmtValue.Anonymous.largeValue };
        if v > 0 {
            total = total.saturating_add(v as u64);
        }
    }
    total
}

fn wide_string(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    /// `wide_string` produces a NUL-terminated UTF-16 buffer.
    #[test]
    fn wide_string_encodes_ascii_with_nul_terminator() {
        let w = wide_string("ABC");
        assert_eq!(w, vec![b'A' as u16, b'B' as u16, b'C' as u16, 0]);
    }

    /// `wide_string` round-trips through paths with backslashes /
    /// special chars — guards against any future escaping bug in the
    /// counter-path construction.
    #[test]
    fn wide_string_preserves_backslashes_and_parens() -> anyhow::Result<()> {
        let path = "\\GPU Process Memory(pid_1234_*)\\Total Committed";
        let w = wide_string(path);
        let decoded: String = char::decode_utf16(w.iter().take(w.len() - 1).copied())
            .collect::<Result<String, _>>()
            .context("decoded buffer is not valid UTF-16")?;
        assert_eq!(decoded, path);
        assert_eq!(
            *w.last().context("buffer should not be empty")?,
            0,
            "must be NUL-terminated"
        );
        Ok(())
    }
}
