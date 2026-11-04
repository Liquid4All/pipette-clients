//! The accelerator watches' GPU and NPU peaks on Windows
//! ([`crate::Watch::PeakWithGpu`] and its siblings): a thread reads PDH's
//! `\GPU Process Memory(pid_<PID>_*)\Total Committed` every 20 ms, sums the
//! instances per kind of adapter, and keeps each kind's maximum. PDH shows
//! whatever WDDM bills the process per adapter, so Vulkan, HIP, SYCL, D3D12,
//! CUDA and an NPU's driver read alike.
//!
//! PDH reads a child's figures from the parent, at any integrity level and
//! with no system state. `QueryVideoMemoryInfo` reports only the calling
//! process, and a Vulkan layer saw only Vulkan and, elevated, needed a
//! system-wide registry entry.
//!
//! `Total Committed` is the per-tick sum. Measured on Strix Halo, summing the
//! separate `Dedicated Usage` and `Shared Usage` peaks overstated it by about
//! 3%, and `Dedicated Usage` alone missed the host-coherent buffers, 10 to 12%
//! short. How the device bytes overlap the working set (so each peak is
//! reported on its own): `docs/methodology/peak-memory-windows.md`.
//!
//! An instance is named `pid_<PID>_luid_0x<HIGH>_0x<LOW>_phys_<N>`. The LUID's
//! `D3DKMT_ADAPTERTYPE` gives its kind: `ComputeOnly` marks the NPU (which sets
//! `RenderSupported` too), `SoftwareDevice` the Basic Render Driver, which is
//! left out, and an adapter that cannot be asked counts as a GPU. Until the
//! child's first allocation there is no instance (`PDH_NO_DATA`) and a tick
//! reads nothing; the wildcard is resolved on every sample, so it appears.

use std::{
    collections::HashMap,
    ffi::c_void,
    sync::mpsc::{self, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

use windows_sys::Wdk::Graphics::Direct3D::{
    D3DKMT_ADAPTERTYPE, D3DKMT_CLOSEADAPTER, D3DKMT_OPENADAPTERFROMLUID, D3DKMT_QUERYADAPTERINFO,
    D3DKMTCloseAdapter, D3DKMTOpenAdapterFromLuid, D3DKMTQueryAdapterInfo, KMTQAITYPE_ADAPTERTYPE,
};
use windows_sys::Win32::{
    Foundation::{ERROR_SUCCESS, LUID},
    System::Performance::{
        PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PdhAddEnglishCounterW, PdhCloseQuery,
        PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
    },
};

use super::PeakError;

/// `PDH_HQUERY` and `PDH_HCOUNTER`, which windows-sys types as `*mut c_void`.
type PdhQuery = *mut c_void;
type PdhCounter = *mut c_void;

/// `pdhmsg.h` codes, which windows-sys does not export.
const PDH_MORE_DATA: u32 = 0x800007D2;
const PDH_NO_DATA: u32 = 0x800007D5;
const PDH_CALC_NEGATIVE_DENOMINATOR: u32 = 0x800007D6;

/// As the macOS peak's. Inference allocations live until exit, so 20 ms gives
/// hundreds of samples at the peak.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// The most memory WDDM billed the process at once on its GPUs, and on its
/// NPUs: each the maximum of a per-sample sum, so the two may peak apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AdapterPeaks {
    pub(crate) gpu_bytes: u64,
    pub(crate) npu_bytes: u64,
}

/// A PDH poller thread: [`Self::stop_and_join`] reads its peaks, a drop
/// discards them.
#[must_use = "PdhGpuMemoryPoller::stop_and_join must be called to retrieve the peaks; \
              dropping the handle silently discards the measurement"]
pub(crate) struct PdhGpuMemoryPoller {
    stop: mpsc::Sender<()>,
    handle: JoinHandle<Result<AdapterPeaks, PeakError>>,
}

impl PdhGpuMemoryPoller {
    /// The peaks feed `max_gpu_bytes` and `max_npu_bytes`, so a PDH that could not
    /// be set up, or a panicked poller, fails the bench rather than reading as
    /// a process that held nothing.
    pub(crate) fn stop_and_join(self) -> Result<AdapterPeaks, PeakError> {
        let _ = self.stop.send(());
        self.handle
            .join()
            .map_err(|_| PeakError::GpuPollerPanicked)?
    }
}

pub(crate) fn spawn_pdh_gpu_memory_poller(pid: u32) -> PdhGpuMemoryPoller {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || {
        let mut query: PdhQuery = std::ptr::null_mut();
        // SAFETY: `query` is a live out-pointer; a null data source means real-time data.
        let open_rc = unsafe { PdhOpenQueryW(std::ptr::null(), 0, &raw mut query) };
        if open_rc != ERROR_SUCCESS {
            return Err(PeakError::Pdh {
                call: "PdhOpenQueryW",
                code: open_rc,
            });
        }
        let polled = add_counter(query, pid, "Total Committed").map(|counter| {
            let mut sampler = Samples::default();
            sampler.sample(query, counter);
            poll_until_stopped(&stop_rx, POLL_INTERVAL, || sampler.sample(query, counter));
            sampler.peaks
        });
        // SAFETY: `query` was opened above and is closed only here.
        unsafe {
            let _ = PdhCloseQuery(query);
        }
        polled
    });
    PdhGpuMemoryPoller {
        stop: stop_tx,
        handle,
    }
}

/// Calls `sample` after each wait of `interval`, the one `stop` ends too, so
/// the last sample is taken at the stop.
fn poll_until_stopped(stop: &mpsc::Receiver<()>, interval: Duration, mut sample: impl FnMut()) {
    loop {
        let stopping = match stop.recv_timeout(interval) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => true,
            Err(RecvTimeoutError::Timeout) => false,
        };
        sample();
        if stopping {
            return;
        }
    }
}

fn add_counter(query: PdhQuery, pid: u32, counter_name: &str) -> Result<PdhCounter, PeakError> {
    let path = wide_string(&format!(
        "\\GPU Process Memory(pid_{pid}_*)\\{counter_name}"
    ));
    let mut counter: PdhCounter = std::ptr::null_mut();
    // SAFETY: `path` is a NUL-terminated wide string that outlives the call, and
    // `counter` is a live out-pointer.
    let rc = unsafe { PdhAddEnglishCounterW(query, path.as_ptr(), 0, &raw mut counter) };
    if rc != ERROR_SUCCESS {
        return Err(PeakError::Pdh {
            call: "PdhAddEnglishCounterW",
            code: rc,
        });
    }
    Ok(counter)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterKind {
    Gpu,
    Npu,
    /// The Basic Render Driver: a CPU rasterizer, not a device.
    Software,
}

/// `D3DKMT_ADAPTERTYPE`'s `SoftwareDevice` bit.
const SOFTWARE_DEVICE: u32 = 1 << 2;
/// `D3DKMT_ADAPTERTYPE`'s `ComputeOnly` bit: an MCDM adapter, which is how
/// WDDM drives an NPU.
const COMPUTE_ONLY: u32 = 1 << 11;

impl AdapterKind {
    fn of(adapter_type: u32) -> Self {
        if adapter_type & SOFTWARE_DEVICE != 0 {
            Self::Software
        } else if adapter_type & COMPUTE_ONLY != 0 {
            Self::Npu
        } else {
            Self::Gpu
        }
    }
}

/// The running peaks, and each adapter's kind, asked once.
#[derive(Default)]
struct Samples {
    peaks: AdapterPeaks,
    kinds: HashMap<u64, AdapterKind>,
    /// Reused across ticks; `u64` words so the items PDH writes are aligned.
    buffer: Vec<u64>,
}

impl Samples {
    /// A failed read is skipped; the next tick tries again.
    fn sample(&mut self, query: PdhQuery, counter: PdhCounter) {
        // SAFETY: `query` is the open query the poller thread owns.
        let collect_rc = unsafe { PdhCollectQueryData(query) };
        if collect_rc != ERROR_SUCCESS {
            // PDH_NO_DATA is expected until the child's first allocation.
            if collect_rc as u32 != PDH_NO_DATA {
                tracing::trace!(
                    code = %format_args!("0x{collect_rc:08X}"),
                    "PdhCollectQueryData failed"
                );
            }
            return;
        }
        let Self {
            peaks,
            kinds,
            buffer,
        } = self;
        let now = sum_by_kind(&counter_instances(counter, buffer), |luid| {
            *kinds.entry(luid).or_insert_with(|| adapter_kind(luid))
        });
        peaks.gpu_bytes = peaks.gpu_bytes.max(now.gpu_bytes);
        peaks.npu_bytes = peaks.npu_bytes.max(now.npu_bytes);
    }
}

/// One sample's `(adapter LUID, bytes)` instances, summed per kind of adapter.
fn sum_by_kind(instances: &[(u64, u64)], mut kind: impl FnMut(u64) -> AdapterKind) -> AdapterPeaks {
    instances
        .iter()
        .fold(AdapterPeaks::default(), |mut sum, &(luid, bytes)| {
            match kind(luid) {
                AdapterKind::Gpu => sum.gpu_bytes = sum.gpu_bytes.saturating_add(bytes),
                AdapterKind::Npu => sum.npu_bytes = sum.npu_bytes.saturating_add(bytes),
                AdapterKind::Software => {}
            }
            sum
        })
}

/// The kind of the adapter `luid` names, asked through D3DKMT; a GPU when it
/// cannot be asked.
fn adapter_kind(luid: u64) -> AdapterKind {
    let kind = adapter_type(luid).map(AdapterKind::of);
    if kind.is_none() {
        tracing::warn!(
            luid = %format_args!("0x{luid:016X}"),
            "could not ask an adapter for its type; its memory counts as the GPU's"
        );
    }
    kind.unwrap_or(AdapterKind::Gpu)
}

fn adapter_type(luid: u64) -> Option<u32> {
    // Before the open, so nothing returns between it and the close.
    let size = u32::try_from(size_of::<D3DKMT_ADAPTERTYPE>()).ok()?;
    let mut open = D3DKMT_OPENADAPTERFROMLUID {
        AdapterLuid: LUID {
            LowPart: (luid & 0xFFFF_FFFF) as u32,
            HighPart: ((luid >> 32) as u32).cast_signed(),
        },
        hAdapter: 0,
    };
    // SAFETY: `open` is a live struct the call fills with a handle.
    if unsafe { D3DKMTOpenAdapterFromLuid(&raw mut open) } != 0 {
        return None;
    }
    let mut adapter_type = D3DKMT_ADAPTERTYPE::default();
    let mut query = D3DKMT_QUERYADAPTERINFO {
        hAdapter: open.hAdapter,
        Type: KMTQAITYPE_ADAPTERTYPE,
        pPrivateDriverData: (&raw mut adapter_type).cast(),
        PrivateDriverDataSize: size,
    };
    // SAFETY: `pPrivateDriverData` points at a `D3DKMT_ADAPTERTYPE` of the size
    // given, which the call fills.
    let status = unsafe { D3DKMTQueryAdapterInfo(&raw mut query) };
    let close = D3DKMT_CLOSEADAPTER {
        hAdapter: open.hAdapter,
    };
    // SAFETY: closes the handle opened above, which nothing else holds.
    unsafe { D3DKMTCloseAdapter(&raw const close) };
    // SAFETY: every bit pattern of the union is a valid `u32`.
    let value = unsafe { adapter_type.Anonymous.Value };
    (status == 0).then_some(value)
}

/// Each instance of `counter` as `(adapter LUID, bytes)`, from
/// `PdhGetFormattedCounterArrayW`; empty on any PDH error (instance not
/// registered yet, transient unavailability).
fn counter_instances(counter: PdhCounter, buffer: &mut Vec<u64>) -> Vec<(u64, u64)> {
    let mut buf_size: u32 = 0;
    let mut item_count: u32 = 0;
    // SAFETY: a null buffer with `buf_size == 0` asks only for the required size.
    let size_rc = unsafe {
        PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_LARGE,
            &raw mut buf_size,
            &raw mut item_count,
            std::ptr::null_mut(),
        )
    };
    if size_rc as u32 != PDH_MORE_DATA {
        if size_rc != ERROR_SUCCESS && size_rc as u32 != PDH_NO_DATA {
            tracing::trace!(
                code = %format_args!("0x{size_rc:08X}"),
                "PdhGetFormattedCounterArrayW(size) failed"
            );
        }
        return Vec::new();
    }
    if buf_size == 0 || item_count == 0 {
        return Vec::new();
    }
    let words = (buf_size as usize).div_ceil(size_of::<u64>());
    if buffer.len() < words {
        buffer.resize(words, 0);
    }
    // SAFETY: `buffer` holds at least `buf_size` bytes, the size PDH asked for.
    let fill_rc = unsafe {
        PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_LARGE,
            &raw mut buf_size,
            &raw mut item_count,
            buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(),
        )
    };
    if fill_rc != ERROR_SUCCESS {
        tracing::trace!(
            code = %format_args!("0x{fill_rc:08X}"),
            "PdhGetFormattedCounterArrayW(fill) failed"
        );
        return Vec::new();
    }

    let item_count = item_count as usize;
    if item_count.saturating_mul(size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>())
        > buffer.len().saturating_mul(size_of::<u64>())
    {
        tracing::warn!(
            items = item_count,
            "PdhGetFormattedCounterArrayW reported items past its buffer"
        );
        return Vec::new();
    }
    // SAFETY: PDH wrote `item_count` items at the start of `buffer` (checked to
    // fit above), and `u64` alignment covers the item's (asserted below).
    let items = unsafe {
        std::slice::from_raw_parts(
            buffer.as_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(),
            item_count,
        )
    };
    // The wildcard restricts the items to pid_<PID>_*, so every one belongs to
    // our child; its name says which adapter.
    items
        .iter()
        .filter(|item| {
            item.FmtValue.CStatus == 0 || item.FmtValue.CStatus == PDH_CALC_NEGATIVE_DENOMINATOR
        })
        .filter_map(|item| {
            let luid = adapter_luid(&wide_str(item.szName)?)?;
            // SAFETY: the largeValue arm of the union is the one valid for
            // PDH_FMT_LARGE. Driver-reported values are GPU-memory bytes; a
            // negative one is skipped rather than wrapped.
            let bytes = u64::try_from(unsafe { item.FmtValue.Anonymous.largeValue }).ok()?;
            Some((luid, bytes))
        })
        .collect()
}

/// A NUL-terminated wide string PDH wrote into its buffer.
fn wide_str(name: *const u16) -> Option<String> {
    if name.is_null() {
        return None;
    }
    let len = (0..)
        // SAFETY: PDH NUL-terminates each name inside the buffer it filled, and
        // the walk stops at the NUL.
        .take_while(|&unit| unsafe { *name.wrapping_add(unit) } != 0)
        .count();
    // SAFETY: the `len` units before the NUL were just read.
    let units = unsafe { std::slice::from_raw_parts(name, len) };
    Some(String::from_utf16_lossy(units))
}

/// The adapter LUID in an instance name, `pid_<PID>_luid_0x<HIGH>_0x<LOW>_phys_<N>`,
/// as `HIGH << 32 | LOW`.
fn adapter_luid(instance: &str) -> Option<u64> {
    let mut parts = instance
        .split('_')
        .skip_while(|part| !part.eq_ignore_ascii_case("luid"))
        .skip(1)
        .map(|part| {
            let hex = part
                .strip_prefix("0x")
                .or_else(|| part.strip_prefix("0X"))?;
            u32::from_str_radix(hex, 16).ok()
        });
    let high = parts.next()??;
    let low = parts.next()??;
    Some(u64::from(high) << 32 | u64::from(low))
}

const _: () =
    assert!(std::mem::align_of::<PDH_FMT_COUNTERVALUE_ITEM_W>() <= std::mem::align_of::<u64>());

fn wide_string(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    /// An instance names its adapter as `luid_0x<HIGH>_0x<LOW>`, the form read
    /// off a Panther Lake box's NPU and Arc iGPU.
    #[rstest::rstest]
    #[case::npu("pid_6584_luid_0x00000000_0x0000dd5c_phys_0", Some(0xDD5C))]
    #[case::high_part("pid_1_luid_0x00000001_0x0000D913_phys_0", Some((1 << 32) | 0xD913))]
    #[case::no_luid("pid_6584_phys_0", None)]
    #[case::not_hex("pid_6584_luid_0x00000000_0xZZ_phys_0", None)]
    fn an_instance_names_its_adapter(#[case] instance: &str, #[case] luid: Option<u64>) {
        assert_eq!(adapter_luid(instance), luid);
    }

    /// The Basic Render Driver's bytes are dropped, and two GPUs add up.
    #[test]
    fn a_sample_sums_each_kind_of_adapter() {
        let kind = |luid| match luid {
            1 | 4 => AdapterKind::Gpu,
            2 => AdapterKind::Npu,
            _ => AdapterKind::Software,
        };
        let sample = [(1, 400), (2, 700), (3, 9_000), (4, 50)];
        assert_eq!(
            sum_by_kind(&sample, kind),
            AdapterPeaks {
                gpu_bytes: 450,
                npu_bytes: 700,
            }
        );
    }

    /// The adapter types read on that box: the NPU sets `RenderSupported` as
    /// well as `ComputeOnly`, so only the latter tells it apart.
    #[rstest::rstest]
    #[case::arc_igpu(0x232B, AdapterKind::Gpu)]
    #[case::npu(0x2901, AdapterKind::Npu)]
    #[case::basic_render_driver(0x105, AdapterKind::Software)]
    fn an_adapter_type_names_its_kind(#[case] adapter_type: u32, #[case] kind: AdapterKind) {
        assert_eq!(AdapterKind::of(adapter_type), kind);
    }

    /// A value set just before the stop is the last one sampled, so a peak
    /// struck after the last tick still counts. The stop is sent after a
    /// sample, landing in a wait.
    #[test]
    fn the_last_sample_is_taken_at_the_stop() -> anyhow::Result<()> {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let (stop, stopped) = mpsc::channel();
        let (sampled, samples) = mpsc::channel();
        let value = Arc::new(AtomicU64::new(1));
        let poller = thread::spawn({
            let value = Arc::clone(&value);
            move || {
                poll_until_stopped(&stopped, Duration::from_millis(50), || {
                    let _ = sampled.send(value.load(Ordering::SeqCst));
                });
            }
        });
        samples.recv().context("no sample before the stop")?;
        value.store(2, Ordering::SeqCst);
        stop.send(())?;
        poller
            .join()
            .map_err(|_| anyhow::anyhow!("the poller panicked"))?;
        assert_eq!(samples.try_iter().last(), Some(2));
        Ok(())
    }

    #[test]
    fn wide_string_encodes_ascii_with_nul_terminator() {
        let w = wide_string("ABC");
        assert_eq!(w, vec![b'A' as u16, b'B' as u16, b'C' as u16, 0]);
    }

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
