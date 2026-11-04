//! An NVIDIA GPU's memory per process, through NVML, the library behind
//! `nvidia-smi`, loaded at run time so the build needs no CUDA: the bytes the
//! driver attributes to a pid (`usedGpuMemory`), its CUDA context and every
//! allocation it holds, summed over every GPU.
//!
//! NVML keeps no per-process watermark (accounting mode would, but needs root),
//! so the sampler reads it on the host's tick and keeps the maximum. A read
//! costs microseconds; `nvmlInit` takes over a second on a host whose driver is
//! not kept loaded (persistence mode off), so it runs once, at the start.
//!
//! The pid is the one the driver sees, so pipette must share the run's pid
//! namespace. A CUDA or OpenCL process is listed among a GPU's compute
//! processes, a Vulkan or OpenGL one among its graphics processes, and one on
//! both in each list with the same bytes. A GPU that NVML cannot open or read
//! at the start is skipped, as is a single list it refuses then, so one card
//! pipette may not touch does not hide the others; a list the GPU does not
//! support counts as empty. A list that answered at the start and refuses a
//! later read leaves that read unanswered, rather than a sum without it.

use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use super::{NvmlError, NvmlReadError};

type NvmlReturn = c_int;
type NvmlDevice = *mut c_void;

const NVML_SUCCESS: NvmlReturn = 0;
const NVML_ERROR_NOT_SUPPORTED: NvmlReturn = 3;
const NVML_ERROR_INSUFFICIENT_SIZE: NvmlReturn = 7;
/// `usedGpuMemory` when the driver cannot say: MIG, vGPU, no permission.
const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

/// `nvmlProcessInfo_t`, as `nvmlDeviceGet{Compute,Graphics}RunningProcesses_v3`
/// fill it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

const _: () = assert!(std::mem::size_of::<ProcessInfo>() == 24);

type FnInit = unsafe extern "C" fn() -> NvmlReturn;
type FnShutdown = unsafe extern "C" fn() -> NvmlReturn;
type FnErrorString = unsafe extern "C" fn(NvmlReturn) -> *const c_char;
type FnDeviceGetCount = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
type FnDeviceGetHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
type FnDeviceGetProcesses =
    unsafe extern "C" fn(NvmlDevice, *mut c_uint, *mut ProcessInfo) -> NvmlReturn;

/// A pid's GPU memory that NVML cannot report.
#[derive(Debug, PartialEq, Eq)]
pub struct NotAvailable;

/// What the driver attributes to each process it lists, by pid.
pub type Pids = BTreeMap<u32, Result<u64, NotAvailable>>;

/// `dlopen`ed `libnvidia-ml`, closed on drop.
struct Library(*mut c_void);

impl Library {
    fn open() -> Result<Self, NvmlError> {
        // SAFETY: a literal, NUL-terminated soname.
        let handle = unsafe {
            libc::dlopen(
                c"libnvidia-ml.so.1".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if handle.is_null() {
            return Err(NvmlError::Load(dlerror()));
        }
        Ok(Self(handle))
    }

    fn symbol(&self, name: &CStr) -> Result<*mut c_void, NvmlError> {
        // SAFETY: `self.0` is a live handle and `name` NUL-terminated; the result
        // is only an address until the caller transmutes it.
        let addr = unsafe { libc::dlsym(self.0, name.as_ptr()) };
        if addr.is_null() {
            return Err(NvmlError::Symbol(name.to_string_lossy().into_owned()));
        }
        Ok(addr)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `dlopen` and nothing resolved from it
        // outlives the `Nvml` that owns it.
        unsafe { libc::dlclose(self.0) };
    }
}

fn dlerror() -> String {
    // SAFETY: `dlerror` returns null or a NUL-terminated message owned by libc.
    let message = unsafe { libc::dlerror() };
    if message.is_null() {
        return "unknown error".to_owned();
    }
    // SAFETY: checked non-null above; copied before any other dl call.
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

macro_rules! resolve {
    ($library:expr, $name:literal, $ty:ty) => {{
        let addr = $library.symbol($name)?;
        // SAFETY: `$ty` is NVML's documented signature for `$name`, and `addr`
        // was checked non-null by `symbol`.
        unsafe { std::mem::transmute::<*mut c_void, $ty>(addr) }
    }};
}

/// One of a GPU's process lists, by the NVML call that fills it.
#[derive(Clone, Copy)]
struct List {
    call: &'static str,
    fill: FnDeviceGetProcesses,
}

/// A GPU NVML opened, with the lists it answered at the start.
struct Gpu {
    device: NvmlDevice,
    lists: Vec<List>,
}

/// A list NVML refused, and its code.
#[derive(Debug, PartialEq, Eq)]
struct Refusal {
    call: &'static str,
    code: NvmlReturn,
}

/// The processes of the GPUs NVML opened, read a GPU and a list at a time.
struct Processes {
    gpus: Vec<Gpu>,
    /// Reused across reads; grown when a GPU lists more processes.
    rows: Vec<ProcessInfo>,
}

impl Processes {
    /// `device` with the `lists` it answers now, or the last refusal when it
    /// answers neither. A list refused at the start, as a card in MIG mode
    /// refuses an unprivileged caller both, would be refused on every read, so
    /// it is left out rather than leaving every read unanswered.
    fn probe(&mut self, device: NvmlDevice, lists: [List; 2]) -> Result<Gpu, Refusal> {
        let [first, second] = lists.map(|list| fill(&mut self.rows, list, device).map(|_| list));
        let lists = match (first, second) {
            (Ok(one), Ok(other)) => vec![one, other],
            (Ok(one), Err(_)) | (Err(_), Ok(one)) => vec![one],
            (Err(_), Err(refusal)) => return Err(refusal),
        };
        Ok(Gpu { device, lists })
    }

    /// [`Nvml::processes`]'s figures; `None` when a GPU refuses a list.
    fn by_pid(&mut self) -> Option<Pids> {
        let Self { gpus, rows } = self;
        gpus.iter().try_fold(Pids::new(), |total, gpu| {
            let on_gpu = gpu.lists.iter().try_fold(Pids::new(), |kept, &list| {
                let filled = fill(rows, list, gpu.device).ok()?;
                Some(merge(kept, rows_by_pid(rows.get(..filled)?), u64::max))
            })?;
            Some(merge(total, on_gpu, u64::saturating_add))
        })
    }
}

/// How many of `rows` `list` filled with `device`'s processes; none for a
/// list the GPU does not support.
fn fill(rows: &mut Vec<ProcessInfo>, list: List, device: NvmlDevice) -> Result<usize, Refusal> {
    let refused = |code| Refusal {
        call: list.call,
        code,
    };
    // A process that starts between NVML's count and its fill makes it ask for
    // more room again.
    let mut grown = 0;
    loop {
        let mut count =
            c_uint::try_from(rows.len()).map_err(|_| refused(NVML_ERROR_INSUFFICIENT_SIZE))?;
        // SAFETY: `rows` holds `count` entries, and NVML writes at most that
        // many and sets `count` to what it wrote, or to what it needs.
        let code = unsafe { (list.fill)(device, &raw mut count, rows.as_mut_ptr()) };
        let count = usize::try_from(count).map_err(|_| refused(NVML_ERROR_INSUFFICIENT_SIZE))?;
        match code {
            NVML_SUCCESS => return Ok(count.min(rows.len())),
            NVML_ERROR_NOT_SUPPORTED => return Ok(0),
            NVML_ERROR_INSUFFICIENT_SIZE if grown < 2 => {
                grown += 1;
                rows.resize(count.saturating_add(16), ProcessInfo::default());
            }
            code => return Err(refused(code)),
        }
    }
}

/// An initialized NVML; shut down on drop.
pub(crate) struct Nvml {
    processes: Processes,
    shutdown: FnShutdown,
    /// Closed after `Drop for Nvml` has shut NVML down.
    _library: Library,
}

// SAFETY: NVML's functions are thread-safe, and its device handles stay valid
// until `nvmlShutdown`, which only `drop` calls.
unsafe impl Send for Nvml {}

impl Nvml {
    /// Takes over a second on a host whose driver is not kept loaded: open it
    /// once, before the run.
    pub(crate) fn open() -> Result<Self, NvmlError> {
        let library = Library::open()?;
        let init = resolve!(library, c"nvmlInit_v2", FnInit);
        let error_string = resolve!(library, c"nvmlErrorString", FnErrorString);
        let count_devices = resolve!(library, c"nvmlDeviceGetCount_v2", FnDeviceGetCount);
        let device_at = resolve!(
            library,
            c"nvmlDeviceGetHandleByIndex_v2",
            FnDeviceGetHandleByIndex
        );
        let failed = |call: &'static str, code: NvmlReturn| {
            // SAFETY: `nvmlErrorString` takes any return code.
            let message = unsafe { error_string(code) };
            // SAFETY: what it returns is a static, NUL-terminated string.
            let message = unsafe { CStr::from_ptr(message) };
            NvmlError::Call {
                call,
                message: message.to_string_lossy().into_owned(),
            }
        };
        let shutdown = resolve!(library, c"nvmlShutdown", FnShutdown);
        let lists = [
            List {
                call: "nvmlDeviceGetComputeRunningProcesses_v3",
                fill: resolve!(
                    library,
                    c"nvmlDeviceGetComputeRunningProcesses_v3",
                    FnDeviceGetProcesses
                ),
            },
            List {
                call: "nvmlDeviceGetGraphicsRunningProcesses_v3",
                fill: resolve!(
                    library,
                    c"nvmlDeviceGetGraphicsRunningProcesses_v3",
                    FnDeviceGetProcesses
                ),
            },
        ];
        // SAFETY: `init` is `nvmlInit_v2`, resolved with its documented
        // signature, which takes nothing.
        match unsafe { init() } {
            NVML_SUCCESS => {}
            code => return Err(failed("nvmlInit_v2", code)),
        }
        // From here its drop shuts NVML down.
        let mut nvml = Self {
            processes: Processes {
                gpus: Vec::new(),
                rows: vec![ProcessInfo::default(); 64],
            },
            shutdown,
            _library: library,
        };
        let mut count: c_uint = 0;
        // SAFETY: `count` is a live out-pointer.
        match unsafe { count_devices(&raw mut count) } {
            NVML_SUCCESS => {}
            code => return Err(failed("nvmlDeviceGetCount_v2", code)),
        }
        let (opened, refused): (Vec<_>, Vec<_>) = (0..count)
            .map(|index| {
                let mut device: NvmlDevice = std::ptr::null_mut();
                // SAFETY: `index` is below the device count and `device` is a
                // live out-pointer.
                match unsafe { device_at(index, &raw mut device) } {
                    NVML_SUCCESS => nvml
                        .processes
                        .probe(device, lists)
                        .map_err(|refusal| failed(refusal.call, refusal.code)),
                    code => Err(failed("nvmlDeviceGetHandleByIndex_v2", code)),
                }
            })
            .partition(Result::is_ok);
        nvml.processes.gpus = opened.into_iter().flatten().collect();
        if nvml.processes.gpus.is_empty() {
            return Err(refused
                .into_iter()
                .filter_map(Result::err)
                .last()
                .unwrap_or(NvmlError::NoGpu));
        }
        Ok(nvml)
    }

    /// What the driver attributes to each process on the GPUs now, by pid: on
    /// each GPU the larger of its two lists' figures, since a process on both
    /// is in each, summed over the GPUs. A pid absent from the map holds no GPU
    /// memory.
    pub(crate) fn processes(&mut self) -> Result<Pids, NvmlReadError> {
        self.processes.by_pid().ok_or(NvmlReadError::Unanswered)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: NVML was initialized when `open` returned this value.
        let _ = unsafe { (self.shutdown)() };
    }
}

/// Each listed pid's bytes in one list, its rows summed: a process with two
/// contexts on a GPU has a row for each.
fn rows_by_pid(rows: &[ProcessInfo]) -> Pids {
    merge(
        Pids::new(),
        rows.iter().map(|row| {
            let bytes = match row.used_gpu_memory {
                NVML_VALUE_NOT_AVAILABLE => Err(NotAvailable),
                bytes => Ok(bytes),
            };
            (row.pid, bytes)
        }),
        u64::saturating_add,
    )
}

/// `one` with `other`, `combine`d for a pid in both; a pid with no figure in
/// either has none.
fn merge(
    one: Pids,
    other: impl IntoIterator<Item = (u32, Result<u64, NotAvailable>)>,
    combine: fn(u64, u64) -> u64,
) -> Pids {
    other.into_iter().fold(one, |mut joined, (pid, bytes)| {
        let bytes = match joined.remove(&pid) {
            Some(mine) => mine.and_then(|mine| Ok(combine(mine, bytes?))),
            None => bytes,
        };
        joined.insert(pid, bytes);
        joined
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, used_gpu_memory: u64) -> ProcessInfo {
        ProcessInfo {
            pid,
            used_gpu_memory,
            ..ProcessInfo::default()
        }
    }

    fn pids(pids: Vec<(u32, Result<u64, NotAvailable>)>) -> Pids {
        pids.into_iter().collect()
    }

    #[rstest::rstest]
    #[case::listed(vec![row(7, 900), row(8, 5)], vec![(7, Ok(900)), (8, Ok(5))])]
    #[case::two_contexts(vec![row(7, 900), row(7, 100)], vec![(7, Ok(1000))])]
    #[case::not_available(
        vec![row(7, 900), row(7, NVML_VALUE_NOT_AVAILABLE)],
        vec![(7, Err(NotAvailable))]
    )]
    #[case::another_pid_not_available(
        vec![row(8, NVML_VALUE_NOT_AVAILABLE), row(7, 3)],
        vec![(7, Ok(3)), (8, Err(NotAvailable))]
    )]
    fn a_pid_counts_its_own_rows(
        #[case] rows: Vec<ProcessInfo>,
        #[case] expected: Vec<(u32, Result<u64, NotAvailable>)>,
    ) {
        assert_eq!(rows_by_pid(&rows), pids(expected));
    }

    /// The GPUs the fake driver knows, by handle. Compute lists: one lists pid
    /// 7 at 100 bytes, one lists 70 other processes before pid 7 at 5 bytes
    /// (more than the first read has room for), one refuses, and one needs
    /// more room on every ask. Their graphics lists are [`fake_graphics`]'s.
    const SMALL: usize = 1;
    const LONG: usize = 2;
    const REFUSING: usize = 3;
    const GROWING: usize = 4;
    /// A GPU whose compute list is empty and whose graphics list has pid 7 at
    /// 30 bytes.
    const GRAPHICS: usize = 5;
    /// A GPU whose compute list has no figure for pid 8, and pid 7 at 3 bytes.
    const UNAVAILABLE: usize = 6;
    /// A GPU whose compute list refuses and whose graphics list has pid 7 at
    /// 30 bytes.
    const COMPUTE_REFUSES: usize = 7;
    /// A GPU whose compute list has pid 7 at 30 bytes and that does not
    /// support a graphics list, as a compute-only card may not.
    const UNSUPPORTED: usize = 8;
    /// A GPU whose compute list has pid 7 at 20 bytes on its first ask on a
    /// thread, and refuses every ask after.
    const FLAKY: usize = 9;

    thread_local! {
        /// How often `FLAKY`'s compute list was asked on this thread.
        static FLAKY_ASKS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    const LISTS: [List; 2] = [
        List {
            call: "compute",
            fill: fake_compute,
        },
        List {
            call: "graphics",
            fill: fake_graphics,
        },
    ];

    fn handle(device: usize) -> NvmlDevice {
        std::ptr::without_provenance_mut(device)
    }

    /// `devices`, each read through both lists, as when both answered at the
    /// start.
    fn processes(devices: &[usize]) -> Processes {
        Processes {
            gpus: devices
                .iter()
                .map(|&device| Gpu {
                    device: handle(device),
                    lists: LISTS.to_vec(),
                })
                .collect(),
            rows: vec![ProcessInfo::default(); 64],
        }
    }

    /// `devices` as NVML's start keeps them: each with the lists it answered.
    fn opened(devices: &[usize]) -> Processes {
        let mut processes = processes(&[]);
        processes.gpus = devices
            .iter()
            .filter_map(|&device| processes.probe(handle(device), LISTS).ok())
            .collect();
        processes
    }

    /// Pid 7's figure on one read; a pid not listed holds none.
    fn seven(read: Option<Pids>) -> Option<Result<u64, NotAvailable>> {
        read.map(|mut pids| pids.remove(&7).unwrap_or(Ok(0)))
    }

    unsafe extern "C" fn fake_compute(
        device: NvmlDevice,
        count: *mut c_uint,
        infos: *mut ProcessInfo,
    ) -> NvmlReturn {
        let rows: Vec<ProcessInfo> = match device.addr() {
            SMALL => vec![row(7, 100), row(8, 5)],
            LONG => (100..170)
                .map(|pid| row(pid, 1))
                .chain([row(7, 5)])
                .collect(),
            GROWING | GRAPHICS => Vec::new(),
            UNAVAILABLE => vec![row(8, NVML_VALUE_NOT_AVAILABLE), row(7, 3)],
            UNSUPPORTED => vec![row(7, 30)],
            FLAKY => {
                let asks = FLAKY_ASKS.get();
                FLAKY_ASKS.set(asks.saturating_add(1));
                if asks > 0 {
                    return 999;
                }
                vec![row(7, 20)]
            }
            _ => return 999,
        };
        // SAFETY: the caller's contract, passed on.
        unsafe { answer(device, &rows, count, infos) }
    }

    /// Graphics lists: pid 7 on `SMALL` at the same 100 bytes, as a process on
    /// both lists is, and on `GRAPHICS` and `COMPUTE_REFUSES` at 30; `REFUSING`
    /// refuses, `UNSUPPORTED` has none, and every other GPU lists nothing.
    unsafe extern "C" fn fake_graphics(
        device: NvmlDevice,
        count: *mut c_uint,
        infos: *mut ProcessInfo,
    ) -> NvmlReturn {
        let rows = match device.addr() {
            SMALL => vec![row(7, 100)],
            GRAPHICS | COMPUTE_REFUSES => vec![row(7, 30)],
            REFUSING => return 999,
            UNSUPPORTED => return NVML_ERROR_NOT_SUPPORTED,
            _ => Vec::new(),
        };
        // SAFETY: the caller's contract, passed on.
        unsafe { answer(device, &rows, count, infos) }
    }

    /// NVML's answer for `rows`: the count it needs, and the rows when they fit.
    ///
    /// # Safety
    ///
    /// `count` is live and `infos` has room for `*count` rows.
    unsafe fn answer(
        device: NvmlDevice,
        rows: &[ProcessInfo],
        count: *mut c_uint,
        infos: *mut ProcessInfo,
    ) -> NvmlReturn {
        // SAFETY: the caller passes a live count.
        let room = unsafe { *count };
        let Ok(needed) = c_uint::try_from(rows.len()) else {
            return 999;
        };
        let needed = if device.addr() == GROWING {
            room.saturating_add(1)
        } else {
            needed
        };
        // SAFETY: the caller passes a live count.
        unsafe { *count = needed };
        if room < needed {
            return NVML_ERROR_INSUFFICIENT_SIZE;
        }
        // SAFETY: `infos` has room for `room` rows, at least `rows.len()`.
        unsafe { std::ptr::copy_nonoverlapping(rows.as_ptr(), infos, rows.len()) };
        NVML_SUCCESS
    }

    /// Pid 7's figure: each GPU's read counts only the rows it filled, so a
    /// longer list read before a shorter one leaves nothing behind to count
    /// twice, and a GPU that refuses a list leaves the read unanswered rather
    /// than summed without it.
    #[rstest::rstest]
    #[case::both_gpus(&[SMALL, LONG], Some(Ok(105)))]
    #[case::the_longer_list_first(&[LONG, SMALL], Some(Ok(105)))]
    #[case::a_process_on_both_lists_counts_once(&[SMALL], Some(Ok(100)))]
    #[case::a_graphics_process_counts(&[GRAPHICS, LONG], Some(Ok(35)))]
    #[case::not_available_for_another_pid(&[SMALL, UNAVAILABLE], Some(Ok(103)))]
    #[case::an_unsupported_list_is_empty(&[UNSUPPORTED, SMALL], Some(Ok(130)))]
    #[case::one_of_two_gpus_refuses(&[SMALL, REFUSING], None)]
    #[case::the_first_of_two_gpus_refuses(&[REFUSING, SMALL], None)]
    #[case::one_list_refuses(&[COMPUTE_REFUSES, SMALL], None)]
    #[case::no_gpu_answers(&[REFUSING], None)]
    #[case::a_list_that_keeps_growing_is_given_up(&[GROWING], None)]
    #[case::a_list_that_keeps_growing_beside_an_answer(&[GROWING, GRAPHICS], None)]
    fn a_read_sums_every_gpu(
        #[case] devices: &[usize],
        #[case] expected: Option<Result<u64, NotAvailable>>,
    ) {
        assert_eq!(seven(processes(devices).by_pid()), expected);
    }

    /// At the start a GPU keeps the lists it answers, so one that refuses
    /// both, as a card in MIG mode refuses an unprivileged caller, does not
    /// leave every read unanswered; a list that answered then and refuses
    /// later still does.
    #[rstest::rstest]
    #[case::a_refusing_gpu_is_left_out(&[REFUSING, SMALL], Some(Ok(100)))]
    #[case::a_refusing_list_is_left_out(&[COMPUTE_REFUSES, SMALL], Some(Ok(130)))]
    #[case::a_list_that_refuses_later(&[FLAKY, SMALL], None)]
    fn the_start_keeps_the_lists_that_answer(
        #[case] devices: &[usize],
        #[case] expected: Option<Result<u64, NotAvailable>>,
    ) {
        FLAKY_ASKS.set(0);
        assert_eq!(seven(opened(devices).by_pid()), expected);
    }

    /// A GPU that answers neither list at the start says why, so NVML's open
    /// can when it is the only one.
    #[test]
    fn a_gpu_that_answers_no_list_gives_its_refusal() {
        let refusal = processes(&[]).probe(handle(REFUSING), LISTS).err();
        assert_eq!(
            refusal,
            Some(Refusal {
                call: "graphics",
                code: 999
            })
        );
    }

    #[rstest::rstest]
    #[case::a_process_on_both_lists(&[SMALL], vec![(7, Ok(100)), (8, Ok(5))])]
    #[case::a_graphics_process(&[GRAPHICS], vec![(7, Ok(30))])]
    #[case::summed_over_the_gpus(&[SMALL, GRAPHICS], vec![(7, Ok(130)), (8, Ok(5))])]
    #[case::not_available_for_one_pid(
        &[SMALL, UNAVAILABLE],
        vec![(7, Ok(103)), (8, Err(NotAvailable))]
    )]
    fn every_listed_pid_is_counted(
        #[case] devices: &[usize],
        #[case] expected: Vec<(u32, Result<u64, NotAvailable>)>,
    ) {
        assert_eq!(processes(devices).by_pid(), Some(pids(expected)));
    }
}
