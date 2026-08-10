// Empirical test: does the ~33 MB "driver overhead" double when two
// llama-bench processes use the GPU simultaneously, or is some of it
// shared / amortized?
//
// Approach:
//   1. Sample adapter-wide `Total Committed` baseline (with nothing
//      else running) — the idle floor from DWM, drivers etc.
//   2. Spawn N llama-bench instances concurrently.
//   3. Poll all three at 20 ms:
//        - Each spawned PID's `\GPU Process Memory(...)\Total Committed`
//        - The adapter's `\GPU Adapter Memory(*)\Total Committed`
//      Track per-PID peaks and the adapter's *delta from baseline*
//      (= physically attributable to our processes).
//   4. After all children exit, compare:
//        - Σ per-PID peaks (what PDH attributes to each, summed)
//        - Adapter Δ (what physically appeared on the GPU above
//          the baseline)
//      If they agree, accounting is straightforward per-process and
//      driver overhead truly doubles. If adapter Δ < Σ per-PID, the
//      driver is sharing pages and PDH double-counts. If adapter Δ >
//      Σ per-PID, something else is showing up.

use std::{
    ffi::c_void,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use windows_sys::Win32::{
    Foundation::ERROR_SUCCESS,
    System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
        PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE,
    },
};

type PdhQuery = *mut c_void;
type PdhCounter = *mut c_void;

const PDH_MORE_DATA: u32 = 0x800007D2;
const POLL_MS: u64 = 20;

const LLAMA_BENCH: &str = r"C:\Users\yuri\mem-test\llama-b9058\llama-bench.exe";
const MODEL: &str = r"C:\Users\yuri\mem-test\LFM2-350M-Q4_K_M.gguf";

fn main() {
    let n_proc: u32 = std::env::args()
        .nth(1)
        .expect("usage: pdh-exp <n_processes> <n_prompt>")
        .parse()
        .expect("n_processes u32");
    let ctx: u32 = std::env::args()
        .nth(2)
        .expect("usage: pdh-exp <n_processes> <n_prompt>")
        .parse()
        .expect("n_prompt u32");

    eprintln!(">>> n_processes={n_proc} n_prompt={ctx}");

    // 1. Open a query first; sample adapter baseline before spawning
    //    anything so we can subtract the idle floor.
    let mut query: PdhQuery = std::ptr::null_mut();
    let rc = unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut query) };
    assert_eq!(rc, ERROR_SUCCESS, "PdhOpenQueryW");

    let adapter_counter = add_counter(query, r"\GPU Adapter Memory(*)\Total Committed");

    // Settle the adapter counter. Two collects, because the first
    // sample of a PDH query is sometimes empty.
    let mut buf = Vec::with_capacity(65_536);
    for _ in 0..2 {
        unsafe {
            let _ = PdhCollectQueryData(query);
        }
        thread::sleep(Duration::from_millis(50));
    }
    let adapter_baseline = sum_counter(adapter_counter, &mut buf);
    eprintln!(
        ">>> adapter baseline: {} bytes ({:.2} MiB)",
        adapter_baseline,
        adapter_baseline as f64 / 1_048_576.0
    );

    // 2. Spawn N llama-bench processes concurrently.
    let mut children: Vec<std::process::Child> = (0..n_proc)
        .map(|_| {
            Command::new(LLAMA_BENCH)
                .args([
                    "--output",
                    "json",
                    "--model",
                    MODEL,
                    "--mmap",
                    "0",
                    "--n-prompt",
                    &ctx.to_string(),
                    "--n-gen",
                    "1",
                    "-r",
                    "1",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn")
        })
        .collect();
    let pids: Vec<u32> = children.iter().map(|c| c.id()).collect();
    eprintln!(">>> spawned pids: {pids:?}");

    // Per-PID counters.
    let pid_counters: Vec<PdhCounter> = pids
        .iter()
        .map(|pid| {
            add_counter(
                query,
                &format!("\\GPU Process Memory(pid_{pid}_*)\\Total Committed"),
            )
        })
        .collect();

    // 3. Poll until all children exit.
    let mut pid_peaks = vec![0u64; pids.len()];
    let mut adapter_peak: u64 = 0;
    let mut adapter_peak_delta: u64 = 0;
    let mut samples: u32 = 0;
    loop {
        let mut all_done = true;
        for c in children.iter_mut() {
            match c.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => all_done = false,
                Err(e) => panic!("try_wait: {e}"),
            }
        }
        samples += 1;
        if unsafe { PdhCollectQueryData(query) } == ERROR_SUCCESS {
            for (i, &c) in pid_counters.iter().enumerate() {
                let v = sum_counter(c, &mut buf);
                if v > pid_peaks[i] {
                    pid_peaks[i] = v;
                }
            }
            let a = sum_counter(adapter_counter, &mut buf);
            if a > adapter_peak {
                adapter_peak = a;
            }
            let delta = a.saturating_sub(adapter_baseline);
            if delta > adapter_peak_delta {
                adapter_peak_delta = delta;
            }
        }
        if all_done {
            break;
        }
        thread::sleep(Duration::from_millis(POLL_MS));
    }

    unsafe {
        let _ = PdhCloseQuery(query);
    }

    // 4. Report.
    println!();
    println!("=== n_processes={n_proc} n_prompt={ctx} samples={samples} ===");
    for (i, pid) in pids.iter().enumerate() {
        let mib = pid_peaks[i] as f64 / 1_048_576.0;
        println!(
            "  pid {pid:>6}  Total Committed peak  {:>14} bytes  {:>10.2} MiB",
            pid_peaks[i], mib
        );
    }
    let sum_peaks: u64 = pid_peaks.iter().sum();
    println!(
        "  Σ per-PID peaks                       {:>14} bytes  {:>10.2} MiB",
        sum_peaks,
        sum_peaks as f64 / 1_048_576.0
    );
    println!(
        "  adapter Total Committed baseline     {:>14} bytes  {:>10.2} MiB",
        adapter_baseline,
        adapter_baseline as f64 / 1_048_576.0
    );
    println!(
        "  adapter Total Committed peak (raw)   {:>14} bytes  {:>10.2} MiB",
        adapter_peak,
        adapter_peak as f64 / 1_048_576.0
    );
    println!(
        "  adapter Δ (peak − baseline)          {:>14} bytes  {:>10.2} MiB",
        adapter_peak_delta,
        adapter_peak_delta as f64 / 1_048_576.0
    );
    let ratio = if adapter_peak_delta > 0 {
        sum_peaks as f64 / adapter_peak_delta as f64
    } else {
        0.0
    };
    println!(
        "  Σ per-PID / adapter Δ                {ratio:>14.3}     \
         (1.0 = perfect agreement; >1 = PDH double-counts)"
    );
}

fn add_counter(query: PdhQuery, path: &str) -> PdhCounter {
    let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    let mut counter: PdhCounter = std::ptr::null_mut();
    let rc = unsafe { PdhAddEnglishCounterW(query, wide.as_ptr(), 0, &mut counter) };
    assert_eq!(rc, ERROR_SUCCESS, "PdhAddEnglishCounterW({path})");
    counter
}

fn sum_counter(counter: PdhCounter, buffer: &mut Vec<u8>) -> u64 {
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
        return 0;
    }
    let items: *const PDH_FMT_COUNTERVALUE_ITEM_W =
        buffer.as_ptr() as *const PDH_FMT_COUNTERVALUE_ITEM_W;
    let mut total: u64 = 0;
    for i in 0..item_count as isize {
        let item = unsafe { &*items.offset(i) };
        if item.FmtValue.CStatus != 0 {
            continue;
        }
        let v = unsafe { item.FmtValue.Anonymous.largeValue };
        if v > 0 {
            total = total.saturating_add(v as u64);
        }
    }
    total
}
