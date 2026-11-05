//! Tests whose own process holds 256 MiB, in a binary of their own: `VmHWM`
//! keeps the process's high mark, so every test sharing the process (`cargo
//! test` runs a binary's tests in one) would read above it.

#[cfg(unix)]
mod exiting_child;

use pipette_memprobe::{Sampler, Watch};

/// What a test holds in this process.
const HELD: usize = 256 << 20;

/// A peak struck after the last tick still counts while the process lives:
/// `reached` is read after it, and the stop reads later still.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "windows"
))]
#[test]
fn a_peak_just_before_stop_is_counted() -> anyhow::Result<()> {
    let pid = std::process::id();
    let watch = Watch::Peak.supported()?;
    let sampler = Sampler::start(pid, watch)?;
    let block = std::hint::black_box(vec![1u8; HELD]);
    let reached = Sampler::start(pid, watch)?.stop().1?.host_bytes;
    let peak = sampler.stop().1?.host_bytes;
    drop(block);
    assert!(peak >= reached, "peak {peak} is below {reached}");
    Ok(())
}

/// When the sampler stops a child, measured from this process holding 256 MiB.
#[cfg(unix)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stopped {
    /// While it runs.
    Running,
    /// After it exits, before the reap.
    Exited,
    /// After it exits, having exited before the sampler started.
    ExitedBeforeTheStart,
}

/// A child is never read as this process. On Linux its exit figure starts at
/// its image before `exec`: a forked copy of this process, or a spawned one's
/// peak, so with 256 MiB held here a 64 MiB child's figure would only repeat
/// it, and the reads while it ran stand; a child never read running has no
/// peak. macOS reads the exited child's own footprint.
#[cfg(unix)]
#[rstest::rstest]
#[case::forked_and_running(true, Stopped::Running)]
#[case::spawned_and_running(false, Stopped::Running)]
#[case::forked_then_exited(true, Stopped::Exited)]
#[case::spawned_then_exited(false, Stopped::Exited)]
#[case::forked_and_exited_before_the_start(true, Stopped::ExitedBeforeTheStart)]
#[case::spawned_and_exited_before_the_start(false, Stopped::ExitedBeforeTheStart)]
fn a_child_is_never_read_as_this_process(
    #[case] forked: bool,
    #[case] stopped: Stopped,
) -> anyhow::Result<()> {
    let held = std::hint::black_box(vec![1u8; HELD]);
    // Enough blocks to outlive a few ticks, or to run until it is killed.
    let blocks = match stopped {
        Stopped::Running => u32::MAX,
        Stopped::Exited | Stopped::ExitedBeforeTheStart => 32,
    };
    let mut child = if forked {
        fork(blocks)?
    } else {
        exiting_child::spawn(blocks)?
    };
    if stopped == Stopped::ExitedBeforeTheStart {
        exiting_child::release(&mut child)?;
        exiting_child::wait_unreaped(child.id())?;
    }
    let watch = Watch::Peak.supported()?;
    let sampler = Sampler::start(child.id(), watch)?;
    match stopped {
        Stopped::Running => {
            exiting_child::release(&mut child)?;
            // A fixed sleep lost the race with dd filling its buffer on a
            // loaded runner. A timeout falls through, so the child is killed.
            let started = std::time::Instant::now();
            while started.elapsed() < std::time::Duration::from_secs(30)
                && Sampler::start(child.id(), watch)?.stop().1?.host_bytes
                    < exiting_child::ALLOCATED
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        Stopped::Exited => {
            exiting_child::release(&mut child)?;
            exiting_child::wait_unreaped(child.id())?;
        }
        Stopped::ExitedBeforeTheStart => {}
    }
    let (observed, peak) = sampler.stop();
    let _ = child.kill();
    child.wait()?;
    drop(held);

    #[cfg(any(target_os = "linux", target_os = "android"))]
    if stopped == Stopped::ExitedBeforeTheStart {
        assert!(
            matches!(peak, Err(pipette_memprobe::PeakError::NoVmHwm { .. })),
            "{peak:?}"
        );
        assert_eq!(observed, pipette_plan_types::MemoryObservation::default());
        return Ok(());
    }
    let bytes = peak?.host_bytes;
    let near = exiting_child::ALLOCATED..exiting_child::ALLOCATED + (32 << 20);
    assert!(
        near.contains(&bytes),
        "{bytes} is not near the child's 64 MiB"
    );
    assert_eq!(observed.max_host_bytes, Some(bytes));
    Ok(())
}

/// [`exiting_child::spawn`], but forked as `pipette_subprocess` does, rather
/// than through `posix_spawn`.
#[cfg(unix)]
fn fork(blocks: u32) -> anyhow::Result<std::process::Child> {
    use std::os::unix::process::CommandExt as _;

    let mut command = exiting_child::command(blocks);
    // SAFETY: the hook does nothing in the forked child; it is there only to
    // keep std from spawning through `posix_spawn`.
    unsafe { command.pre_exec(|| Ok(())) };
    #[expect(clippy::disallowed_methods, reason = "its output goes to null")]
    let child = command.spawn()?;
    Ok(child)
}
