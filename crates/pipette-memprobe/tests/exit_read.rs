//! A process that exits is read until it is reaped, and not after. A binary of
//! its own: the Linux exit figure counts only above this process's own peak,
//! which a test that holds memory would raise for every test sharing the
//! process (`cargo test` runs a binary's tests in one).
#![cfg(unix)]

mod exiting_child;

use pipette_memprobe::{PeakError, Sampler, Watch};
use pipette_plan_types::MemoryObservation;

/// A sampler stopped after its process exits but before the reap still
/// reads the run, even when no tick landed while it ran (the observation's
/// 100 ms is longer than the child lives), or it had exited at the start.
#[rstest::rstest]
#[case::observed(Watch::Observe, false)]
#[case::peak(Watch::Peak, false)]
#[case::exited_at_the_start(Watch::Peak, true)]
fn a_process_stopped_before_its_reap_is_read(
    #[case] watch: Watch,
    #[case] exited_at_the_start: bool,
) -> anyhow::Result<()> {
    let mut child = exiting_child::spawn(1)?;
    if exited_at_the_start {
        exiting_child::release(&mut child)?;
        exiting_child::wait_unreaped(child.id())?;
    }
    let sampler = Sampler::start(child.id(), watch.supported()?)?;
    if !exited_at_the_start {
        exiting_child::release(&mut child)?;
        exiting_child::wait_unreaped(child.id())?;
    }
    let (observed, peak) = sampler.stop();
    child.wait()?;
    let host_bytes = if watch == Watch::Observe {
        observed.max_host_bytes
    } else {
        Some(peak?.host_bytes)
    };
    anyhow::ensure!(
        host_bytes.is_some_and(|bytes| bytes >= exiting_child::ALLOCATED),
        "{host_bytes:?} is below the {} bytes it held",
        exiting_child::ALLOCATED
    );
    Ok(())
}

/// Reaped before the stop, it has no read there, so the reads before it
/// cannot pass for the run: the peak is refused and the observation withheld.
/// It outlives a few ticks, so those reads did land.
#[rstest::rstest]
#[case::observed(Watch::Observe)]
#[case::peak(Watch::Peak)]
fn a_process_reaped_before_the_stop_is_refused(#[case] watch: Watch) -> anyhow::Result<()> {
    let mut child = exiting_child::spawn(32)?;
    let sampler = Sampler::start(child.id(), watch.supported()?)?;
    exiting_child::release(&mut child)?;
    child.wait()?;
    let (observed, peak) = sampler.stop();
    assert_eq!(observed, MemoryObservation::default());
    if watch == Watch::Peak {
        assert!(
            matches!(peak, Err(PeakError::UnreadAtStop { .. })),
            "{peak:?}"
        );
    }
    Ok(())
}
