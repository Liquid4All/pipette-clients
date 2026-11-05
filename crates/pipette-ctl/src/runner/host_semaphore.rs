//! Per-physical-machine concurrency throttle: one `HostSemaphore` per
//! distinct `physical_id`, held around `transport.exec`. Capacity is the
//! largest `parallelism` among the transports sharing that `physical_id`.

use std::num::NonZeroUsize;
use std::sync::{Condvar, Mutex};

#[derive(Debug)]
pub(super) struct HostSemaphore {
    free: Mutex<usize>,
    cv: Condvar,
}

impl HostSemaphore {
    pub(super) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            free: Mutex::new(capacity.get()),
            cv: Condvar::new(),
        }
    }

    /// Blocks until a slot is free; the guard releases it on drop.
    pub(super) fn acquire(&self) -> HostSemaphoreGuard<'_> {
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while *free == 0 {
            free = self
                .cv
                .wait(free)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *free -= 1;
        HostSemaphoreGuard { sem: self }
    }
}

pub(super) struct HostSemaphoreGuard<'a> {
    sem: &'a HostSemaphore,
}

impl Drop for HostSemaphoreGuard<'_> {
    fn drop(&mut self) {
        let mut free = self
            .sem
            .free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *free += 1;
        self.sem.cv.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    use super::*;

    fn non_zero(n: usize) -> anyhow::Result<NonZeroUsize> {
        NonZeroUsize::new(n).ok_or_else(|| anyhow::anyhow!("expected non-zero capacity"))
    }

    // Counts the release before dropping the guard, so any acquirer that
    // gets the freed slot is guaranteed to see the count.
    fn spawn_holder(
        sem: &Arc<HostSemaphore>,
        releases: &Arc<AtomicUsize>,
        acquired_signal: &Arc<Barrier>,
        hold: Duration,
    ) -> std::thread::JoinHandle<()> {
        let sem = Arc::clone(sem);
        let releases = Arc::clone(releases);
        let acquired_signal = Arc::clone(acquired_signal);
        std::thread::spawn(move || {
            let g = sem.acquire();
            acquired_signal.wait();
            std::thread::sleep(hold);
            releases.fetch_add(1, Ordering::SeqCst);
            drop(g);
        })
    }

    #[test]
    fn capacity_one_serializes() -> anyhow::Result<()> {
        let sem = Arc::new(HostSemaphore::new(non_zero(1)?));
        let releases = Arc::new(AtomicUsize::new(0));
        let after_t1_acquired = Arc::new(Barrier::new(2));

        let t1 = spawn_holder(
            &sem,
            &releases,
            &after_t1_acquired,
            Duration::from_millis(20),
        );

        let s2 = Arc::clone(&sem);
        let b2 = Arc::clone(&after_t1_acquired);
        let r2 = Arc::clone(&releases);
        let t2 = std::thread::spawn(move || {
            b2.wait();
            let _g = s2.acquire();
            r2.load(Ordering::SeqCst)
        });

        t1.join()
            .map_err(|e| anyhow::anyhow!("t1 panicked: {e:?}"))?;
        let releases_at_t2 = t2
            .join()
            .map_err(|e| anyhow::anyhow!("t2 panicked: {e:?}"))?;
        assert_eq!(
            releases_at_t2, 1,
            "t2 acquired after {releases_at_t2} releases; expected 1 (waited for t1)"
        );
        Ok(())
    }

    #[test]
    fn capacity_two_runs_concurrently() -> anyhow::Result<()> {
        let sem = Arc::new(HostSemaphore::new(non_zero(2)?));
        let currently = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let both_in_critical = Arc::new(Barrier::new(2));

        let make_holder = || {
            let sem = Arc::clone(&sem);
            let currently = Arc::clone(&currently);
            let peak = Arc::clone(&peak);
            let barrier = Arc::clone(&both_in_critical);
            std::thread::spawn(move || {
                let _g = sem.acquire();
                let n = currently.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(n, Ordering::SeqCst);
                barrier.wait();
                currently.fetch_sub(1, Ordering::SeqCst);
            })
        };
        let t1 = make_holder();
        let t2 = make_holder();
        t1.join()
            .map_err(|e| anyhow::anyhow!("t1 panicked: {e:?}"))?;
        t2.join()
            .map_err(|e| anyhow::anyhow!("t2 panicked: {e:?}"))?;
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn third_acquirer_waits_when_capacity_two() -> anyhow::Result<()> {
        let sem = Arc::new(HostSemaphore::new(non_zero(2)?));
        let releases = Arc::new(AtomicUsize::new(0));
        let after_t1_t2_acquired = Arc::new(Barrier::new(3));

        let t1 = spawn_holder(
            &sem,
            &releases,
            &after_t1_t2_acquired,
            Duration::from_millis(20),
        );
        let t2 = spawn_holder(
            &sem,
            &releases,
            &after_t1_t2_acquired,
            Duration::from_millis(20),
        );

        let s3 = Arc::clone(&sem);
        let b3 = Arc::clone(&after_t1_t2_acquired);
        let r3 = Arc::clone(&releases);
        let t3 = std::thread::spawn(move || {
            b3.wait();
            let _g = s3.acquire();
            r3.load(Ordering::SeqCst)
        });

        t1.join()
            .map_err(|e| anyhow::anyhow!("t1 panicked: {e:?}"))?;
        t2.join()
            .map_err(|e| anyhow::anyhow!("t2 panicked: {e:?}"))?;
        let releases_at_t3 = t3
            .join()
            .map_err(|e| anyhow::anyhow!("t3 panicked: {e:?}"))?;
        assert!(
            releases_at_t3 >= 1,
            "t3 saw {releases_at_t3} releases at acquire time; expected to wait for at least 1"
        );
        Ok(())
    }
}
