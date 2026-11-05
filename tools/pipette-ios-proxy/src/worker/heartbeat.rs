//! The lease heartbeat for one run, on its own thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use crate::server::{self, LeaseKeepalive, Session};
use crate::timing;

/// A running heartbeat thread, stopped by [`HeartbeatGuard::stop`] or by
/// dropping the guard. `Drop` drops the sender FIRST so a thread parked in
/// `recv_timeout` wakes immediately: the disconnect IS the stop signal and
/// `recv_timeout` is the sleep.
#[derive(Debug)]
pub(crate) struct HeartbeatGuard {
    stop_tx: Option<std::sync::mpsc::Sender<()>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl HeartbeatGuard {
    /// Stop beating now, keeping the guard: the app has submitted, which closes
    /// the lease, so the next beat would read as a lost one.
    pub(crate) fn stop(&self) {
        if let Some(stop_tx) = &self.stop_tx {
            // Gone already when the thread ended on its own.
            let _ = stop_tx.send(());
        }
    }
}

impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        drop(self.stop_tx.take());
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Heartbeat for the duration of one launch. The first tick is after `every`: the
/// claim that produced the job has just renewed the lease. `lease_lost` is what
/// the main thread reads afterwards to decide whether to submit at all.
pub(crate) fn spawn_heartbeat(
    ctx: Arc<Session>,
    job_id: String,
    every: Duration,
    lease_lost: Arc<AtomicBool>,
) -> HeartbeatGuard {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    // The caller's span and subscriber: every line names the job, and a test
    // capturing the caller's thread sees these too.
    let span = tracing::Span::current();
    let dispatch = tracing::dispatcher::get_default(tracing::Dispatch::clone);
    let join = thread::spawn(move || {
        let _dispatch = tracing::dispatcher::set_default(&dispatch);
        let _span = span.entered();
        let every_secs = every.as_secs().to_string();
        tracing::debug!(
            first_tick_secs = %every_secs,
            "heartbeat thread started"
        );
        let mut wait = every;
        let mut ticks: u32 = 0;
        loop {
            match stop_rx.recv_timeout(wait) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                    tracing::debug!(
                        ticks = %ticks,
                        "heartbeat thread stopped"
                    );
                    return;
                }
                Err(RecvTimeoutError::Timeout) => match server::keepalive_lease(&ctx, &job_id) {
                    LeaseKeepalive::Ok => {
                        ticks += 1;
                        tracing::debug!(
                            tick = %ticks,
                            next_secs = %every_secs,
                            "heartbeat ok"
                        );
                        wait = every;
                    }
                    LeaseKeepalive::Abort => {
                        // Nothing on the phone takes a cancel token, so the flag
                        // is the whole mechanism.
                        lease_lost.store(true, Ordering::SeqCst);
                        tracing::error!(
                            ticks = %ticks,
                            "heartbeat lost the lease; the run is terminated and submit is skipped"
                        );
                        return;
                    }
                    LeaseKeepalive::Retry => {
                        tracing::warn!(
                            retry_secs = %timing::HEARTBEAT_RETRY_WAIT.as_secs().to_string(),
                            "heartbeat failed transiently; retrying"
                        );
                        wait = timing::HEARTBEAT_RETRY_WAIT;
                    }
                },
            }
        }
    });
    HeartbeatGuard {
        stop_tx: Some(stop_tx),
        join: Some(join),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use httpmock::prelude::*;

    use super::*;
    use crate::device::fake::FakeDevice;
    use crate::logging::testing::captured;
    use crate::worker::testing::{phone, session};

    /// The heartbeat runs on a thread of its own, and still names the job.
    #[test]
    fn the_heartbeat_names_its_job() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (phone, _) = phone(FakeDevice::new("phone-a"), state.path())?;
        let server = MockServer::start();
        let taken = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-1/heartbeat");
            then.status(409);
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);
        let lost = Arc::new(AtomicBool::new(false));

        let ((), log) = captured(|| {
            let _job = tracing::info_span!("job", id = %"job-1").entered();
            let _heartbeat = spawn_heartbeat(
                ctx,
                "job-1".to_owned(),
                Duration::from_millis(1),
                Arc::clone(&lost),
            );
            let deadline = Instant::now() + Duration::from_secs(10);
            while !lost.load(Ordering::SeqCst) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
        });

        taken.assert();
        assert!(lost.load(Ordering::SeqCst));
        assert!(
            log.contains("job{id=job-1}: heartbeat lost the lease"),
            "{log}"
        );
        Ok(())
    }

    /// Once stopped, the thread beats no more: a lease the app's upload closed,
    /// `409` from then on, is not read as lost.
    #[test]
    fn a_stopped_heartbeat_beats_no_more() -> anyhow::Result<()> {
        let state = tempfile::tempdir()?;
        let (phone, _) = phone(FakeDevice::new("phone-a"), state.path())?;
        let server = MockServer::start();
        let beat = server.mock(|when, then| {
            when.method(PUT).path("/plans/job-1/heartbeat");
            then.status(409);
        });
        let ctx = Arc::new(session(&phone, &server.base_url())?);
        let lost = Arc::new(AtomicBool::new(false));
        let every = Duration::from_millis(200);
        let heartbeat = spawn_heartbeat(ctx, "job-1".to_owned(), every, Arc::clone(&lost));

        heartbeat.stop();
        // Three ticks' worth: an unstopped thread would have beaten and lost.
        thread::sleep(every * 3);

        beat.assert_calls(0);
        assert!(!lost.load(Ordering::SeqCst));
        drop(heartbeat);
        Ok(())
    }
}
