use crate::transport::Transport;

const TRANSPORT_RETRY_DELAY_SECS: u64 = 30;
const TRANSPORT_MAX_RETRIES: usize = 10;

pub(super) fn probe_device(transport: &Transport, label: &str) -> bool {
    match transport.probe() {
        Ok(o) if o.status == 0 => true,
        Ok(o) => {
            eprintln!("[{label}] device probe failed with exit code {}", o.status);
            false
        }
        Err(e) => {
            eprintln!("[{label}] device probe failed: {e:#}");
            false
        }
    }
}

pub(super) fn wait_for_device(transport: &Transport, label: &str) -> bool {
    for retry in 1..=TRANSPORT_MAX_RETRIES {
        eprintln!(
            "[{label}] waiting {TRANSPORT_RETRY_DELAY_SECS}s before retry {retry}/{TRANSPORT_MAX_RETRIES}..."
        );
        std::thread::sleep(std::time::Duration::from_secs(TRANSPORT_RETRY_DELAY_SECS));
        match transport.probe() {
            Ok(o) if o.status == 0 => {
                eprintln!("[{label}] device back online");
                return true;
            }
            Ok(o) => {
                eprintln!("[{label}] probe exit code {}", o.status);
            }
            Err(e) => {
                eprintln!("[{label}] still unreachable: {e:#}");
            }
        }
    }
    eprintln!("[{label}] giving up after {TRANSPORT_MAX_RETRIES} retries");
    false
}
