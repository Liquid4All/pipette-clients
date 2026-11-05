//! A Windows Job Object that stops every started child when this process
//! exits. Windows does not stop a child when its parent exits. A hard kill
//! (`TerminateProcess`, `Stop-Process -Force`) runs no teardown.
//!
//! Only this process holds a handle to the job. The system closes the handle
//! when the process exits for any reason. `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
//! then stops every process in the job.
//!
//! A child joins the job after it starts. A process that the child starts in
//! that interval stays outside the job.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::process::Child;
use std::sync::OnceLock;

use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};

/// Add `child`, and every process that it starts later, to the job.
pub(crate) fn assign(child: &Child) {
    let Some(job) = job() else {
        return;
    };
    // SAFETY: both handles stay open for the call.
    let assigned =
        unsafe { AssignProcessToJobObject(job.as_raw_handle(), child.as_raw_handle()) } != 0;
    if !assigned {
        // Read before any other call can overwrite it.
        let err = std::io::Error::last_os_error();
        tracing::warn!(
            pid = child.id(),
            error = %err,
            "failed to add the child to the job that stops it with this process"
        );
    }
}

/// The job. The first call makes it, and gets `None` if it cannot.
fn job() -> Option<&'static OwnedHandle> {
    static JOB: OnceLock<Option<OwnedHandle>> = OnceLock::new();
    JOB.get_or_init(|| {
        let made = create();
        if let Err(err) = &made {
            tracing::warn!(
                error = %err,
                "failed to make the job that stops started children with this process"
            );
        }
        made.ok()
    })
    .as_ref()
}

fn create() -> std::io::Result<OwnedHandle> {
    // SAFETY: null attributes and a null name ask for an unnamed job with a
    // handle that children do not inherit.
    let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if raw.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a fresh handle that nothing else owns.
    let job = unsafe { OwnedHandle::from_raw_handle(raw as RawHandle) };
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    let size = u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
        .map_err(std::io::Error::other)?;
    // SAFETY: `limits` is a live struct of `size` bytes, of the class named.
    let set = unsafe {
        SetInformationJobObject(
            job.as_raw_handle(),
            JobObjectExtendedLimitInformation,
            (&raw const limits).cast(),
            size,
        )
    } != 0;
    if !set {
        return Err(std::io::Error::last_os_error());
    }
    Ok(job)
}
