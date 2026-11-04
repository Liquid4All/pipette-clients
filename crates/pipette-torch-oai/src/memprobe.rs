//! Which cgroup holds a cell's server, for `pipette_memprobe`'s cgroup sampler,
//! which reads its memory on the host from the server's launch, so the load
//! counts. pipette-torch-oai must run on the host, not in a container of its
//! own:
//!
//! - **docker:** the container's, once `docker container inspect` names it and
//!   its main process, whose `/proc/<pid>/cgroup` places it.
//! - **uv:** pipette's own, which the server shares, as a child does: its
//!   session and what that starts count, and the cgroup's other
//!   processes, pipette's own among them, are taken off.
//!
//! A memory cell reads the host figure every 10 ms, and an NVIDIA or AMD
//! flavor's GPU on the same tick; any other docker cell but `eval` observes
//! the host figure every 100 ms.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::time::Duration;

use pipette_memprobe::cgroup::{self, Cgroup, Members, TreeError, Unlocated};
use pipette_plan_types::{MemoryObservation, Runtime, VllmFlavor, flavor_from_uv_build};
use pipette_subprocess::OutputError;

use crate::{cell_flags, flavor::sglang_to_vllm_flavor};

/// How long one of the probe's `docker` calls may take, so a stalled daemon
/// cannot hold the probe, or the drop that joins its thread, for good.
pub(crate) const DOCKER_LIMIT: Duration = Duration::from_secs(30);

pub(crate) type Probe = cgroup::Sampler<LocateError>;

/// Why one of the probe's commands has no output.
#[derive(Debug, thiserror::Error)]
#[error("running `{command}`")]
pub(crate) struct RunError {
    command: String,
    source: OutputError,
}

/// Why `docker container inspect` named no running container.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InspectError {
    #[error(transparent)]
    Run(#[from] RunError),
    #[error("`docker container inspect {container}` failed ({status}): {stderr}")]
    Failed {
        container: String,
        status: ExitStatus,
        stderr: String,
    },
    #[error(
        "`docker container inspect {container}` printed {printed:?}, not a container id and pid"
    )]
    Unparsed { container: String, printed: String },
    #[error("`docker container inspect {container}` shows no process: it has not started")]
    NotStarted { container: String },
}

/// Why the cgroup of a cell's server is not found.
#[derive(Debug, thiserror::Error)]
pub(crate) enum LocateError {
    #[error("`docker container inspect` never named the container")]
    Uninspected(#[source] InspectError),
    #[error(
        "could not read {path}, which places the container's main process: pipette runs in \
         another pid namespace than the container, docker in a VM, or /proc hides other users' \
         processes (hidepid)"
    )]
    Unplaced {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "{path} names no cgroup v2 path of container {id}: docker runs in another pid namespace \
         or a VM, or `docker` is another runtime that names its cgroups otherwise"
    )]
    Elsewhere { path: PathBuf, id: String },
    #[error("the uv server shares pipette's own cgroup, which counts no memory of its own")]
    Own(#[source] TreeError),
}

/// `program` with `args`, killed when it has not exited within `limit`.
pub(crate) fn output_within(
    program: &Path,
    args: &[&str],
    limit: Duration,
) -> Result<Output, RunError> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    pipette_subprocess::echo_debug(&cmd);
    let command = pipette_subprocess::render(&cmd);
    pipette_subprocess::output_within(cmd, limit).map_err(|source| RunError { command, source })
}

/// A container's full id, as its processes' cgroups name it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContainerId(String);

impl ContainerId {
    /// What `docker container inspect --format {{.Id}}` prints: 64 hex digits.
    fn parse(printed: &str) -> Option<Self> {
        (printed.len() == 64 && printed.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .then(|| Self(printed.to_owned()))
    }

    /// Whether a part of a cgroup path is the container's own cgroup:
    /// `docker-<id>.scope` under systemd's driver and `<id>` under cgroupfs.
    fn names(&self, part: &str) -> bool {
        part == self.0
            || part
                .strip_prefix("docker-")
                .and_then(|rest| rest.strip_suffix(".scope"))
                == Some(self.0.as_str())
    }
}

/// The container as `docker container inspect` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Container {
    id: ContainerId,
    /// Its main process on the host; 0 until it has started.
    pid: u32,
}

impl Container {
    /// What `docker container inspect --format '{{.Id}} {{.State.Pid}}'`
    /// prints.
    fn parse(printed: &str) -> Option<Self> {
        let (id, pid) = printed.split_once(' ')?;
        Some(Self {
            id: ContainerId::parse(id)?,
            pid: pid.parse().ok()?,
        })
    }
}

/// The container and its main process, once `docker run` has started it;
/// `docker` is killed when it has not answered within `limit`.
fn inspect(docker: &Path, container: &str, limit: Duration) -> Result<Container, InspectError> {
    let output = output_within(
        docker,
        &[
            "container",
            "inspect",
            "--format",
            "{{.Id}} {{.State.Pid}}",
            container,
        ],
        limit,
    )?;
    if !output.status.success() {
        return Err(InspectError::Failed {
            container: container.to_owned(),
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let printed = String::from_utf8_lossy(&output.stdout);
    let printed = printed.trim();
    match Container::parse(printed) {
        Some(Container { pid: 0, .. }) => Err(InspectError::NotStarted {
            container: container.to_owned(),
        }),
        Some(found) => Ok(found),
        None => Err(InspectError::Unparsed {
            container: container.to_owned(),
            printed: printed.to_owned(),
        }),
    }
}

/// Where a cell's server has its cgroup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServerCgroup {
    /// The container `docker` runs, by its name.
    Container { docker: PathBuf, name: String },
    /// pipette's own, which the uv server, the leader of session `pid`,
    /// shares.
    Shared { pid: u32 },
}

impl ServerCgroup {
    /// The cgroup: the container's, asked again until `docker` names it, or
    /// pipette's own.
    pub(crate) fn locate(&self, tree: &cgroup::Tree) -> Result<Cgroup, Unlocated<LocateError>> {
        match self {
            Self::Container { docker, name } => {
                let container = inspect(docker, name, DOCKER_LIMIT)
                    .map_err(|err| Unlocated::Yet(LocateError::Uninspected(err)))?;
                container_cgroup(tree, &container).map_err(Unlocated::Failed)
            }
            Self::Shared { .. } => tree
                .own_cgroup()
                .map_err(|err| Unlocated::Failed(LocateError::Own(err))),
        }
    }

    /// Whose memory in the cgroup is the server's: every process in a
    /// container, the uv server's session and what it starts in pipette's
    /// cgroup.
    pub(crate) fn members(&self) -> Members {
        match self {
            Self::Container { .. } => Members::All,
            Self::Shared { pid } => Members::Session { leader: *pid },
        }
    }
}

/// The container's cgroup, where `/proc/<pid>/cgroup` places its main
/// process. A process in a cgroup below the container's, as docker in docker
/// puts its init, gives the container's.
fn container_cgroup(tree: &cgroup::Tree, container: &Container) -> Result<Cgroup, LocateError> {
    let path = tree.cgroups_of(container.pid);
    let cgroups = match std::fs::read_to_string(&path) {
        Ok(cgroups) => cgroups,
        Err(source) => return Err(LocateError::Unplaced { path, source }),
    };
    Cgroup::named(&cgroups, |part| container.id.names(part)).ok_or_else(|| LocateError::Elsewhere {
        path,
        id: container.id.0.clone(),
    })
}

/// What a memory cell reads of its server, picked from the runtime's flavor
/// before the launch: a uv build's from its tag. A cell whose GPU memory
/// cannot be read is refused here, before anything starts. NVML opens for an
/// NVIDIA flavor, which takes over a second on a host whose driver is not
/// kept loaded; an AMD flavor reads what its job's cards hold before the
/// launch.
pub(crate) fn memory_watch(runtime: &Runtime) -> anyhow::Result<cgroup::Watch> {
    let flavor = match runtime {
        Runtime::DockerVllm(d) => d.flavor,
        Runtime::DockerSglang(d) => sglang_to_vllm_flavor(d.flavor),
        Runtime::UvVllm(uv) => flavor_from_uv_build(&uv.build),
        Runtime::UvSglang(uv) => flavor_from_uv_build(&uv.build),
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => return Err(cell_flags::unsupported_runtime(other)),
    };
    match flavor {
        VllmFlavor::Cpu => Ok(cgroup::Watch::Peak),
        VllmFlavor::AmdGpu => open_amd(),
        VllmFlavor::NvidiaGpu => open_nvidia(),
    }
}

#[cfg(target_os = "linux")]
fn open_amd() -> anyhow::Result<cgroup::Watch> {
    Ok(cgroup::Watch::PeakWithAmdGpu(
        pipette_memprobe::amd::AmdGpu::open()?,
    ))
}

#[cfg(not(target_os = "linux"))]
fn open_amd() -> anyhow::Result<cgroup::Watch> {
    anyhow::bail!(
        "a server's AMD GPU memory is read from the driver on a Linux host only, so this cell's \
         peak would be the host's alone; run it on Linux"
    )
}

#[cfg(target_os = "linux")]
fn open_nvidia() -> anyhow::Result<cgroup::Watch> {
    Ok(cgroup::Watch::PeakWithNvidiaGpu(cgroup::NvidiaGpu::open()?))
}

#[cfg(not(target_os = "linux"))]
fn open_nvidia() -> anyhow::Result<cgroup::Watch> {
    anyhow::bail!(
        "a server's NVIDIA GPU memory is read through NVML on a Linux host only, so this cell's \
         peak would be the host's alone; run it on Linux"
    )
}

/// The probe's observation, for a cell that reports time, which watching
/// cannot fail: nothing, with a warning, when it read no figure.
pub(crate) fn observe(probe: Probe) -> MemoryObservation {
    probe.stop().map_or_else(
        |err| {
            tracing::warn!(
                error = %format_args!("{:#}", anyhow::Error::from(err)),
                "no memory observation from the container"
            );
            MemoryObservation::default()
        },
        |peak| peak.memory,
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, DockerSglang, DockerVllm, NonEmptyString, SglangFlavor, UvBuild,
        UvPythonVersion, UvRuntimeSource, UvServerVersion, UvSglang, UvVllm,
    };

    use super::*;
    use crate::server::test_support::{
        FAKE_CONTAINER_ID, FAKE_PID, FAKE_SESSION, fake_session, fake_tree,
    };

    const OTHER_ID: &str = "0e542162bd21d5a7e966b18488d758dd974af53e757b73375b2c91d9160ec50c";

    fn vllm(flavor: VllmFlavor) -> anyhow::Result<Runtime> {
        Ok(Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.21.0".to_owned())?,
            flavor,
        }))
    }

    fn sglang(flavor: SglangFlavor) -> anyhow::Result<Runtime> {
        Ok(Runtime::DockerSglang(DockerSglang {
            image_name: NonEmptyString::try_new("lmsysorg/sglang".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.5.21".to_owned())?,
            flavor,
        }))
    }

    fn uv_build(build: &str) -> anyhow::Result<UvBuild> {
        Ok(UvBuild::try_new(build.to_owned())?)
    }

    fn uv_source() -> anyhow::Result<UvRuntimeSource> {
        Ok(UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new("/venv".to_owned())?,
        })
    }

    fn uv_vllm(build: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.21.0".to_owned())?,
            build: uv_build(build)?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: uv_source()?,
        }))
    }

    fn uv_sglang(build: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvSglang(UvSglang {
            server_version: UvServerVersion::try_new("0.5.21".to_owned())?,
            build: uv_build(build)?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: uv_source()?,
        }))
    }

    /// A uv build's flavor is its tag's.
    #[rstest]
    #[case::vllm(vllm(VllmFlavor::Cpu))]
    #[case::sglang(sglang(SglangFlavor::Cpu))]
    #[case::uv_vllm(uv_vllm("cpu"))]
    #[case::uv_sglang(uv_sglang("cpu"))]
    fn a_cpu_flavor_reads_no_gpu(#[case] runtime: anyhow::Result<Runtime>) -> anyhow::Result<()> {
        assert!(matches!(memory_watch(&runtime?)?, cgroup::Watch::Peak));
        Ok(())
    }

    /// Off Linux, or with no AMD card this job has alone, the cell fails
    /// before its server starts rather than record the host alone.
    #[rstest]
    #[case::vllm(vllm(VllmFlavor::AmdGpu))]
    #[case::sglang(sglang(SglangFlavor::AmdGpu))]
    #[case::uv_rocm(uv_vllm("rocm6"))]
    fn an_amd_flavor_reads_its_cards_or_is_refused(
        #[case] runtime: anyhow::Result<Runtime>,
    ) -> anyhow::Result<()> {
        match memory_watch(&runtime?) {
            Ok(cgroup::Watch::Observe | cgroup::Watch::Peak) => {
                anyhow::bail!("an AMD flavor read no GPU")
            }
            #[cfg(target_os = "linux")]
            Ok(cgroup::Watch::PeakWithAmdGpu(_)) => {}
            #[cfg(target_os = "linux")]
            Ok(cgroup::Watch::PeakWithNvidiaGpu(_)) => anyhow::bail!("an AMD flavor read NVML"),
            Err(err) => assert!(format!("{err:#}").contains("AMD GPU"), "{err:#}"),
        }
        Ok(())
    }

    /// Off Linux, or where NVML does not open, the cell fails before its
    /// server starts rather than record the host alone.
    #[rstest]
    #[case::vllm(vllm(VllmFlavor::NvidiaGpu))]
    #[case::sglang(sglang(SglangFlavor::NvidiaGpu))]
    #[case::uv_cuda(uv_vllm("cu129"))]
    fn an_nvidia_flavor_reads_its_gpu_or_is_refused(
        #[case] runtime: anyhow::Result<Runtime>,
    ) -> anyhow::Result<()> {
        match memory_watch(&runtime?) {
            Ok(cgroup::Watch::Observe | cgroup::Watch::Peak) => {
                anyhow::bail!("an NVIDIA flavor read no GPU")
            }
            #[cfg(target_os = "linux")]
            Ok(cgroup::Watch::PeakWithNvidiaGpu(_)) => {}
            #[cfg(target_os = "linux")]
            Ok(cgroup::Watch::PeakWithAmdGpu(_)) => anyhow::bail!("an NVIDIA flavor read AMD"),
            Err(err) => assert!(format!("{err:#}").contains("NVML"), "{err:#}"),
        }
        Ok(())
    }

    /// A part names the container when it is its systemd scope or its id, and
    /// the container's cgroup is the path through it; another container, a
    /// longer name or a short id names none.
    #[rstest]
    #[case::systemd(
        format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope\n"),
        Some(format!("/system.slice/docker-{FAKE_CONTAINER_ID}.scope"))
    )]
    #[case::below_its_scope(
        format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope/init\n"),
        Some(format!("/system.slice/docker-{FAKE_CONTAINER_ID}.scope"))
    )]
    #[case::cgroupfs(format!("0::/docker/{FAKE_CONTAINER_ID}\n"), Some(format!("/docker/{FAKE_CONTAINER_ID}")))]
    #[case::under_another_parent(
        format!("0::/ci.slice/runners/{FAKE_CONTAINER_ID}/workers\n"),
        Some(format!("/ci.slice/runners/{FAKE_CONTAINER_ID}"))
    )]
    #[case::cgroup_v1(
        format!("12:memory:/docker/{FAKE_CONTAINER_ID}\n11:cpu,cpuacct:/docker/{FAKE_CONTAINER_ID}\n0::/\n"),
        None
    )]
    #[case::another_container(format!("0::/system.slice/docker-{OTHER_ID}.scope\n"), None)]
    #[case::a_longer_name(format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope-1\n"), None)]
    #[case::a_prefixed_id(format!("0::/system.slice/podman-{FAKE_CONTAINER_ID}.scope\n"), None)]
    #[case::a_short_id("0::/system.slice/docker-3f7c9a1b2d4e.scope\n".to_owned(), None)]
    #[case::a_host_process("0::/user.slice/user-1001.slice/session-7.scope\n".to_owned(), None)]
    fn a_process_s_cgroups_give_the_container_s(
        #[case] cgroups: String,
        #[case] expected: Option<String>,
    ) -> anyhow::Result<()> {
        let id = ContainerId::parse(FAKE_CONTAINER_ID).context("not a container id")?;
        assert_eq!(
            Cgroup::named(&cgroups, |part| id.names(part))
                .as_ref()
                .map(Cgroup::path),
            expected.as_deref()
        );
        Ok(())
    }

    #[rstest]
    #[case::started(format!("{FAKE_CONTAINER_ID} 4242"), Some(4242))]
    #[case::created(format!("{FAKE_CONTAINER_ID} 0"), Some(0))]
    #[case::no_pid(FAKE_CONTAINER_ID.to_owned(), None)]
    #[case::not_a_pid(format!("{FAKE_CONTAINER_ID} x"), None)]
    #[case::a_short_id("0e542162bd21 4242".to_owned(), None)]
    #[case::a_name("pipette-bench-1a2b3c4d 4242".to_owned(), None)]
    #[case::nothing(String::new(), None)]
    fn inspect_prints_the_full_id_and_pid(#[case] printed: String, #[case] pid: Option<u32>) {
        assert_eq!(Container::parse(&printed).map(|found| found.pid), pid);
    }

    /// `docker container inspect` names the container and its process, or
    /// says why not: a stalled daemon is cut off at the limit, so the probe's
    /// thread, and the drop that joins it, cannot wait on it for good.
    #[cfg(unix)]
    #[rstest]
    #[case::started(None, Ok(()))]
    #[case::not_yet(Some("no-container"), Err("No such container: pipette-bench-test"))]
    #[case::not_started(Some("not-started"), Err("it has not started"))]
    #[case::stalled(Some("hang"), Err("did not exit within"))]
    fn docker_names_the_container(
        #[case] state: Option<&str>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        use crate::server::test_support::fake_docker;

        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), "")?;
        if let Some(state) = state {
            std::fs::write(dir.path().join(state), "")?;
        }
        // A short limit for the stall alone: a loaded Mac has taken over 2 s
        // to run the stand-in.
        let limit = Duration::from_secs(if state == Some("hang") { 2 } else { 30 });
        let started = std::time::Instant::now();
        let named = inspect(&docker, "pipette-bench-test", limit);
        assert!(
            started.elapsed() < limit.saturating_add(Duration::from_secs(2)),
            "{:?}",
            started.elapsed()
        );
        match (named, expected) {
            (Ok(found), Ok(())) => assert_eq!(
                found,
                Container {
                    id: ContainerId::parse(FAKE_CONTAINER_ID).context("not a container id")?,
                    pid: FAKE_PID,
                }
            ),
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// The container [`fake_host`](crate::server::test_support::fake_host)
    /// places [`FAKE_PID`] in.
    fn container() -> anyhow::Result<Container> {
        Ok(Container {
            id: ContainerId::parse(FAKE_CONTAINER_ID).context("not a container id")?,
            pid: FAKE_PID,
        })
    }

    /// Writes `cgroups` as `/proc/<pid>/cgroup` under `dir`, and returns the
    /// file.
    fn place(dir: &Path, pid: u32, cgroups: &str) -> anyhow::Result<PathBuf> {
        let pid = dir.join("proc").join(pid.to_string());
        std::fs::create_dir_all(&pid)?;
        std::fs::write(pid.join("cgroup"), cgroups)?;
        Ok(pid.join("cgroup"))
    }

    /// The container's cgroup is the one its main process's names, under
    /// either driver; a host whose `/proc` or cgroups do not place it there is
    /// refused, naming the file read.
    #[rstest]
    #[case::systemd(
        Some(format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope\n")),
        Ok(format!("/system.slice/docker-{FAKE_CONTAINER_ID}.scope"))
    )]
    #[case::cgroupfs(Some(format!("0::/docker/{FAKE_CONTAINER_ID}\n")), Ok(format!("/docker/{FAKE_CONTAINER_ID}")))]
    #[case::below_its_scope(
        Some(format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope/init\n")),
        Ok(format!("/system.slice/docker-{FAKE_CONTAINER_ID}.scope"))
    )]
    #[case::cgroup_v1(
        Some(format!("12:memory:/docker/{FAKE_CONTAINER_ID}\n0::/\n")),
        Err("names no cgroup v2 path of container 3f7c9a1b")
    )]
    #[case::not_in_this_proc(None, Err("could not read "))]
    fn the_container_s_cgroup_is_its_main_process_s(
        #[case] cgroups: Option<String>,
        #[case] expected: Result<String, &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir
            .path()
            .join("proc")
            .join(FAKE_PID.to_string())
            .join("cgroup");
        if let Some(cgroups) = cgroups {
            place(dir.path(), FAKE_PID, &cgroups)?;
        }
        match (
            container_cgroup(&fake_tree(dir.path())?, &container()?),
            expected,
        ) {
            (Ok(found), Ok(want)) => assert_eq!(found.path(), want),
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
                assert!(message.contains(&file.display().to_string()), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// A container `docker` has not named yet is asked again; one whose main
    /// process sits in no cgroup of it never will be, so the probe stops at
    /// once.
    #[cfg(unix)]
    #[rstest]
    #[case::not_named_yet(Some("no-container"), None, false)]
    #[case::placed_elsewhere(None, Some("0::/user.slice/session-7.scope\n"), true)]
    fn a_container_is_asked_for_until_docker_names_it(
        #[case] state: Option<&str>,
        #[case] cgroups: Option<&str>,
        #[case] failed: bool,
    ) -> anyhow::Result<()> {
        use crate::server::test_support::fake_docker;

        let dir = tempfile::tempdir()?;
        let target = ServerCgroup::Container {
            docker: fake_docker(dir.path(), "")?,
            name: "pipette-bench-test".to_owned(),
        };
        if let Some(state) = state {
            std::fs::write(dir.path().join(state), "")?;
        }
        if let Some(cgroups) = cgroups {
            place(dir.path(), FAKE_PID, cgroups)?;
        }
        match target.locate(&fake_tree(dir.path())?) {
            Err(Unlocated::Failed(LocateError::Elsewhere { .. })) if failed => {}
            Err(Unlocated::Yet(LocateError::Uninspected(_))) if !failed => {}
            other => anyhow::bail!("unexpected {other:?}"),
        }
        Ok(())
    }

    /// A uv server's cgroup is pipette's own, read through
    /// `/proc/self/cgroup`, where its session and what that starts
    /// count; one that counts no memory of its own fails the probe at once.
    #[rstest]
    #[case::a_login_session(true, Ok(FAKE_SESSION))]
    #[case::the_host_root(false, Err("counts no memory of its own"))]
    fn a_uv_server_s_cgroup_is_pipette_s_own(
        #[case] counted: bool,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let target = ServerCgroup::Shared { pid: 77 };
        assert_eq!(target.members(), Members::Session { leader: 77 });
        match (target.locate(&fake_session(dir.path(), counted)?), expected) {
            (Ok(found), Ok(want)) => assert_eq!(found.path(), want),
            (Err(Unlocated::Failed(err)), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }
}
