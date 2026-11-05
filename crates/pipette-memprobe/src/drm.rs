//! A DRM GPU's memory per process, from the fdinfo the kernel's DRM drivers
//! print for each open of a device (`Documentation/gpu/drm-usage-stats.rst`):
//! amdgpu since Linux 5.14, i915 and xe since 6.8. NVIDIA's driver prints none,
//! so an fd on one of its devices is only noted; NVML reads them
//! ([`crate::nvml`]).
//!
//! Each open of a device is a client, named by `drm-pdev` and
//! `drm-client-id`, and counted once however many fds refer to it. Its figure
//! is the sum over regions of the first of three key families it prints:
//! `drm-resident-<region>`, the pages its buffers hold; `drm-memory-<region>`,
//! its deprecated alias, which amdgpu prints in its place before 6.13 and
//! beside it since; `drm-total-<region>`, on a driver that prints neither. The
//! families are never added, as the alias repeats the resident figure. On i915
//! a client's pages are outside the process's resident set, so the host peak
//! misses them.
//!
//! amdgpu from 5.14 to 5.18 prints no `drm-` keys: `pdev` and `pasid`, the id
//! of the open's address space (under ROCm its process's, on every GPU), name
//! the client, and `vram mem`, `gtt mem` and `cpu mem` are the sums 5.19 prints
//! as `drm-memory-<region>`. That layout is read only from an fd that names no
//! `drm-client-id`, as later kernels still print `pasid`.

use std::collections::HashMap;
use std::path::Path;

/// Where each DRM node names its device, and the device its driver.
const SYSFS_DRM: &str = "/sys/class/drm";

/// A DRM fd that names no client, prints none of the memory keys, or prints a
/// size it cannot read: i915 before Linux 6.8 prints no memory, amdgpu before
/// 5.14 prints nothing.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Unread;

/// A client, `(device, client id or pasid)`, and the bytes its buffers hold.
type Client = ((String, u64), u64);

/// What a process's open fds say of its GPUs at one read.
pub(crate) struct Fds {
    /// What the DRM drivers bill it, summed over its clients: 0 while it has
    /// none.
    pub(crate) drm: Result<u64, Unread>,
    /// It holds an NVIDIA device open.
    pub(crate) on_nvidia: bool,
}

/// `pid`'s fds now; `None` when they cannot be listed.
pub(crate) fn read(pid: u32) -> Option<Fds> {
    let devices: Vec<_> = std::fs::read_dir(format!("/proc/{pid}/fd"))
        .ok()?
        .filter_map(|fd| {
            let fd = fd.ok()?;
            let target = std::fs::read_link(fd.path()).ok()?;
            Some((
                fd.file_name(),
                device(Path::new(SYSFS_DRM), target.to_str()?)?,
            ))
        })
        .collect();
    let drm = total(
        devices
            .iter()
            .filter(|(_, device)| *device == Device::Drm)
            .filter_map(|(fd, _)| {
                // An fd closed since the listing is skipped, not unread.
                let fdinfo = format!("/proc/{pid}/fdinfo/{}", fd.to_str()?);
                Some(client(&std::fs::read_to_string(fdinfo).ok()?).ok_or(Unread))
            }),
    );
    Some(Fds {
        drm,
        on_nvidia: devices.iter().any(|(_, device)| *device == Device::Nvidia),
    })
}

/// The clients' bytes, each client once; unread when any fd is.
fn total(clients: impl IntoIterator<Item = Result<Client, Unread>>) -> Result<u64, Unread> {
    let clients: HashMap<_, _> = clients.into_iter().collect::<Result<_, _>>()?;
    Ok(clients
        .values()
        .fold(0u64, |sum, &bytes| sum.saturating_add(bytes)))
}

/// A GPU an fd can hold open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Device {
    /// A DRM node whose driver prints its clients' memory.
    Drm,
    /// One of NVIDIA's: `/dev/nvidia*`, or a DRM node its driver backs.
    Nvidia,
}

/// The GPU at `path`, or `None` for an fd on none. A DRM node sysfs names no
/// driver for is read: if it prints no memory, the read fails as unread rather
/// than drop it from the count.
fn device(sysfs: &Path, path: &str) -> Option<Device> {
    if path.starts_with("/dev/nvidia") {
        return Some(Device::Nvidia);
    }
    let node = path.strip_prefix("/dev/dri/")?;
    let driver = std::fs::read_link(sysfs.join(node).join("device/driver")).ok();
    Some(
        if driver.is_some_and(|driver| driver.file_name() == Some("nvidia".as_ref())) {
            Device::Nvidia
        } else {
            Device::Drm
        },
    )
}

/// The key families that bill a client's memory per region: the first an
/// fdinfo prints is its figure.
const MEMORY: [&str; 3] = ["drm-resident-", "drm-memory-", "drm-total-"];

/// amdgpu's memory lines from Linux 5.14 to 5.18, each in `kB` of 1024 bytes.
const AMDGPU_5_14_MEMORY: [&str; 3] = ["vram mem:", "gtt mem:", "cpu mem:"];

/// A DRM fd's client; `None` when the fdinfo names none, prints no memory, or
/// prints a size it cannot read.
fn client(fdinfo: &str) -> Option<Client> {
    let field = |name: &str| {
        fdinfo
            .lines()
            .find_map(|line| Some(line.strip_prefix(name)?.trim()))
    };
    match field("drm-client-id:") {
        Some(id) => Some((
            (
                field("drm-pdev:").unwrap_or_default().to_owned(),
                id.parse().ok()?,
            ),
            drm_bytes(fdinfo)?,
        )),
        None => Some((
            (field("pdev:")?.to_owned(), field("pasid:")?.parse().ok()?),
            AMDGPU_5_14_MEMORY.into_iter().try_fold(0u64, |sum, key| {
                let kib: u64 = field(key)?.strip_suffix(" kB")?.parse().ok()?;
                Some(sum.saturating_add(kib.saturating_mul(1 << 10)))
            })?,
        )),
    }
}

/// The bytes of the first [`MEMORY`] family an fdinfo prints.
fn drm_bytes(fdinfo: &str) -> Option<u64> {
    let family = |prefix: &str| {
        fdinfo
            .lines()
            .filter_map(|line| line.strip_prefix(prefix)?.split_once(':'))
            // `drm-total-cycles-<engine>` counts time, not memory.
            .filter(|(region, _)| !region.starts_with("cycles-"))
            .map(|(_, size)| size)
            .collect::<Vec<_>>()
    };
    MEMORY
        .into_iter()
        .map(family)
        .find(|sizes| !sizes.is_empty())?
        .into_iter()
        .try_fold(0u64, |sum, size| Some(sum.saturating_add(bytes(size)?)))
}

/// A size as fdinfo prints it: a count, in bytes or with a `KiB` or `MiB`
/// unit.
fn bytes(size: &str) -> Option<u64> {
    let mut parts = size.split_whitespace();
    let count: u64 = parts.next()?.parse().ok()?;
    let unit = match parts.next() {
        None => 1,
        Some("KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some(_) => return None,
    };
    Some(count.saturating_mul(unit))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An OpenVINO server's fd on an Iris Xe (i915, Linux 7.0), abridged.
    const I915: &str = "\
pos:\t0
flags:\t02100002
drm-driver:\ti915
drm-client-id:\t2
drm-pdev:\t0000:00:02.0
drm-total-system0:\t517176 KiB
drm-shared-system0:\t0
drm-active-system0:\t0
drm-resident-system0:\t516984 KiB
drm-purgeable-system0:\t1156 KiB
drm-total-stolen-system0:\t0
drm-resident-stolen-system0:\t0
drm-engine-render:\t246835160 ns
";

    /// amdgpu's fdinfo as Linux 5.14 prints it (`amdgpu_fdinfo.c`), with a
    /// usage line per busy ring; the figures are made up.
    const AMDGPU_5_14: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t25
ino:\t1083
pdev:\t0000:03:00.0
pasid:\t32774
vram mem:\t2109440 kB
gtt mem:\t12288 kB
cpu mem:\t4 kB
gfx0:\t12.34%
compute0:\t0.5%
dma0:\t1.20%
";

    const AMDGPU_5_14_BYTES: u64 = (2_109_440 + 12_288 + 4) << 10;

    /// The same layout as Linux 5.18 prints it.
    const AMDGPU_5_18: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t26
ino:\t1077
pdev:\t0000:0b:00.0
pasid:\t32769
vram mem:\t7340032 kB
gtt mem:\t1048576 kB
cpu mem:\t0 kB
gfx0:\t97.80%
dec0:\t0.0%
jpeg0:\t0.3%
";

    /// amdgpu's fdinfo as Linux 5.19 prints it (`amdgpu_fdinfo.c`), the first
    /// with `drm-` keys; the sizes are made up.
    const AMDGPU_5_19: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t25
ino:\t1083
pasid:\t32774
drm-driver:\tamdgpu
drm-pdev:\t0000:03:00.0
drm-client-id:\t41
drm-memory-vram:\t2109440 KiB
drm-memory-gtt: \t12288 KiB
drm-memory-cpu: \t0 KiB
drm-engine-gfx:\t28562171 ns
drm-engine-compute:\t9123456 ns
";

    /// amdgpu's fdinfo as Ubuntu's 6.8 kernels print it, the format of
    /// Linux 6.9 to 6.12: the core prints the client, amdgpu the rest.
    const AMDGPU_6_8: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t25
ino:\t1083
drm-driver:\tamdgpu
drm-client-id:\t17
drm-pdev:\t0000:c3:00.0
pasid:\t32771
drm-memory-vram:\t524288 KiB
drm-memory-gtt: \t65536 KiB
drm-memory-cpu: \t2048 KiB
amd-memory-visible-vram:\t262144 KiB
amd-evicted-vram:\t0 KiB
amd-evicted-visible-vram:\t0 KiB
amd-requested-vram:\t524288 KiB
amd-requested-visible-vram:\t262144 KiB
amd-requested-gtt:\t65536 KiB
drm-shared-vram:\t0 KiB
drm-shared-gtt:\t4096 KiB
drm-shared-cpu:\t0 KiB
drm-engine-gfx:\t1458734021 ns
drm-engine-compute:\t318822411 ns
";

    /// The same client as Linux 6.13 prints it: the core's memory keys per
    /// region (`drm_print_memory_stats`, sizes in the largest unit that
    /// divides them), then amdgpu's legacy alias of the resident ones.
    const AMDGPU_6_13: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t25
ino:\t1083
drm-driver:\tamdgpu
drm-client-id:\t17
drm-pdev:\t0000:c3:00.0
pasid:\t32771
drm-total-vram:\t520 MiB
drm-shared-vram:\t0
drm-active-vram:\t0
drm-resident-vram:\t512 MiB
drm-purgeable-vram:\t0
drm-total-gtt:\t64 MiB
drm-shared-gtt:\t4 MiB
drm-active-gtt:\t0
drm-resident-gtt:\t64 MiB
drm-purgeable-gtt:\t0
drm-total-cpu:\t2 MiB
drm-shared-cpu:\t0
drm-active-cpu:\t0
drm-resident-cpu:\t2 MiB
drm-purgeable-cpu:\t0
drm-total-gds:\t0
drm-shared-gds:\t0
drm-active-gds:\t0
drm-resident-gds:\t0
drm-purgeable-gds:\t0
drm-total-gws:\t0
drm-shared-gws:\t0
drm-active-gws:\t0
drm-resident-gws:\t0
drm-purgeable-gws:\t0
drm-total-oa:\t0
drm-shared-oa:\t0
drm-active-oa:\t0
drm-resident-oa:\t0
drm-purgeable-oa:\t0
drm-total-doorbell:\t0
drm-shared-doorbell:\t0
drm-active-doorbell:\t0
drm-resident-doorbell:\t0
drm-purgeable-doorbell:\t0
drm-memory-vram:\t524288 KiB
drm-memory-gtt: \t65536 KiB
drm-memory-cpu: \t2048 KiB
amd-evicted-vram:\t0 KiB
amd-requested-vram:\t532480 KiB
amd-requested-gtt:\t65536 KiB
drm-engine-gfx:\t1458734021 ns
drm-engine-compute:\t318822411 ns
";

    /// xe's fdinfo on an integrated GPU as Linux 6.12 prints it
    /// (`xe_drm_client.c`); the sizes are made up.
    const XE: &str = "\
pos:\t0
flags:\t02100002
mnt_id:\t25
ino:\t1083
drm-driver:\txe
drm-client-id:\t4
drm-pdev:\t0000:00:02.0
drm-total-system:\t412 MiB
drm-shared-system:\t0
drm-active-system:\t0
drm-resident-system:\t412 MiB
drm-purgeable-system:\t0
drm-total-gtt:\t4 MiB
drm-shared-gtt:\t0
drm-active-gtt:\t0
drm-resident-gtt:\t4 MiB
drm-total-stolen:\t0
drm-shared-stolen:\t0
drm-active-stolen:\t0
drm-resident-stolen:\t0
drm-cycles-rcs:\t28257900
drm-total-cycles-rcs:\t7655183225
drm-cycles-vcs:\t0
drm-total-cycles-vcs:\t7655183225
drm-engine-capacity-vcs:\t2
drm-cycles-vecs:\t0
drm-total-cycles-vecs:\t7655183225
drm-cycles-bcs:\t0
drm-total-cycles-bcs:\t7655183225
drm-cycles-ccs:\t69012345
drm-total-cycles-ccs:\t7655183225
";

    #[rstest::rstest]
    #[case::resident_over_total(I915, Some((("0000:00:02.0".to_owned(), 2), 516_984 << 10)))]
    #[case::xe(XE, Some((("0000:00:02.0".to_owned(), 4), (412 + 4) << 20)))]
    #[case::amdgpu_5_19_prints_the_alias_alone(
        AMDGPU_5_19,
        Some((("0000:03:00.0".to_owned(), 41), (2_109_440 + 12_288) << 10))
    )]
    #[case::amdgpu_6_8_prints_the_alias_alone(
        AMDGPU_6_8,
        Some((("0000:c3:00.0".to_owned(), 17), (524_288 + 65_536 + 2_048) << 10))
    )]
    #[case::amdgpu_6_13_prints_both_and_counts_once(
        AMDGPU_6_13,
        Some((("0000:c3:00.0".to_owned(), 17), (512 + 64 + 2) << 20))
    )]
    #[case::resident_over_its_alias(
        "drm-client-id:\t5\ndrm-resident-vram:\t8 KiB\ndrm-resident-doorbell:\t4 KiB\n\
         drm-memory-vram:\t8 KiB\n",
        Some(((String::new(), 5), 12 << 10))
    )]
    #[case::alias_over_total(
        "drm-client-id:\t5\ndrm-total-vram:\t16 KiB\ndrm-memory-vram:\t8 KiB\n",
        Some(((String::new(), 5), 8 << 10))
    )]
    #[case::amdgpu_5_14_names_a_client_by_pasid(
        AMDGPU_5_14,
        Some((("0000:03:00.0".to_owned(), 32_774), AMDGPU_5_14_BYTES))
    )]
    #[case::amdgpu_5_18_prints_the_same_layout(
        AMDGPU_5_18,
        Some((("0000:0b:00.0".to_owned(), 32_769), (7_340_032 + 1_048_576) << 10))
    )]
    #[case::a_drm_client_id_never_reads_the_5_14_layout(
        "drm-pdev:\t0000:03:00.0\ndrm-client-id:\t41\ndrm-memory-vram:\t8 KiB\n\
         pdev:\t0000:03:00.0\npasid:\t32774\nvram mem:\t8 kB\ngtt mem:\t4 kB\ncpu mem:\t0 kB\n",
        Some((("0000:03:00.0".to_owned(), 41), 8 << 10))
    )]
    #[case::a_drm_client_id_without_drm_memory_is_unread(
        "drm-client-id:\t41\npdev:\t0000:03:00.0\npasid:\t32774\nvram mem:\t8 kB\n\
         gtt mem:\t4 kB\ncpu mem:\t0 kB\n",
        None
    )]
    #[case::amdgpu_5_14_missing_a_line(
        "pdev:\t0000:03:00.0\npasid:\t32774\nvram mem:\t8 kB\ngtt mem:\t4 kB\n",
        None
    )]
    #[case::amdgpu_5_14_in_another_unit(
        "pdev:\t0000:03:00.0\npasid:\t32774\nvram mem:\t8 KiB\ngtt mem:\t4 kB\ncpu mem:\t0 kB\n",
        None
    )]
    #[case::amdgpu_5_14_without_a_pasid(
        "pdev:\t0000:03:00.0\nvram mem:\t8 kB\ngtt mem:\t4 kB\ncpu mem:\t0 kB\n",
        None
    )]
    #[case::amdgpu_5_14_without_a_device(
        "pasid:\t32774\nvram mem:\t8 kB\ngtt mem:\t4 kB\ncpu mem:\t0 kB\n",
        None
    )]
    #[case::only_keys_that_are_not_a_figure(
        "drm-driver:\tamdgpu\ndrm-client-id:\t17\namd-memory-visible-vram:\t262144 KiB\n\
         amd-requested-vram:\t524288 KiB\ndrm-shared-vram:\t0 KiB\ndrm-engine-gfx:\t5 ns\n\
         drm-total-cycles-gfx:\t7655183225\n",
        None
    )]
    #[case::regions_add_up(
        "drm-client-id:\t7\ndrm-resident-system0:\t3 MiB\ndrm-resident-local0:\t4096\n",
        Some(((String::new(), 7), (3 << 20) + 4096))
    )]
    #[case::total_without_resident(
        "drm-pdev:\t0000:03:00.0\ndrm-client-id:\t9\ndrm-total-vram0:\t8 KiB\n",
        Some((("0000:03:00.0".to_owned(), 9), 8 << 10))
    )]
    #[case::engine_cycles_are_not_memory(
        "drm-client-id:\t3\ndrm-total-memory:\t192 KiB\ndrm-total-cycles-rcs:\t7655183225\n",
        Some(((String::new(), 3), 192 << 10))
    )]
    #[case::no_memory_printed(
        "drm-driver:\ti915\ndrm-client-id:\t2\ndrm-engine-render:\t5 ns\n",
        None
    )]
    #[case::no_client("pos:\t0\nflags:\t02100002\n", None)]
    #[case::an_unknown_unit("drm-client-id:\t2\ndrm-resident-system0:\t5 GiB\n", None)]
    #[case::an_unknown_unit_is_not_passed_over(
        "drm-client-id:\t2\ndrm-resident-system0:\t5 GiB\ndrm-total-system0:\t5 MiB\n",
        None
    )]
    fn a_drm_fd_gives_its_client_and_bytes(#[case] fdinfo: &str, #[case] expected: Option<Client>) {
        assert_eq!(client(fdinfo), expected);
    }

    #[test]
    fn a_device_is_named_by_its_driver() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let sysfs = root.path();
        let drivers = sysfs.join("bus/pci/drivers");
        for (node, driver) in [("renderD128", "i915"), ("renderD129", "nvidia")] {
            std::fs::create_dir_all(drivers.join(driver))?;
            std::fs::create_dir_all(sysfs.join(node).join("device"))?;
            std::os::unix::fs::symlink(
                drivers.join(driver),
                sysfs.join(node).join("device/driver"),
            )?;
        }
        let device = |path| device(sysfs, path);
        assert_eq!(device("/dev/dri/renderD128"), Some(Device::Drm));
        assert_eq!(device("/dev/dri/renderD129"), Some(Device::Nvidia));
        assert_eq!(
            device("/dev/dri/card7"),
            Some(Device::Drm),
            "a node sysfs cannot name is read"
        );
        assert_eq!(device("/dev/nvidiactl"), Some(Device::Nvidia));
        assert_eq!(device("/dev/null"), None);
        Ok(())
    }

    fn on(pdev: &str, id: u64, bytes: u64) -> Result<Client, Unread> {
        Ok(((pdev.to_owned(), id), bytes))
    }

    #[rstest::rstest]
    #[case::no_client(vec![], Ok(0))]
    #[case::two_fds_on_one_client_count_once(vec![on("a", 2, 100), on("a", 2, 100)], Ok(100))]
    #[case::two_clients_add_up(vec![on("a", 2, 100), on("a", 3, 5)], Ok(105))]
    #[case::one_id_on_two_devices_is_two_clients(vec![on("a", 2, 100), on("b", 2, 5)], Ok(105))]
    #[case::one_unread_fd_fails_the_read(vec![on("a", 2, 100), Err(Unread)], Err(Unread))]
    fn a_read_counts_each_client_once(
        #[case] clients: Vec<Result<Client, Unread>>,
        #[case] expected: Result<u64, Unread>,
    ) {
        assert_eq!(total(clients), expected);
    }

    /// [`AMDGPU_5_14`] read beside another fd's fdinfo.
    #[rstest::rstest]
    #[case::a_dup_without_close_on_exec_is_one_client(
        AMDGPU_5_14.replace("flags:\t02100002", "flags:\t0100002"),
        AMDGPU_5_14_BYTES
    )]
    #[case::two_opens_add_up(
        AMDGPU_5_14.replace("pasid:\t32774", "pasid:\t32775"),
        2 * AMDGPU_5_14_BYTES
    )]
    // ROCm gives a process one pasid on every GPU it uses.
    #[case::one_pasid_on_two_devices_is_two_clients(
        AMDGPU_5_14.replace("pdev:\t0000:03:00.0", "pdev:\t0000:43:00.0"),
        2 * AMDGPU_5_14_BYTES
    )]
    fn amdgpu_5_14_counts_each_open_once(#[case] other: String, #[case] expected: u64) {
        assert_ne!(other, AMDGPU_5_14);
        let clients = [AMDGPU_5_14, other.as_str()].map(|fdinfo| client(fdinfo).ok_or(Unread));
        assert_eq!(total(clients), Ok(expected));
    }
}
