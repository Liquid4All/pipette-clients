//! DXGI adapter enumeration and selection, including the Vulkan LUID match that
//! picks the same GPU llama.cpp's Vulkan backend uses.

use std::ffi::CStr;
use std::fmt;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use known_folders::{KnownFolder, get_known_folder_path};
use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
    IDXGIAdapter1, IDXGIFactory1,
};
use windows::core::PCWSTR;

/// The Vulkan loader can wedge, so enumeration is bounded.
pub(crate) const VULKAN_TIMEOUT: Duration = Duration::from_secs(60);

/// A DXGI LUID, displayed high:low in hex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Luid(pub LUID);

impl From<[u8; 8]> for Luid {
    /// Vulkan reports a LUID as the eight little-endian bytes of the Windows one.
    fn from(bytes: [u8; 8]) -> Self {
        Self(LUID {
            LowPart: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            HighPart: i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
    }
}

impl fmt::Display for Luid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:x}:{:x}",
            self.0.HighPart as u32, self.0.LowPart
        )
    }
}

pub struct Adapter {
    pub index: u32,
    pub name: String,
    pub luid: Luid,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_bytes: usize,
    pub software: bool,
    pub handle: IDXGIAdapter1,
}

/// How to pick the adapter to pulse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdapterChoice {
    /// The single GPU Vulkan reports, matched to DXGI by LUID. The default,
    /// because indices and LUIDs both move across reboots.
    #[default]
    Vulkan,
    /// The Microsoft Basic Render Driver. Keeps no real GPU awake; for CI.
    Software,
    /// A fixed DXGI adapter index.
    Index(u32),
}

impl fmt::Display for AdapterChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => formatter.write_str("vulkan"),
            Self::Software => formatter.write_str("software"),
            Self::Index(index) => write!(formatter, "{index}"),
        }
    }
}

impl std::str::FromStr for AdapterChoice {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "vulkan" => Ok(Self::Vulkan),
            "software" => Ok(Self::Software),
            other => other.parse::<u32>().map(Self::Index).map_err(|_| {
                format!("expected vulkan, software or an adapter index, got '{other}'")
            }),
        }
    }
}

pub struct VulkanGpu {
    pub name: String,
    pub luid: Option<Luid>,
    pub cpu: bool,
}

pub fn enumerate_adapters() -> anyhow::Result<Vec<Adapter>> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
    let mut adapters = Vec::new();
    for index in 0.. {
        let handle: IDXGIAdapter1 = match unsafe { factory.EnumAdapters1(index) } {
            Ok(handle) => handle,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(error.into()),
        };
        let description = unsafe { handle.GetDesc1() }?;
        // DXGI always NUL-terminates Description.
        let name = unsafe { PCWSTR::from_raw(description.Description.as_ptr()).to_string() }?;
        adapters.push(Adapter {
            index,
            name,
            luid: Luid(description.AdapterLuid),
            vendor_id: description.VendorId,
            device_id: description.DeviceId,
            dedicated_bytes: description.DedicatedVideoMemory,
            software: DXGI_ADAPTER_FLAG(description.Flags as i32) & DXGI_ADAPTER_FLAG_SOFTWARE
                != DXGI_ADAPTER_FLAG(0),
            handle,
        });
    }
    Ok(adapters)
}

/// Enumerates Vulkan physical devices in this process, bounded to 60 seconds: on
/// timeout the worker is abandoned and dies with the process.
pub fn enumerate_vulkan() -> anyhow::Result<Vec<VulkanGpu>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(enumerate_vulkan_now());
    });
    match receiver.recv_timeout(VULKAN_TIMEOUT) {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!(
            "Vulkan enumeration timed out after {} seconds",
            VULKAN_TIMEOUT.as_secs()
        )),
    }
}

fn vulkan_loader_path() -> anyhow::Result<PathBuf> {
    // System32 by absolute path: this runs as SYSTEM, so never search the
    // working directory for a DLL.
    let directory = get_known_folder_path(KnownFolder::System)
        .ok_or_else(|| anyhow::anyhow!("Locating the system directory failed"))?;
    Ok(directory.join("vulkan-1.dll"))
}

fn enumerate_vulkan_now() -> anyhow::Result<Vec<VulkanGpu>> {
    let path = vulkan_loader_path()?;
    let entry = unsafe { ash::Entry::load_from(&path) }
        .map_err(|error| anyhow::anyhow!("Loading {} failed: {error}", path.display()))?;
    let version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(|error| anyhow::anyhow!("vkEnumerateInstanceVersion failed: {error}"))?
        .unwrap_or(ash::vk::API_VERSION_1_0);
    if version < ash::vk::API_VERSION_1_1 {
        return Err(anyhow::anyhow!(
            "The Vulkan loader predates Vulkan 1.1, which reports device LUIDs"
        ));
    }

    let application = ash::vk::ApplicationInfo::default().api_version(ash::vk::API_VERSION_1_1);
    let create_info = ash::vk::InstanceCreateInfo::default().application_info(&application);
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|error| anyhow::anyhow!("vkCreateInstance failed: {error}"))?;
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| anyhow::anyhow!("vkEnumeratePhysicalDevices failed: {error}"));
    let gpus = devices.map(|devices| {
        devices
            .into_iter()
            .map(|device| {
                let mut id = ash::vk::PhysicalDeviceIDProperties::default();
                // Scoped so the chained struct stops borrowing `id` before it is read.
                let (name, device_type) = {
                    let mut properties =
                        ash::vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                    unsafe { instance.get_physical_device_properties2(device, &mut properties) };
                    let name =
                        unsafe { CStr::from_ptr(properties.properties.device_name.as_ptr()) };
                    (
                        name.to_string_lossy().into_owned(),
                        properties.properties.device_type,
                    )
                };
                VulkanGpu {
                    name,
                    luid: (id.device_luid_valid == ash::vk::TRUE)
                        .then(|| Luid::from(id.device_luid)),
                    cpu: device_type == ash::vk::PhysicalDeviceType::CPU,
                }
            })
            .collect()
    });
    unsafe { instance.destroy_instance(None) };
    gpus
}

/// Resolves the choice against the adapters this machine reports.
pub fn select_adapter(choice: AdapterChoice) -> anyhow::Result<Adapter> {
    let mut adapters = enumerate_adapters()?;
    match choice {
        AdapterChoice::Index(index) => {
            let position = adapters
                .iter()
                .position(|adapter| adapter.index == index)
                .ok_or_else(|| anyhow::anyhow!("No adapter {index}; see gpu-keepalive list"))?;
            if adapters[position].software {
                return Err(anyhow::anyhow!(
                    "Adapter {index} is a software adapter; select it with 'software'"
                ));
            }
            Ok(adapters.swap_remove(position))
        }
        AdapterChoice::Software => {
            let position = adapters
                .iter()
                .position(|adapter| adapter.software)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No software adapter (Microsoft Basic Render Driver) is present"
                    )
                })?;
            Ok(adapters.swap_remove(position))
        }
        AdapterChoice::Vulkan => {
            let gpus: Vec<VulkanGpu> = enumerate_vulkan()?
                .into_iter()
                .filter(|gpu| !gpu.cpu)
                .collect();
            let [gpu] = &gpus[..] else {
                return Err(anyhow::anyhow!(
                    "Expected exactly one Vulkan GPU, found {}; pin a DXGI adapter index instead",
                    gpus.len()
                ));
            };
            let luid = gpu.luid.ok_or_else(|| {
                anyhow::anyhow!("The Vulkan GPU {} reports no valid LUID", gpu.name)
            })?;
            let position = adapters
                .iter()
                .position(|adapter| adapter.luid == luid)
                .ok_or_else(|| anyhow::anyhow!("No DXGI adapter matches Vulkan LUID {luid}"))?;
            if adapters[position].software {
                return Err(anyhow::anyhow!(
                    "The Vulkan GPU matches a software DXGI adapter"
                ));
            }
            Ok(adapters.swap_remove(position))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AdapterChoice, Luid};

    type Result = std::result::Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn adapter_choice_round_trips_through_its_text_form() -> Result {
        for text in ["vulkan", "software", "0", "3"] {
            let choice: AdapterChoice = text.parse()?;
            assert_eq!(choice.to_string(), text);
        }
        for text in ["gpu0", "", "-1", "Vulkan", "3.0", "auto"] {
            assert!(
                text.parse::<AdapterChoice>().is_err(),
                "{text} should not parse"
            );
        }
        Ok(())
    }

    #[test]
    fn luid_reads_vulkan_bytes_little_endian_and_prints_high_low() {
        let luid = Luid::from([0x78, 0x56, 0x34, 0x12, 0x21, 0x43, 0x65, 0x87]);
        assert_eq!(luid.0.LowPart, 0x1234_5678);
        assert_eq!(luid.0.HighPart, 0x8765_4321_u32 as i32);
        assert_eq!(luid.to_string(), "87654321:12345678");
    }
}
