//! The pulse itself: one compute thread increments a four-byte buffer per pulse.

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D11;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
use windows::Win32::Graphics::Dxgi::IDXGIAdapter1;
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
use windows::core::s;

const GPU_TIMEOUT: Duration = Duration::from_secs(5);

const SHADER: &[u8] = b"RWStructuredBuffer<uint> counter : register(u0);\n\
    [numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) { counter[0] += 1; }\n";

/// A stop request that both the console handler and SCM can raise, and that the
/// pulse loop waits on instead of sleeping.
#[derive(Clone, Default)]
pub struct Stop(Arc<(Mutex<bool>, Condvar)>);

impl Stop {
    pub fn signal(&self) {
        let (requested, waiters) = &*self.0;
        *requested.lock().expect("stop mutex") = true;
        waiters.notify_all();
    }

    pub fn is_set(&self) -> bool {
        *self.0.0.lock().expect("stop mutex")
    }

    /// Waits up to `timeout`, returning true when a stop was requested.
    pub fn wait(&self, timeout: Duration) -> bool {
        let (requested, waiters) = &*self.0;
        let guard = requested.lock().expect("stop mutex");
        if *guard {
            return true;
        }
        let (guard, _) = waiters.wait_timeout(guard, timeout).expect("stop mutex");
        *guard
    }
}

pub struct Pulse {
    device: Direct3D11::ID3D11Device,
    context: Direct3D11::ID3D11DeviceContext,
    buffer: Direct3D11::ID3D11Buffer,
    completed: Direct3D11::ID3D11Query,
    _view: Direct3D11::ID3D11UnorderedAccessView,
    _shader: Direct3D11::ID3D11ComputeShader,
}

impl Pulse {
    pub fn new(adapter: &IDXGIAdapter1) -> anyhow::Result<Self> {
        let mut device = None;
        let mut context = None;
        unsafe {
            Direct3D11::D3D11CreateDevice(
                adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                Direct3D11::D3D11_CREATE_DEVICE_SINGLETHREADED,
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                Direct3D11::D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .context("D3D11CreateDevice failed")?;
        let device = device.context("D3D11CreateDevice returned no device")?;
        let context = context.context("D3D11CreateDevice returned no context")?;

        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        let compiled = unsafe {
            D3DCompile(
                SHADER.as_ptr().cast(),
                SHADER.len(),
                s!("gpu-keepalive"),
                None,
                None,
                s!("main"),
                s!("cs_5_0"),
                D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(error) = compiled {
            let detail = errors.map(|blob| blob_text(&blob)).unwrap_or_default();
            return Err(anyhow::anyhow!("D3DCompile failed: {error} {detail}"));
        }
        let code = code.context("D3DCompile returned no bytecode")?;
        let bytecode = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        let mut shader = None;
        unsafe { device.CreateComputeShader(bytecode, None, Some(&mut shader)) }
            .context("CreateComputeShader failed")?;
        let shader = shader.context("CreateComputeShader returned nothing")?;

        let description = Direct3D11::D3D11_BUFFER_DESC {
            ByteWidth: size_of::<u32>() as u32,
            Usage: Direct3D11::D3D11_USAGE_DEFAULT,
            BindFlags: Direct3D11::D3D11_BIND_UNORDERED_ACCESS.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: Direct3D11::D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
            StructureByteStride: size_of::<u32>() as u32,
        };
        let zero: u32 = 0;
        let initial = Direct3D11::D3D11_SUBRESOURCE_DATA {
            pSysMem: std::ptr::addr_of!(zero).cast(),
            SysMemPitch: 0,
            SysMemSlicePitch: 0,
        };
        let mut buffer = None;
        unsafe { device.CreateBuffer(&description, Some(&initial), Some(&mut buffer)) }
            .context("CreateBuffer failed")?;
        let buffer = buffer.context("CreateBuffer returned nothing")?;

        let view_description = Direct3D11::D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: Direct3D11::D3D11_UAV_DIMENSION_BUFFER,
            Anonymous: Direct3D11::D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: Direct3D11::D3D11_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 1,
                    Flags: 0,
                },
            },
        };
        let mut view = None;
        unsafe {
            device.CreateUnorderedAccessView(&buffer, Some(&view_description), Some(&mut view))
        }
        .context("CreateUnorderedAccessView failed")?;
        let view = view.context("CreateUnorderedAccessView returned nothing")?;

        let query_description = Direct3D11::D3D11_QUERY_DESC {
            Query: Direct3D11::D3D11_QUERY_EVENT,
            MiscFlags: 0,
        };
        let mut completed = None;
        unsafe { device.CreateQuery(&query_description, Some(&mut completed)) }
            .context("CreateQuery failed")?;
        let completed = completed.context("CreateQuery returned nothing")?;

        unsafe {
            context.CSSetUnorderedAccessViews(0, 1, Some(&Some(view.clone())), None);
            context.CSSetShader(&shader, None);
        }
        Ok(Self {
            device,
            context,
            buffer,
            completed,
            _view: view,
            _shader: shader,
        })
    }

    /// Runs the first dispatch and verifies the GPU wrote it.
    pub fn self_check(&self) -> anyhow::Result<()> {
        self.run()?;
        if self.read_counter()? != 1 {
            return Err(anyhow::anyhow!("GPU self-check failed"));
        }
        Ok(())
    }

    /// One dispatch, waited to completion. Returns CPU-observed elapsed time.
    pub fn run(&self) -> anyhow::Result<Duration> {
        let start = Instant::now();
        unsafe { self.context.Dispatch(1, 1, 1) };
        self.wait_for_gpu()?;
        Ok(start.elapsed())
    }

    pub fn read_counter(&self) -> anyhow::Result<u32> {
        let description = Direct3D11::D3D11_BUFFER_DESC {
            ByteWidth: size_of::<u32>() as u32,
            Usage: Direct3D11::D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: Direct3D11::D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        let mut staging = None;
        unsafe {
            self.device
                .CreateBuffer(&description, None, Some(&mut staging))
        }
        .context("Create staging buffer failed")?;
        let staging = staging.context("Create staging buffer returned nothing")?;
        unsafe { self.context.CopyResource(&staging, &self.buffer) };
        self.wait_for_gpu()?;
        let mut mapped = Direct3D11::D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context.Map(
                &staging,
                0,
                Direct3D11::D3D11_MAP_READ,
                0,
                Some(&mut mapped),
            )
        }
        .context("Map counter failed")?;
        let value = unsafe { *mapped.pData.cast::<u32>() };
        unsafe { self.context.Unmap(&staging, 0) };
        Ok(value)
    }

    fn wait_for_gpu(&self) -> anyhow::Result<()> {
        // Flush submits work; the event query confirms it actually completed.
        unsafe {
            self.context.End(&self.completed);
            self.context.Flush();
        }
        let deadline = Instant::now() + GPU_TIMEOUT;
        loop {
            // GetData maps the S_FALSE "not ready yet" to Ok and leaves the
            // out-parameter untouched, so readiness is read from `ready` alone.
            let mut ready: u32 = 0;
            unsafe {
                self.context.GetData(
                    &self.completed,
                    Some(std::ptr::addr_of_mut!(ready).cast::<c_void>()),
                    size_of::<u32>() as u32,
                    Direct3D11::D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
                )
            }
            .context("GetData failed")?;
            if ready != 0 {
                return Ok(());
            }
            unsafe { self.device.GetDeviceRemovedReason() }.context("GPU device status failed")?;
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "GPU completion timed out after {} seconds",
                    GPU_TIMEOUT.as_secs()
                ));
            }
            std::thread::sleep(Duration::from_millis(1)); // No busy-spin while the GPU works.
        }
    }
}

fn blob_text(blob: &ID3DBlob) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    String::from_utf8_lossy(bytes).into_owned()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoopStats {
    /// Pulses after the self-check.
    pub pulses: u64,
    pub missed_deadlines: u64,
    pub completion_total: Duration,
    pub completion_max: Duration,
    pub wall: Duration,
    pub process_cpu: Duration,
}

impl LoopStats {
    pub fn mean_completion(&self) -> Duration {
        if self.pulses == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.completion_total.as_secs_f64() / self.pulses as f64)
    }

    /// What the GPU's counter has to read: one write per pulse, plus the
    /// self-check that preceded them.
    pub fn expected_counter(&self) -> u32 {
        // Add before narrowing: `as u32 + 1` panics on a debug build once the
        // truncation lands on u32::MAX. Wrapping matches the GPU's own counter.
        (self.pulses + 1) as u32
    }
}

fn process_cpu() -> anyhow::Result<Duration> {
    let (mut created, mut exited, mut kernel, mut user) = Default::default();
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
    }?;
    let ticks = |time: windows::Win32::Foundation::FILETIME| {
        (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)
    };
    // FILETIME counts 100-nanosecond ticks.
    Ok(Duration::from_nanos((ticks(kernel) + ticks(user)) * 100))
}

/// Every duration this tool prints is in milliseconds.
pub fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn verify(pulse: &Pulse, stats: &LoopStats) -> anyhow::Result<()> {
    let expected = stats.expected_counter();
    let counter = pulse.read_counter()?;
    if counter != expected {
        return Err(anyhow::anyhow!(
            "GPU counter verification failed: expected {expected}, read {counter}"
        ));
    }
    Ok(())
}

/// Pulses every `interval` until `stop` is raised or `limit` elapses (`None` runs
/// forever), then reads the counter back and verifies every pulse landed.
pub fn pulse_loop(
    pulse: &Pulse,
    interval: Duration,
    limit: Option<Duration>,
    stop: &Stop,
) -> anyhow::Result<LoopStats> {
    let mut stats = LoopStats::default();
    let start = Instant::now();
    let finish = limit.map(|limit| start + limit);
    let cpu_start = process_cpu()?;
    let mut next = start;
    while !stop.is_set() && finish.is_none_or(|finish| Instant::now() < finish) {
        let now = Instant::now();
        if now < next {
            let until = finish.map_or(next, |finish| next.min(finish));
            if let Some(remaining) = until.checked_duration_since(now)
                && stop.wait(remaining)
            {
                break;
            }
            continue;
        }
        let elapsed = pulse.run()?;
        stats.completion_total += elapsed;
        stats.completion_max = stats.completion_max.max(elapsed);
        stats.pulses += 1;
        next += interval;
        if next < Instant::now() {
            // Skip missed deadlines; never burst to catch up.
            stats.missed_deadlines += 1;
            next = Instant::now() + interval;
        }
    }
    stats.wall = start.elapsed();
    stats.process_cpu = process_cpu()?.saturating_sub(cpu_start);
    verify(pulse, &stats)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::LoopStats;

    #[test]
    fn mean_completion_is_zero_before_the_first_pulse() {
        let empty = LoopStats::default();
        assert_eq!(empty.mean_completion(), Duration::ZERO);
        assert_eq!(
            empty.expected_counter(),
            1,
            "the self-check still wrote once"
        );

        let stats = LoopStats {
            pulses: 4,
            completion_total: Duration::from_millis(10),
            ..LoopStats::default()
        };
        assert_eq!(stats.mean_completion(), Duration::from_micros(2500));
        assert_eq!(stats.expected_counter(), 5);
    }
}
