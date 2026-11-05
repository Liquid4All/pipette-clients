//! The GPU's own readings on Windows from `D3DKMTQueryAdapterInfo`, what Task
//! Manager shows, with no driver. The gate reads its temperature and the per-rep
//! readings post the same struct, so the two cannot disagree.
//!
//! It answers on the Radeon 890M (HX 370) and 8060S (Ryzen AI Max+ 395);
//! Intel's driver fills the structs with zeros. Every adapter is asked because
//! the GMKtecs also enumerate a virtual display. Physical adapter 0 only: an
//! integrated GPU is never a linked-adapter chain.
//!
//! The records the per-rep readings post keep the structs' own field names,
//! less the adapter's `Power`: no power is recorded.

use serde_json::{Value, json};
use windows_sys::Wdk::Graphics::Direct3D::{
    D3DKMT_ADAPTER_PERFDATA, D3DKMT_ADAPTERINFO, D3DKMT_CLOSEADAPTER, D3DKMT_ENUMADAPTERS2,
    D3DKMT_NODE_PERFDATA, D3DKMT_QUERYADAPTERINFO, D3DKMT_QUERYSTATISTICS,
    D3DKMT_QUERYSTATISTICS_ADAPTER, D3DKMTCloseAdapter, D3DKMTEnumAdapters2,
    D3DKMTQueryAdapterInfo, D3DKMTQueryStatistics, KMTQAITYPE_ADAPTERPERFDATA,
    KMTQAITYPE_NODEPERFDATA, KMTQUERYADAPTERINFOTYPE,
};
use windows_sys::Win32::Foundation::LUID;

pub(crate) struct AdapterReadings {
    pub luid: LUID,
    pub perf: Option<D3DKMT_ADAPTER_PERFDATA>,
    pub nodes: Vec<D3DKMT_NODE_PERFDATA>,
}

/// The hottest GPU's own temperature in whole °C, or `None` where no adapter
/// reports one (a driver that leaves it unset reports 0).
pub fn gpu_temp_max_c() -> Option<u32> {
    let adapters = Adapters::open()?;
    adapters
        .0
        .iter()
        .filter_map(|a| adapter_perf(a.hAdapter))
        .filter_map(|perf| super::whole_celsius(perf.Temperature))
        .max()
}

/// Every adapter's performance data and its nodes', as the driver filled them.
pub(crate) fn read_adapters() -> Vec<AdapterReadings> {
    let Some(adapters) = Adapters::open() else {
        return Vec::new();
    };
    adapters
        .0
        .iter()
        .map(|a| AdapterReadings {
            luid: a.AdapterLuid,
            perf: adapter_perf(a.hAdapter),
            nodes: (0..node_count(a.AdapterLuid))
                .map_while(|node| node_perf(a.hAdapter, node))
                .collect(),
        })
        .collect()
}

struct Adapters(Vec<D3DKMT_ADAPTERINFO>);

impl Adapters {
    fn open() -> Option<Self> {
        let mut enumerate = D3DKMT_ENUMADAPTERS2::default();
        // SAFETY: a null adapter array asks only for the count, written to a
        // struct we own.
        if unsafe { D3DKMTEnumAdapters2(&raw mut enumerate) } != 0 {
            return None;
        }
        let mut adapters =
            vec![D3DKMT_ADAPTERINFO::default(); usize::try_from(enumerate.NumAdapters).ok()?];
        enumerate.pAdapters = adapters.as_mut_ptr();
        // SAFETY: `pAdapters` points at `NumAdapters` entries we own, and the
        // call writes at most that many.
        if unsafe { D3DKMTEnumAdapters2(&raw mut enumerate) } != 0 {
            return None;
        }
        adapters.truncate(usize::try_from(enumerate.NumAdapters).ok()?);
        Some(Self(adapters))
    }
}

impl Drop for Adapters {
    fn drop(&mut self) {
        self.0.iter().for_each(|adapter| {
            let close = D3DKMT_CLOSEADAPTER {
                hAdapter: adapter.hAdapter,
            };
            // SAFETY: closes a handle the enumeration opened, once.
            unsafe { D3DKMTCloseAdapter(&raw const close) };
        });
    }
}

fn adapter_perf(adapter: u32) -> Option<D3DKMT_ADAPTER_PERFDATA> {
    query(
        adapter,
        KMTQAITYPE_ADAPTERPERFDATA,
        D3DKMT_ADAPTER_PERFDATA::default(),
    )
}

fn node_perf(adapter: u32, node: u32) -> Option<D3DKMT_NODE_PERFDATA> {
    let request = D3DKMT_NODE_PERFDATA {
        NodeOrdinal: node,
        ..D3DKMT_NODE_PERFDATA::default()
    };
    query(adapter, KMTQAITYPE_NODEPERFDATA, request)
}

/// `data` filled in by `D3DKMTQueryAdapterInfo`, or `None` if the call failed.
/// The request's own fields (a node ordinal) go in through `data`.
fn query<T>(adapter: u32, kind: KMTQUERYADAPTERINFOTYPE, mut data: T) -> Option<T> {
    let mut query = D3DKMT_QUERYADAPTERINFO {
        hAdapter: adapter,
        Type: kind,
        pPrivateDriverData: (&raw mut data).cast(),
        PrivateDriverDataSize: u32::try_from(size_of::<T>()).ok()?,
    };
    // SAFETY: `pPrivateDriverData` points at a `T` we own, of exactly the size
    // declared, which is the struct `kind` fills.
    (unsafe { D3DKMTQueryAdapterInfo(&raw mut query) } == 0).then_some(data)
}

/// How many engine nodes the adapter has, 0 where it does not say.
fn node_count(luid: LUID) -> u32 {
    let mut stats = D3DKMT_QUERYSTATISTICS {
        Type: D3DKMT_QUERYSTATISTICS_ADAPTER,
        AdapterLuid: luid,
        ..D3DKMT_QUERYSTATISTICS::default()
    };
    // SAFETY: an adapter query reads `Type` and `AdapterLuid` and writes the
    // result into the struct we own; the header declares it `const` all the
    // same, hence the cast.
    if unsafe { D3DKMTQueryStatistics((&raw mut stats).cast_const()) } != 0 {
        return 0;
    }
    // SAFETY: an adapter query fills the `AdapterInformation` arm of the union.
    unsafe { stats.QueryResult.AdapterInformation.NodeCount }
}

pub(super) fn adapter_record(luid: LUID, p: &D3DKMT_ADAPTER_PERFDATA) -> Value {
    json!({
        "AdapterLuid": luid_record(luid),
        "PhysicalAdapterIndex": p.PhysicalAdapterIndex,
        "MemoryFrequency": p.MemoryFrequency,
        "MaxMemoryFrequency": p.MaxMemoryFrequency,
        "MaxMemoryFrequencyOC": p.MaxMemoryFrequencyOC,
        "MemoryBandwidth": p.MemoryBandwidth,
        "PCIEBandwidth": p.PCIEBandwidth,
        "FanRPM": p.FanRPM,
        "Temperature": p.Temperature,
        "PowerStateOverride": p.PowerStateOverride,
    })
}

pub(super) fn node_record(luid: LUID, n: &D3DKMT_NODE_PERFDATA) -> Value {
    json!({
        "AdapterLuid": luid_record(luid),
        "NodeOrdinal": n.NodeOrdinal,
        "PhysicalAdapterIndex": n.PhysicalAdapterIndex,
        "Frequency": n.Frequency,
        "MaxFrequency": n.MaxFrequency,
        "MaxFrequencyOC": n.MaxFrequencyOC,
        "Voltage": n.Voltage,
        "VoltageMax": n.VoltageMax,
        "VoltageMaxOC": n.VoltageMaxOC,
        "MaxTransitionLatency": n.MaxTransitionLatency,
    })
}

fn luid_record(luid: LUID) -> Value {
    json!({ "LowPart": luid.LowPart, "HighPart": luid.HighPart })
}
