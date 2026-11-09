# Peak Memory: Linux

This is the per-platform companion to the high-level
[peak memory methodology](peak-memory-usage.md). On Linux the memory benchmark
runs on these engines:

- torch-oai's Docker and uv servers.
- llama.cpp's CPU and GPU flavors.
- The OVMS and OpenVINO servers, on the CPU or a GPU.
- The ONNX GenAI server, on the CPU or CUDA. A DirectML cell is watched as a GPU
  cell, and fails where the server cannot load DirectML.

pipette refuses a cell before anything starts when nothing on Linux reads its
device memory:

- torch-oai's AMD flavors on a host whose cards are not the job's alone
  ([AMD GPU counter](#amd-gpu-counter)).
- llama.cpp's Ascend flavors.
- llama.cpp's OpenVINO backend on any device but the CPU. A memory reading
  cannot tell a run on that device from a fallback to the CPU.
- llama.cpp's Snapdragon builds, Android's and Linux's, whose GPU and NPU
  memory nothing reads on Linux.
- OVMS and OpenVINO on `npu` or on a custom device such as `AUTO`.
- ONNX GenAI on WinML or on a custom provider.

## torch-oai (Docker and uv)

The torch-oai path measures the server through a cgroup, because vLLM and
SGLang can use worker processes in addition to the parent server: the
container's, or for a uv server pipette's own, which the server shares as
pipette's child ([uv servers](#uv-servers)).
The probe starts as the server launches, before it is ready, so the model load
counts, as on every other engine. One thread reads the host and the GPU figure
on the same 10 ms tick once it has the cgroup, and both once more after the
request. A container's cgroup is known once `docker container inspect` gives
its full id and the pid of its main process, about 200 ms after `docker run`;
pipette's own at once. The reading is
`pipette-memprobe`'s cgroup sampler, and torch-oai only finds the cgroup.

### Host Counter

`max_host_bytes` is the cgroup's own memory: the anonymous memory of its
processes, its shared memory (tmpfs and shm segments) and the kernel memory
billed to it, each page counted once however many of its processes share it.
The probe reads the three from the cgroup v2 `memory.stat` (`anon`, `shmem`,
`kernel`) on every tick and keeps the largest sum, since the kernel keeps no
high-water mark for it.

The file is read on the docker host, not through `docker exec`:
`/proc/<pid>/cgroup` of the container's main process gives the container's
cgroup, and its `memory.stat` sits under `/sys/fs/cgroup` at that path, which is
`system.slice/docker-<id>.scope` under systemd's cgroup driver and
`docker/<id>` under cgroupfs. A main process in a cgroup below the container's,
where docker in docker puts its init, still gives the container's. pipette's
own cgroup, a uv server's, sits in the same tree, at
`user.slice/user-<uid>.slice/session-<n>.scope` for a login session. So
pipette must run on the host itself, on Linux 5.18 or later, which is when
`memory.stat` gained `kernel`. One read took about 11 µs on the NVIDIA A10 host
below while vLLM loaded and served (16 µs at the 99th percentile and 0.18 ms at
worst over 24,000 reads), about a thousandth of the tick.

The figure leaves the page cache out. The kernel bills a cached file page to
the first cgroup that reads it and keeps the bill after that reader is gone. So
whether a container is billed for its image's libraries, its Python files and
the model's weights depends on what ran on the host before it, not on the
server. The cgroup's `memory.peak` counts that cache. On an NVIDIA A10 with
`LiquidAI/LFM2.5-350M` on vLLM v0.21.0 (`--gpu-memory-utilization 0.5`,
`max_memory_usage_512`), two runs in each state:

| Page cache before the run | `memory.peak` | This probe |
| --- | ---: | ---: |
| Image and model cached (right after another run) | 2,489,380,864 and 2,493,558,784 B | 2,359,459,840 and 2,362,474,496 B |
| Image layers and model files evicted | 4,642,521,088 and 4,646,535,168 B | 2,368,479,232 and 2,368,638,976 B |

To evict, each file of the image's layers and of the model was advised out of
the page cache (`posix_fadvise` with `POSIX_FADV_DONTNEED`), and `fincore`
showed none of them cached before the run. At the stop the container held
2.24 GB of anonymous memory and 20 MB of shared memory in every run; its file
cache was 0.15 GB cached against 2.29 GB evicted, and its kernel memory 74 MB
against 83 MB. That kernel bookkeeping for the files it read is why the
evicted runs read 6 to 9 MB higher here. With only the model's files evicted,
or the image partly cached, `memory.peak` read 3.20 GB and 4.30 GB while the
container's own memory read 2.36 GB. `max_gpu_bytes` read 13,021,216,768 B in
every run.

The server needs almost none of that cache, only the 40 to 90 MB of library
code it keeps running. Started with nothing cached under a docker memory limit,
it served at its usual steady speed at 2.4 GB once past its first two requests
(80% and 31% slower than unlimited), thrashed at 2.35 GB and was killed at
2.30 GB.

The figure is not the `VmHWM` the other Linux engines report:

- It adds the kernel memory billed to the container (page tables, kernel
  stacks, slab: the 74 to 83 MB above), which `VmHWM` never counts.
- It leaves out the file pages the processes map, mostly library code: about
  0.9 GB across vLLM's two processes here, which `VmHWM` counts.
- It leaves out about 66 MB of NVIDIA driver memory mapped into the engine's
  process, which the container's accounting never sees.
- Swapped-out pages are not in it, as with `VmHWM`. The run's
  `observation_max_swap_bytes` is the cgroup's `memory.swap.peak`, the kernel's
  high-water mark, where the kernel keeps one.

Its meaning is closest to the macOS figure (`phys_footprint`), which also
leaves file pages out.

A row whose command preview reads `memprobe: cgroup memory.peak` recorded
`memory.peak` as `max_host_bytes`, file cache included, and is not comparable
with one that reads `memory.stat anon+shmem+kernel`. A row that reads
`memory.stat anon+shmem+kernel` carries in its `client_extras` `memory.peak`
at the stop (`cgroup_memory_peak_bytes`, absent before Linux 5.19 and on a uv
row, whose cgroup is pipette's), `memory.stat`'s `anon`, `file`, `shmem` and
`kernel` at the stop (`memory_stat`), where the page cache is `file` less
`shmem`, and the number of `memory.stat` reads (`host_samples`).

The cell fails, naming the file it tried, rather than reporting another figure.
Three hosts are refused before the server starts:

- pipette does not run on Linux: off Linux, docker runs in a VM (Docker
  Desktop's) whose cgroups pipette cannot see;
- `/sys/fs/cgroup/cgroup.controllers` is missing or lists no `memory`
  controller: cgroup v1, a hybrid layout, or a kernel booted without the
  memory controller;
- on uv, pipette's own cgroup counts no memory of its own: the host's root
  cgroup, which has no `memory.current` ([uv servers](#uv-servers)).

The others show once the probe has the cgroup, and the cell reports them at its
stop, after the load and its request:

- `/proc/<pid>/cgroup` of the container's main process cannot be read, or
  names no cgroup v2 path of the container: docker runs in another pid
  namespace or a VM, `/proc` hides other users' processes (`hidepid`), or
  `docker` is another runtime that names its cgroups otherwise;
- `memory.stat` is not under `/sys/fs/cgroup` at that path: a tree hidden by a
  container pipette itself runs in, or rootless docker without a delegated
  cgroup;
- `memory.stat` prints no `kernel` count (Linux before 5.18).

### GPU Counter

An NVIDIA flavor reads the server's GPU memory through NVML, as the servers
below do. On every tick from shortly after the server starts the probe takes
the `usedGpuMemory` NVML lists for every process on every GPU, the larger of a
GPU's compute and graphics figures for a process in both lists, and sums it
over the server's processes whose `/proc/<pid>/cgroup` is its cgroup or one
below it; `max_gpu_bytes` is the largest sum. NVML reports the host's pids, and
every process of the server and the workers it starts (vLLM's engine core, a
tensor parallel worker) sits in its cgroup: the container's, which names it by
its full id (`docker-<id>.scope` under systemd's cgroup driver, `docker/<id>`
under cgroupfs), or pipette's own for a uv server, where only the server's
processes count ([uv servers](#uv-servers)). The lookup that gives the cgroup
takes about 200 ms for a container, long before the engine reaches the GPU. A
process that exits between NVML's list and its cgroup read is skipped. NVML
opens before the server launches and treats a GPU or a process list it cannot
read as the servers' reader does ([below](#ovms-openvino-and-onnx-genai)); a
card in MIG mode refuses an unprivileged caller both.

On vLLM and SGLang this figure is what the engine reserves on the card when it
starts, not what the model needs. vLLM takes the `--gpu-memory-utilization`
fraction of the card's memory (vLLM's own default, 0.92 in the builds measured
here, unless the cell sets it; the vLLM versions pipette installs are the
[bundled torch_oai catalog](../../crates/pipette-artifacts-catalog/bundled-catalog/torch_oai/catalog.toml)'s)
and fills what the weights leave with KV cache. SGLang sizes its weights
and KV cache pool by `--mem-fraction-static`, which it derives from the card's
memory when the cell does not set it. The A10 run above, at 0.5, read half of
the card's 24,564 MiB plus about 136 MiB. So `max_gpu_bytes` follows the card
and the launch setting: it is not comparable across cards, or with the other
engines, which hold what the model uses.

The cell fails rather than reporting a host-only figure when:

- NVML cannot load or opens no GPU, before the server starts;
- `docker container inspect` never names the container;
- on every read a GPU refuses one of NVML's process lists;
- NVML reports one of the cgroup's processes as not available (MIG, vGPU, no
  permission); another process's does not count;
- NVML never attributes memory to a process in the server's cgroup: the server
  did not reach the GPU, pipette runs in another pid namespace than the
  server, the `docker` pipette runs is another runtime whose cgroups do not
  name the container, or `/proc` hides other users' processes (`hidepid`), so
  their cgroups cannot be read.

Each `docker` call the probe makes is killed after 30 s, so a stalled daemon
fails the cell rather than holding it. Off Linux an NVIDIA flavor's cell is
refused before anything starts. CPU flavors read no GPU and leave
`max_gpu_bytes` unset. A uv build's flavor is its tag's: `cpu` is the CPU,
`rocm` an AMD GPU, any other an NVIDIA GPU.

On the NVIDIA A10 cell above, the NVML figure, 13,021,216,768 B, is the
12,418 MiB that `nvidia-smi` showed for the container's engine core when
sampled every 50 ms beside the run. vLLM allocates its KV cache and captures its
CUDA graphs before it is ready. It climbs to 12,416 MiB with no higher
transient, and the request adds 2 MiB. Some rows carry a GPU figure from an
`nvidia-smi` poll that started at readiness, and others carry no GPU figure
where that poll failed. On this server, a row with a figure from that poll
agrees with NVML: that poll read the same 13,021,216,768 B in each of two runs.

### AMD GPU Counter

An AMD flavor reads whole cards, as no per-process source attributes a
server's memory to the card that holds it. On an MI325X (Linux 6.8, vLLM 0.26
ROCm, `Qwen/Qwen2.5-1.5B-Instruct`), with the card's own rise as the reference:

| Source | One card | Two cards (`--tensor-parallel-size 2`) |
|---|---|---|
| amdgpu's fdinfo, summed over the server's DRM clients (`drm-resident-vram`) | 100.00% | 200.16%: each worker holds a client on both cards, and each bills its ~250 GB |
| KFD's `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` | exact | exact, but named by the host's pids |
| The card's `mem_info_vram_used` | exact | exact |

fdinfo bills a buffer on every card that its process maps it to, and a worker
maps its own buffer to its peer's card. The kernel fusion driver (KFD) names a
process by its pid on the host. A job in a pid namespace of its own, as a Slurm
job in a Kubernetes pod runs, cannot match that pid to its own. The `pasid` that
KFD prints for the process reads 0, so it cannot join KFD to fdinfo either.

So the probe reads whole cards, and only cards that belong to the job alone. It
reads each card that this process may open. A scheduler such as Slurm that
gives a job whole cards lets the job read the driver's topology
(`/sys/class/kfd/kfd/topology/nodes/<n>`) for those cards alone, and refuses it
the others (permission denied). That refusal is what shows that no one else's
memory is on the job's cards. In a Slurm job, the count of cards that the job
may open must equal `SLURM_GPUS_ON_NODE`, the GPUs that Slurm gave it on the
node. That rule also admits a job that Slurm gave every card of its node.

Each card's `drm_render_minor` names its `renderD<n>`, and the probe reads that
node's `device/mem_info_vram_used` on every tick. A card that the job holds
alone holds nothing of anyone else's, so all it holds above its floor is the
server's. The floor is the least that the card has held since the probe opened
it, before the launch: a server that stopped just before can still be handing
its memory back then. `max_gpu_bytes` is the largest sum over the cards of the
reading less the floor. The row records that floor (`gpu_floor_bytes` in
`client_extras`) and whether the open saw the cards stop falling
(`gpu_floor_settled`). Two cells run back to back read 209 GB and 254.7 GB with
the reading at the open as the floor, and 254,747,144,192 and 254,747,127,808 B
with the least reading as the floor.

A card hands memory back after the processes that held it have exited: on an
MI325X, vLLM's 237.7 GiB took 7.7 s more to free once its last process was
gone, by SIGKILL or SIGTERM alike, in steps of about 8 GiB a third of a second
apart. A server launched at once read the old figure at its launch, idle 7.7 s
later, and allocated its first memory 16 to 17 s after the launch, so the least
since the open was the idle card. A server that allocates sooner would hide
that floor, so the open waits, up to 30 s, until the cards' used VRAM has not
fallen by more than 64 MiB for a second; a row whose cards were still falling
at the limit says `gpu_floor_settled: false`.

The cell is refused before the launch when:

- this process may open every card on the host outside a Slurm job, since
  another process may hold memory on the card that the server uses;
- in a Slurm job, it may open another count of cards than Slurm gave it;
- it may open no card;
- a card's used VRAM cannot be read;
- a card's `gpu_id` fails to read for a reason other than a refusal;
- `SLURM_GPUS_ON_NODE` is not a count;
- a card that it may open is one partition of a card.

A card split into compute partitions (`DPX`, `QPX` or `CPX` in its
`current_compute_partition`) shares its VRAM between them. A partition's render
node, the driver's `amdgpu_xcp_<n>` device, reports no VRAM of its own. The
cards measured ran in `SPX`, which gives one render node the card's figure. The
driver also lists seven `amdgpu_xcp_<n>` render nodes per card beside it, which
the topology does not name. The cell fails at its stop when the cards' used
VRAM never rose above the floor.

On an AMD Instinct MI325X in a one-card Slurm job, in vLLM's ROCm container
(vLLM 0.26.0, `Qwen/Qwen2.5-1.5B-Instruct`, `max_memory_usage_100`, vLLM's
default `--gpu-memory-utilization` of 0.92), run as an ordinary user, the cell
read 254.7 GB of GPU memory, 0.92 of the card's 256 GiB plus vLLM's own, and
5.96 GB of host memory, taking 1.2 MB off for pipette.

### uv servers

A uv server runs as plain processes of pipette's user, started by pipette, so
it shares pipette's own cgroup: a login session's (`session-<n>.scope`, which
an ssh session gets, a worker started as `ssh <host> <command>` included), a
service's, a batch job's, or a container's. The probe reads that cgroup's
`memory.stat` and takes off the own memory of the processes in it that are
not the server's: pipette itself and the processes above or beside it, such
as an ssh daemon's and a login shell.

The server's processes are its session, which pipette starts it leading,
whatever process group each is in. That includes a build step that `ninja`
moves into a group of its own during a JIT compile, and every process that
those start in a session of its own. The probe decides each process once, when
it first sees it, by its session and parents in `/proc`. A process in a session
of its own whose parent has exited goes to init and counts as another's, which
`others_max_bytes` shows.

The others' own memory is `RssAnon + RssShmem + VmPTE` from their
`/proc/<pid>/status`, read on every tick. Their page tables are the one part of
the kernel memory that the cgroup bills them that a process reports; the rest
of their kernel memory stays in the figure. pipette's own memory always comes
off, whether or not `cgroup.procs` lists it. NVML's processes are filtered the
same way, so only the server's count on the GPU. Each tick reads the others
before and after `memory.stat` and takes off each one's larger figure: one that
exits between the reads is still in `memory.stat`, and the peak would keep a
figure that counted it.

The swap term is the largest `memory.swap.current` less the others' `VmSwap`
over the reads, since the cgroup's `memory.swap.peak` would count pipette's swap
too.

A uv row records, in its `client_extras`:

- `others_bytes`: what the stop's read took off.
- `others_max_bytes`: the most that any read took off. The cell warns when it
  comes to 512 MiB or more: other work that large in pipette's cgroup leaves the
  figure less certain, in either direction.
- `host_bytes_at_launch`: the first read's figure, right after the launch.
  Memory that the cgroup held before the server grew, such as kernel memory
  from earlier work, counts in the server's figure, and this says how much. In
  a one-card MI325X job in vLLM's ROCm container it read 407 to 424 MB: the
  container's own memory, in the host figure too.
- `others_listed`: `false` where `cgroup.procs` reads empty, so only pipette's
  own memory came off.
- `memory_stat`: the whole cgroup's, pipette's included, as a diagnostic. The
  row leaves out `memory.peak`, which is the whole session's or job's watermark
  and covers the time before the server too.

Its command preview reads `memprobe: pipette's cgroup memory.stat
anon+shmem+kernel less its other processes' RssAnon+RssShmem+VmPTE`.

The probe reads the cgroup whose `cgroup.procs` lists pipette. Inside a
container whose cgroup namespace names a parent, as pyxis's does, that is the
cgroup below it (`task_0`). The probe refuses the cell before the launch when
another cgroup below pipette's holds processes.

Taking the others off, rather than adding up the server's processes, keeps a
page counted once: SGLang's forked workers share pages copy-on-write, so
summing their `RssAnon` read 16.4 GB against the cgroup's 6.4 GB on an H100
while it compiled, and 32.5 GB against 12.2 GB over two AMD GPUs at tensor
parallel 2. pipette's own memory is small next to a server: about 25 MB in a
short run, and 158 MB in a worker that had run for days. A process the server
started that left its group, and whose parent then exited, counts as another
process, and comes off with the others.

A cell is refused before the launch when pipette's own cgroup counts no memory
of its own: the host's root cgroup, which holds every process on the host and
has no `memory.current`. Inside a cgroup namespace, a container's or a batch
job's, `/proc/self/cgroup` names `/`, the namespace's root, which is that
cgroup and counts its memory. There `cgroup.procs` can list no pid at all, from
another pid namespace, and only pipette's own memory comes off. The other uv
cells record no memory observation.

On the NVIDIA A10 with `LiquidAI/LFM2.5-350M` on vLLM 0.30.0 from uv
(`uv-vllm://server=0.30.0&build=cu129&python=3.12`, `max_memory_usage_100`,
vLLM's default `--gpu-memory-utilization`), alternating with a reading of the
same server started in a systemd user scope of its own (`systemd-run --user
--scope`), this read 2,408,022,016 and 2,421,350,400 B of host memory against
the scope's 2,428,239,872 B, and 23,597,154,304 B of GPU memory in each. It
took off 24.6 MB: pipette, an ssh daemon's process and a shell. The scope's
first run, on a cold compile cache, read 4,323,577,856 B, since the compiler's
processes count whichever way the cgroup is read; the torch-oai cell runs no
compile first ([compile caches](peak-memory-usage.md#compile-caches)). On an
H100, a reader that took the others off pipette's way agreed with a scope's
reading of the same server within 0.2%: 2,886 against 2,887 MB on vLLM 0.27.1,
and 4,049 against 4,042 MB on SGLang 0.5.8. A scope is not used because it
needs the user's systemd manager, which a batch job or a container lacks.

### Host and GPU are separate pools

The cgroup's own memory is host memory. A discrete NVIDIA or AMD card's VRAM is
a separate pool, so the two fields are disjoint.

### Workload

An OpenAI-compatible `/v1/completions` request with exact-token text,
`max_tokens = 1`, `temperature = 0.0`, and `ignore_eos = true`. The probe
starts at the server's launch; the runner sends the request once the server is
ready, stops the probe after it returns, then validates the response usage
fields. The one token comes off the prefill, so no decode step runs. On vLLM
and SGLang a decode step would not move the peak: the KV cache and the decode
graphs are reserved when the server starts, and a decode step needs less
scratch than the prefill before it. On an NVIDIA A10 with
`LiquidAI/LFM2.5-350M` on vLLM v0.21.0, three runs of `max_memory_usage_512`
at 16 tokens and three at 1 read the same GPU bytes, and host bytes about
1 MB apart out of 2.36 GB. On vLLM, `--max-model-len` defaults to the prompt
plus that token, so `max_memory_usage_4096` runs with 4097; a cell that sets
`max_model_len` keeps its value.

### Caveats

- Both figures are current readings, not peaks the kernel or the driver
  keeps, so a rise and fall within one 10 ms tick can be missed. The
  driver-plus-runtime overhead mix in the GPU figure is not comparable across
  drivers or vendors.
- Both need pipette on the host itself: the cgroup tree, the pids NVML
  reports and the cgroups `/proc` shows for them are the host's.

## llama.cpp (Linux)

Implemented for the CPU flavors (`linux-arm64-cpu`, `linux-x64-cpu`,
`linux-s390x-cpu`), the OpenVINO backend (`linux-x64-openvino`) on the CPU, and
the GPU flavors: Vulkan (`linux-x64-vulkan`, `linux-arm64-vulkan`), SYCL
(`linux-x64-sycl-fp16`, `linux-x64-sycl-fp32`), ROCm (`linux-x64-rocm`) and
CUDA (`linux-cuda-12-8`, `linux-x64-cuda-13`, `linux-arm64-cuda-13`). The
Ascend flavors are refused before anything starts: the Ascend driver bills no
memory through DRM fdinfo, and pipette has no other reader of it. So is the
OpenVINO backend on any other device, for the reason under
[llama.cpp GPU counter](#llamacpp-gpu-counter), and the
Snapdragon build (`linux-arm64-snapdragon`), whose Adreno and Hexagon memory
nothing reads on Linux. A flavor that `pipette runtimes flavors` does not
list, an operator's own build, is refused too: pipette does not know its
devices.

### llama.cpp host counter

The runner launches `llama-bench` directly and reads it with
[`pipette-memprobe`'s `/proc` sampler](../../crates/pipette-memprobe/src/linux.rs),
the one the Android arm uses: the swap-aware figure, its guards and its
`client_extras` are described under [Android's host
counter](peak-memory-android.md#host-counter), with no
[model-file floor](peak-memory-android.md#no-model-file-floor). On a host with
no swap `VmSwap` is zero and `VmRSS` never exceeds `VmHWM`, so the figure is
`VmHWM`. On Linux, with pipette holding 600 MiB, the exit read's `ru_maxrss`
read 631 MB for a 313 MB `llama-bench`. That is why the exit figure counts only
when it is above pipette's own `VmHWM`.
The VL and ASR cells (`vl_max_memory`, `asr_max_memory`) report the same figure
and `client_extras` from their `llama-server`.

### llama.cpp GPU counter

Unset on the CPU flavors: they have no GPU allocation to attribute, so host RSS
is the whole picture.

A GPU flavor reads the tool's GPU memory with the reader the OVMS, OpenVINO and
ONNX GenAI servers use ([below](#ovms-openvino-and-onnx-genai)): every 10 ms,
the resident bytes of the tool's DRM clients plus what NVML attributes to it on
an NVIDIA GPU, the largest sum as `max_gpu_bytes`. These are current figures,
not watermarks, so they end at the tool's last tick before it exits. Vulkan and
SYCL on Intel (i915 and xe, Linux 6.8 or later) and Vulkan and ROCm on AMD
(amdgpu, Linux 5.14 or later) bill through the DRM drivers. Vulkan and CUDA on
an NVIDIA card run on NVIDIA's own driver, which prints no DRM fdinfo, so NVML
reads them instead: Vulkan from its graphics process list, CUDA from its compute
one. A CUDA figure is the CUDA context plus every allocation the tool holds on
the card.

The CUDA backend's pinned host buffers (`CUDA_Host` in the tool's log: the
weights left on the CPU, the output and CPU compute buffers) are in the
resident set and not in NVML's figure, so on CUDA `max_host_bytes` and
`max_gpu_bytes` are separate and the run needs their sum. While the weights
loaded at `-ngl 99` and `--load-mode none`, the resident set grew by the
`CUDA_Host` model buffer (402 MiB for its 398 on Qwen3.5-2B Q4_K_M, 1,769 for
1,756 on gemma-4-E2B Q4_K_M) and NVML's figure by the `CUDA0` one (1,212 for
1,211, 1,408 for 1,408).

A Jetson is the exception: its iGPU allocates from host memory mapped into
the tool, so the resident set already holds the CUDA buffers and nothing bills
them apart (NVML reports `Not Supported` for its memory, and nvgpu prints no
DRM fdinfo). On a host whose device tree names an `nvidia,tegra` SoC, the GPU
watch reads the resident set alone and reports no `max_gpu_bytes`. On a Jetson
AGX Orin (JetPack 7.2.1, `linux-arm64-cuda-13` b11829), gemma-3-4b-it Q4_K_M
(2,375 MiB) at `--load-mode none` held a `VmRSS` of 3,981 MiB through its
decode at `-ngl -1` and 4,000 MiB at `-ngl 0`, while `MemAvailable` fell by
256 MiB on the GPU run and 4,453 MiB on the CPU one.

Vulkan on an Intel iGPU, CUDA on an NVIDIA H100 and CUDA on a Jetson AGX Orin
have been run against the reader; SYCL, ROCm, CUDA 13 on a discrete card and
Vulkan on an AMD or NVIDIA card have not. On the H100,
`vl_max_memory_512x512_text1024_img1` on Qwen3.5-2B Q4_K_M read a
`max_gpu_bytes` of 2,835,349,504 (2,704 MiB) in each of three runs, the peak a
separate 5 ms NVML read of the same server command gave; `nvidia-smi` was not
read alongside. The cell fails rather than reporting a host-only figure on the
servers' conditions, chiefly when the tool's GPU held no memory (the run never
reached the GPU) or a DRM fd names no client or prints none of the memory keys
the read takes: i915 before Linux 6.8 prints no memory, and amdgpu before 5.14
prints nothing.

The OpenVINO backend runs on the device `GGML_OPENVINO_DEVICE` names in the
tool's environment (the cell's `envs`, the client configuration, or pipette's
own environment), and on the CPU when it is unset or empty. Unset, empty or
`CPU` watches the host alone. Any other device (`GPU`, `GPU.<n>`, `NPU`,
`AUTO`) is refused: the backend falls back to the CPU on a name OpenVINO does
not list (a lone GPU is `GPU`, two are `GPU.0` and `GPU.1`), and its device
query has opened the GPU by then, holding about 12 KiB there, so a memory
reading cannot tell that fallback from a GPU run. Such a cell runs with
`GGML_OPENVINO_DEVICE` unset or `CPU`.

The GPU figure counts the driver's host-visible buffers too. With the weights
not mapped, llama.cpp puts the layers it keeps on the CPU in a `Vulkan_Host`
buffer the driver allocates, so at `-ngl 0` the weights sit in GPU-billed pages
outside `VmHWM`, beside a compute buffer for the prompt on the GPU. As with the
servers, an Intel iGPU's bytes are outside the resident set, so the run needs
host plus GPU.

| Flavor, `-ngl` | `max_host_bytes` | `max_gpu_bytes` |
| --- | ---: | ---: |
| `linux-x64-cpu` | 298 MiB | `null` |
| `linux-x64-vulkan`, 99 | 132 MiB | 463 MiB |
| `linux-x64-vulkan`, 0 | 131 MiB | 433 MiB |

`LiquidAI/LFM2.5-350M-GGUF` Q4_K_M (a 219 MiB file), `max_memory_usage_512`,
llama.cpp b11109, on an Intel Core i9-13900HK (Iris Xe, i915, Linux 7.0, Mesa
26.0). Every GPU figure equalled a 2 ms read of the tool's fdinfo beside the run
(485,392,384 and 454,316,032 bytes), on each of four runs. llama.cpp's own log
accounts for most of it: at `-ngl 99`, 216 MiB of weights in the GPU buffer,
52 MiB (the token embedding) in the host-visible one and a 135 MiB compute
buffer; at `-ngl 0`, the 216 MiB of weights in the host-visible buffer and a
185 MiB compute buffer. The rest is, at `-ngl 99`, LFM2's 9 MiB KV cache and
its recurrent state on `Vulkan0` and 5 MiB of host-visible output and compute
buffers, and at both what the driver holds for the client.

### Shader cache

A GPU flavor compiles its kernels the first time a machine runs them: Vulkan
builds its pipelines through Mesa, which keeps them in the run's user's shader
cache (`~/.cache/mesa_shader_cache`, or `MESA_SHADER_CACHE_DIR`). A CUDA build
whose archive carries PTX but no compiled code for the card's architecture has
the driver compile its kernels on first use, into the CUDA cache
(`~/.nv/ComputeCache`, or `CUDA_CACHE_PATH`). Every flavor runs a warm-up
first, so the measured run loads its pipelines or kernels from the cache
([compile caches](peak-memory-usage.md#compile-caches)).

Without that first run, on an empty cache and on a cache an earlier run of the
same cell filled:

| Cell, flavor, `-ngl` | `max_host_bytes`, empty cache | `max_host_bytes`, filled cache | `max_gpu_bytes`, either |
| --- | ---: | ---: | ---: |
| `max_memory_usage_512`, `linux-x64-vulkan`, 99 | 232 MiB | 133 MiB | 463 MiB |
| `max_memory_usage_512`, `linux-x64-vulkan`, 0 | 179 MiB | 132 MiB | 433 MiB |
| `vl_max_memory_smoke`, `linux-x64-vulkan`, 99 | 241 MiB | 149 MiB | 493 MiB |
| `max_memory_usage_512`, `linux-x64-cpu` | 298 MiB | 298 MiB | `null` |
| `vl_max_memory_smoke`, `linux-x64-cpu` | 457 MiB | 457 MiB | `null` |

Three runs of each, on the Iris Xe above (Mesa 26.0, llama.cpp b11109), with
`LiquidAI/LFM2.5-350M-GGUF` Q4_K_M and `LiquidAI/LFM2.5-VL-450M-GGUF` Q4_0 with
its F16 projector. An empty cache is an empty `MESA_SHADER_CACHE_DIR`; on the
CPU rows the model files were also dropped from the page cache. Each set of
three agreed within 0.2 MiB, and the GPU figure did not move. An empty cache
took 2.9 s against 0.7 s for the text cell. With the run first, an empty cache
reads as a filled one: 133, 132 and 149 MiB for the three Vulkan rows, two runs
each, with the same GPU figures. The text cell then takes 1.5 s on a filled
cache. These are separate runs from the table above, whose filled-cache Vulkan
rows read under 1 MiB lower (132 and 131 MiB).

### llama.cpp workload

The cell runs [the llama-bench shape](peak-memory-usage.md#the-llama-bench-shape).
A cell that sets neither `mmap` nor `load_mode` runs with `--load-mode none`.
On a CPU flavor the weights then load into anonymous memory and count fully
toward the resident set, rather than as file-backed mmap pages, which would
under-report. On a GPU flavor they load into the driver's buffers. Each run,
the warm-up included, leads a session of its own and is stopped once it passes
the cell's `attempt_timeout_seconds`: 300 s by default, and raisable per cell
([the attempt limit](../pipette-cli/usage.md#5c-the-attempt-limit)).

## OVMS, OpenVINO and ONNX GenAI

Each engine runs one server process, so the cell watches that process from its
launch.

### Server host counter

The same `/proc/<pid>/status` sampler as llama.cpp, every 10 ms from the
measured server's launch until after the cell's one request, so the model load
counts: `VmHWM` as `max_host_bytes`, with the swap term beside it in the
observation. Each engine runs a warm-up server first
([compile caches](peak-memory-usage.md#compile-caches)).

### Server GPU counter

A GPU cell (OVMS and OpenVINO on `gpu`, ONNX GenAI's CUDA provider) reads the
server's GPU memory from two sources each tick and adds them: each bills a
different driver's devices. `max_gpu_bytes` is the largest sum.

The kernel's DRM drivers print each client's memory in `/proc/<pid>/fdinfo/<fd>`
for every fd the process holds on `/dev/dri/*`, with `drm-pdev` and
`drm-client-id` naming the client. The kernel source shows amdgpu printing its
clients' memory this way since Linux 5.19, and i915 and xe since 6.8. The read
sums `drm-resident-<region>`, the pages the client's buffers hold, over the
server's clients, each counted once however many fds refer to it. amdgpu from
Linux 5.19 to 6.12, Ubuntu's 6.8 kernels among them, prints
`drm-memory-<region>` instead (`vram`, `gtt` and `cpu`), which the kernel's
documentation names the deprecated alias of the resident term, so the read takes
it when no resident term is printed. From 6.13 amdgpu prints both, with the same
bytes, and the read takes the resident term alone rather than count the memory
twice. A driver that prints neither gives `drm-total-<region>`. NVIDIA's driver
prints no memory there, so an fd on a node whose sysfs driver is `nvidia` is
skipped. On an Iris Xe under i915 (Linux 7.0) the cell read 578,461,696 bytes
for OpenVINO and 537,817,088 for OVMS, each equal to a 2 ms read of the same
fdinfo beside the run. The kernel's figure sits above OpenVINO's own allocator
statistics (505 MiB against 412 MiB in a direct load), by what the driver holds
for the client beyond its tensors. No run on an AMD GPU has been checked against
the read.

amdgpu from Linux 5.14 to 5.18, Ubuntu 22.04's 5.15 kernels among them, prints
an older layout with no `drm-` keys. `pdev:` and `pasid:` name the client: each
open of the device gets its own GPU address space id, which every fd of that
open shows. ROCm swaps it for one id per process, the same on each of its GPUs,
so the device is part of the name. If the driver runs out of ids (about 65,000
opens at once across the system), it gives every new open id 0, and two such
opens of one device in one process count as one. `vram mem:`, `gtt mem:` and
`cpu mem:` give its memory in kB of 1024 bytes, computed by the same driver code
that fills `drm-memory-<region>` in 5.19, so the read sums the three as it sums
the alias. It takes this layout only from an fd that prints no `drm-client-id`:
later kernels still print `pasid:`, and their fds are read through the `drm-`
keys alone. So the floor for amdgpu is Linux 5.14, read from kernel source;
before it amdgpu prints nothing in fdinfo.

NVML, the library behind `nvidia-smi`, loaded at run time from the driver, reads
an NVIDIA GPU: the `usedGpuMemory` its compute and graphics process lists give
the server's pid, on each GPU the larger of the two, summed across the GPUs.
That figure is the CUDA context plus everything the process holds on the card.
On an NVIDIA A10 with Phi-3 mini int4 the CUDA cell read 3,630,170,112 bytes,
the 3,462 MiB `nvidia-smi` showed for the same process. OpenVINO's `gpu` runs on
the same card through NVIDIA's OpenCL driver and is read the same way:
1,455,423,488 bytes, `nvidia-smi`'s 1,388 MiB. `nvmlInit` takes over a second
on a host that does not keep the driver loaded (persistence mode off), so NVML
opens once, before the first tick. A host without NVIDIA's driver reads the DRM drivers alone. A GPU
NVML cannot open or read at the start is skipped, as is a single process list it
refuses then, so one card pipette may not touch does not hide the others; a list
a GPU does not support counts as empty. A list that answered at the start and
refuses a later read leaves that read's NVML figure unanswered, rather than a
sum without that list's processes, and any answered read outranks it.

The cell fails rather than reporting a host-only figure when:

- neither source billed the server any memory: the run did not reach the GPU,
  NVML could not open on an NVIDIA host (the error carries why), or pipette
  runs in another pid namespace than the server;
- a DRM fd names no client or prints none of the memory keys the read takes:
  i915 before Linux 6.8 prints no memory, and amdgpu before 5.14 prints
  nothing;
- NVML reports the server's memory as not available (MIG, vGPU, no permission)
  or, on every read, a GPU refuses one of its process lists;
- the server's fds cannot be listed.

OVMS and OpenVINO on `npu` or a custom device are refused. The NPU's driver
(`intel_vpu`) prints the same kind of fdinfo since Linux 6.19, but no NPU run
has been checked against it, and a custom device may resolve to either.

### Host and GPU pools of the servers

The CUDA allocations sit in the card's VRAM, outside `VmHWM`, so the two fields
are disjoint. So are an Intel iGPU's on Linux, though they are system RAM: the
buffers are not counted in the process's resident set (`RssShmem` stays 0), and
the box's available memory fell by the anonymous resident memory plus the GPU
bytes. The run needs both: `max_host_bytes` alone under-reports it.

| Engine, device | `max_host_bytes` | `max_gpu_bytes` |
| --- | ---: | ---: |
| OpenVINO, `cpu` | 1,058 MiB | `null` |
| OpenVINO, `gpu` | 763 MiB | 552 MiB |
| OVMS, `cpu` | 1,006 MiB | `null` |
| OVMS, `gpu` | 493 MiB | 513 MiB |

`OpenVINO/LFM2.5-350M-int8-ov`, `max_memory_usage_512`, on an Intel Core
i9-13900HK (Iris Xe, i915, Linux 7.0), OpenVINO 2026.4.0 and OVMS 2026.4. This
is the opposite of the same vendor's driver on Windows, where the device bytes
sit inside the working set ([Windows](peak-memory-windows.md#host-and-device-pools)).

## Code References

- [`pipette-torch-oai max_memory_usage`](../../crates/pipette-torch-oai/src/execute/max_memory_usage.rs)
- [`pipette-torch-oai memprobe`](../../crates/pipette-torch-oai/src/memprobe.rs),
  which finds the server's cgroup, a container's or pipette's own for a
  [uv server](../../crates/pipette-torch-oai/src/server/uv.rs), and
  [`pipette-memprobe`'s cgroup sampler](../../crates/pipette-memprobe/src/cgroup.rs)
- [`pipette-llamacpp max_memory_usage`](../../crates/pipette-llamacpp/src/execute/max_memory_usage.rs)
  and [its watch per flavor](../../crates/pipette-llamacpp/src/memory.rs)
- [`pipette-memprobe`'s GPU reader](../../crates/pipette-memprobe/src/linux.rs),
  over its [DRM](../../crates/pipette-memprobe/src/drm.rs) and
  [NVML](../../crates/pipette-memprobe/src/nvml.rs) sources, and its
  [AMD whole-card reader](../../crates/pipette-memprobe/src/amd.rs)
- [`pipette-onnx-genai max_memory_usage`](../../crates/pipette-onnx-genai/src/execute/max_memory_usage.rs),
  [`pipette-openvino max_memory_usage`](../../crates/pipette-openvino/src/execute/max_memory_usage.rs),
  [`pipette-ovms max_memory_usage`](../../crates/pipette-ovms/src/execute/max_memory_usage.rs)
