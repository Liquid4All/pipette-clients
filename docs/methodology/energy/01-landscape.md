# Energy benchmarks: what other providers measure

Survey of the metrics, scopes and methods that benchmark bodies, leaderboards,
model providers and silicon vendors publish for large language model (LLM)
inference energy. The survey is as of 2026-09-19. Its purpose is to pick metrics pipette can publish that readers
already know how to interpret, and to avoid the mistakes the field has already
made. Items marked *unverified* could not be confirmed from a primary source.

The main finding: **no public benchmark measures LLM energy on phones, Macs
or mini-PCs to a standard.** Datacenter energy is well covered (MLPerf Power,
ML.ENERGY, AI Energy Score). On-device coverage is vendor percentage claims and
one-off academic studies. Pipette can publish what is missing. No ready-made set
of rules exists to adopt; the rules in
[03-proposal.md](03-proposal.md) are assembled from the datacenter methods and
the phone studies.

## 1. Benchmark bodies and leaderboards

### MLPerf Inference Power (MLCommons)

The strictest measurement anywhere and the reference for the run rules.

- **Scope:** whole system at the wall (AC), "all components that are sensitized
  by LoadGen".
- **Method:** an analyzer certified for PTDaemon, the power and temperature
  daemon of the Standard Performance Evaluation Corporation (SPEC); Yokogawa
  in practice. It runs on a separate director host; a *ranging* pass fixes the analyzer's current and
  voltage ranges, then the *testing* pass runs with auto-ranging forbidden.
  Power and performance are taken from the same run. Ambient 20–25 °C. Minimum
  power window 60 s.
- **Metrics:** Offline/Server report throughput per watt (samples/J, tokens/J
  for LLMs since v4.1); single- and multi-stream report energy per query.
- **On-device:** edge systems yes; **phones explicitly excluded** ("complexity
  and variability of smartphones"). The v6.1 Edge Agentic benchmark (Sep 2026,
  Qwen3.6-27B on llama.cpp) reports time to first token (TTFT), time per
  output token (TPOT) and per-turn latency with no energy metric.
- Sources: rules
  <https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc>,
  methodology paper <https://arxiv.org/abs/2410.12032> (HPCA 2025), tooling
  <https://github.com/mlcommons/power-dev>.

### MLPerf Client v1.5 (Nov 2025)

Windows x64, Windows on Arm, Linux, macOS; iPad GUI. LLM metrics are TTFT and
tokens/s. Power measurement is **experimental**: optional CLI hooks into MLPerf
Power / PTDaemon, so it is wall power from an external meter, and the release
note names neither the metric nor its units.
<https://mlcommons.org/2025/11/mlperf-client-1-5-release/>

### MLPerf Mobile, MLPerf Tiny

Mobile: performance only, no power, for the reason quoted above. Tiny v1.4 (Jul
2026): µJ per inference via EEMBC EnergyRunner with calibrated DC measurement on
microcontrollers, 25 configurations submitted.
<https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/>

### ML.ENERGY Leaderboard v3.0 and Zeus (University of Michigan)

- **Scope:** GPU only, from the cumulative energy counter of the NVIDIA
  Management Library (NVML), through the Zeus library. Facility figures are
  estimates multiplied by the power usage effectiveness (PUE).
- **Method:** vLLM 0.11.1 on H100 and B200, 46 models, 7 tasks, 1,858
  configurations, batch sizes 8–3,072. Energy counted only in the
  **steady-state window** of a saturated batch, excluding ramp-up and drain.
- **Metrics:** J per output token, J per response (= J/token × mean output
  length, which captures verbosity), tokens/s/W, median inter-token latency,
  power as % of thermal design power (TDP), and time-versus-energy Pareto frontiers under a TPOT limit.
- Sources: <https://arxiv.org/abs/2505.06371>,
  <https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/>.
  Zeus has an open RFC for Apple Silicon via IOReport
  (<https://github.com/ml-energy/zeus/issues/159>).

### Hugging Face AI Energy Score (v1 Feb 2025, v2 Dec 2025)

- **Scope:** GPU energy on one H100 80 GB, batch size 1. CPU and RAM energy are
  logged "for transparency" but not scored.
- **Method:** CodeCarbon (reads NVML), 1,000 samples per task, 10 runs, 11
  tasks.
- **Metrics:** GPU **Wh per 1,000 queries**, plus a 1–5 star rating by
  percentile within a task.
- Finding worth carrying over: v2 added a reasoning task, and reasoning models
  used ~30× more energy on average (150–700× worst case), driven by 300–800×
  more output tokens. **Per-token efficiency and per-request energy diverge as
  soon as output length varies.**
- Sources: <https://huggingface.github.io/AIEnergyScore/>,
  <https://huggingface.co/blog/sasha/ai-energy-score-v2>. The older
  optimum-benchmark LLM-Perf leaderboard reported tokens/kWh over GPU+CPU+RAM
  (*secondary source only*).

### TokenPowerBench (arXiv 2512.03024, AAAI-26)

J/token, J/response and instantaneous power with energy **split between prefill
and decode**, and three declared measurement levels: GPU telemetry, node
(Intelligent Platform Management Interface, IPMI), and rack (power distribution
unit, PDU). Idle baseline subtracted. The prefill/decode split and
the declared-level idea carry over directly to devices.
<https://arxiv.org/abs/2512.03024>

### SemiAnalysis InferenceMAX / InferenceX

Tokens/s per "all-in provisioned utility MW" covering GPUs, CPUs, network and
facility. The MW figure is **summed TDPs, not measured**; measuring with
ipmitool is listed as future work.
<https://newsletter.semianalysis.com/p/inferencemax-open-source-inference>

### Others

- **Artificial Analysis:** no energy metric found (*unverified negative*).
- **Epoch AI (Feb 2025):** analytical estimate, ~0.3 Wh per GPT-4o query from
  100B active parameters, 500 output tokens, H100 at ~1,500 W all-in, 10%
  compute utilization.
  <https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use>
- **SPEC ML:** unreleased since 2021. SPEC's real contribution is PTDaemon.

## 2. On-device benchmark apps

None of the commercial on-device benchmarks publish LLM energy.

| App | AI metric | Energy |
|---|---|---|
| UL Procyon AI Text Generation | geometric mean of TTFT and tokens/s scores over four models | none. UL's separate one-hour battery benchmark covers office/video/idle and states it must not be extrapolated |
| Geekbench AI | scores at full / half / quantized precision | none |
| AI Benchmark (ETH) | performance ranking | store text claims "power consumption" but no published figures (*unverified*) |
| AnTuTu AI, PassMark | performance | none (*unverified negative*) |
| Qualcomm AI Hub profile jobs | runtime, load time, per-layer timing, memory, compute-unit placement | none |

The closest analogues to what pipette would do are academic or hobbyist:

- **MELTing Point** (Laskaridis et al., MobiCom 2024,
  <https://arxiv.org/abs/2403.12844>): the phone reference setup. Monsoon HV
  power monitor at 5 kHz with the battery bypassed (phones disassembled), a
  YKUSH hub cutting the USB bus power (VBUS) so charging current is never
  counted, a FLIR camera for skin temperature, brightness fixed at 25 %, 3 repetitions with
  mean ± SD. Reports mAh/token, mWh/token and "prompts per full battery".
  Found iPhone 14 Pro sustaining up to 13.8 W (18 W peaks), Galaxy S23 under
  8.5 W (14 W peaks), with pronounced throttling. Also covers Jetson.
- **"Understanding LLMs in Your Pockets"** (<https://arxiv.org/abs/2410.03613>):
  stock dynamic voltage and frequency scaling (DVFS) governors cost up to
  40 % latency at equal energy versus the best CPU/GPU/memory frequency
  setting. **Governor and power mode are confounds to
  record.**
- **"Is Your NPU Ready for LLMs?"** (<https://arxiv.org/abs/2607.05475>, Jul
  2026): reads Qualcomm Power Telemetry through Linux powercap on SM8650/8750/
  8850, reporting µJ/token per prefill and decode, per rail (system on chip
  (SoC), CPU, GPU, neural processing unit (NPU)). NPU decode energy is often 2–4× CPU/GPU, and the host CPU is up to
  30 % of energy even during NPU inference.
- **apple-silicon-llm-bench** (github.com/john-rocky): J/token on Mac from
  `powermetrics` package power; on iPhone uses battery-gauge tick windows
  because the gauge reports in 5 % steps. Finding: **the energy ranking inverts
  the tokens/s ranking**. On an M4 Max, MLX 0.24 J/token at 24.7 W versus
  CoreML/ANE 0.48 J/token at 12.7 W.
- **GreenBench** (<https://arxiv.org/abs/2608.28667>, Aug 2026): M4 Pro,
  `powermetrics` at 2 s, mJ/token and gCO₂ per 1k tokens, **no wall
  validation**. Its 0.47 W "package power" next to 8–12 W system power is not
  credible. It shows what happens without a wall check.
- **"Intelligence per Watt"** (Saad-Falcon et al.,
  <https://arxiv.org/abs/2511.07885>, Nov 2025): accuracy per watt and per
  joule for 20+ local models on 8 accelerators over 1M real queries, 10 runs
  per query, 50 ms polling. Accelerator scope only: NVML, ROCm SMI, and on
  Apple Silicon the GPU line of `powermetrics` alone, chosen so Macs compare
  like-for-like with cloud GPUs. No idle subtraction, no thermal or charge
  gating, vLLM standardized. Two lessons carry over. Its headline per-watt
  metric rewards slow devices, and its own appendix shows it: an A18 Pro scores
  ~7× the IPW of an M4 Max while taking 9–26× longer per query, and the
  per-joule gap "narrows or inverts". And its iPhone rows name no instrument:
  the harness lists three telemetry sources, none of which exist on iOS, and
  the measured FP16 power lands on the chip's spec-sheet peak. Its M4 Max "GPU
  subsystem" figure of 240 W is also not credible next to the 73.6 W our M5
  Max IOReport `GPU Energy` counter read under an MLX matmul loop
  ([02-device-capabilities.md](02-device-capabilities.md#ioreport-energy-model-no-root)). Useful for its accuracy term and its real-query
  distribution; not a measurement method to copy.
- **"LLM Inference at the Edge"** (<https://arxiv.org/abs/2603.23640>, Jun
  2026): Android BatteryManager with screen off, iPhone battery state-of-charge,
  INA219 at 1 kHz on a Hailo NPU, nvidia-smi at 100 ms. iPhone 16 Pro lost
  ~41 % of peak throughput under sustained load.
- **"Part-time Power Measurements"** (<https://arxiv.org/abs/2312.02741>,
  SC24): on A100/H100 NVML samples only ~25 % of runtime, so short kernels are
  badly mis-estimated; mitigations are longer runs, repetition and external
  correction. **Measurement windows of seconds, not single requests.**

## 3. Model-provider disclosures

All datacenter, none on-device; relevant only as reference magnitudes and for
what a full accounting includes.

| Provider | Figure | Scope | Method |
|---|---|---|---|
| Google Gemini (Aug 2025, <https://arxiv.org/abs/2508.15734>) | median text prompt 0.24 Wh, 0.03 gCO₂e, 0.26 mL water | accelerators 58 %, host CPU/DRAM 25 %, idle provisioned machines 10 %, PUE overhead 8 %; embodied excluded | fleet telemetry. Accelerator-only would read 0.10 Wh: a **2.4× gap between narrow and full scope** |
| OpenAI (Altman, Jun 2025) | 0.34 Wh per average query | undisclosed | undisclosed |
| Microsoft / Oviedo et al. in *Joule* (Jun 2026) | 0.16–0.60 Wh per query | datacenter, modeled | model, ~300 output tokens |
| Mistral Large 2 life-cycle assessment (LCA) (Jul 2025) | 1.14 gCO₂e, 45 mL water, materials depletion per 400-token response | full lifecycle over 18 months; training+inference 85.5 %, manufacturing/transport/end-of-life 11 % | AFNOR frugal-AI framework, peer reviewed |
| Meta Llama model cards | training GPU-hours × TDP, PUE-adjusted tCO₂eq | training only | estimate |
| Anthropic, Apple | no primary per-query figure found | | |

## 4. Silicon vendors

Every figure is relative, from the vendor's own tests, with no stated
measurement boundary.

- Qualcomm Snapdragon 8 Elite Gen 5: NPU 37 % faster, "16 % better performance
  per watt".
- MediaTek Dimensity 9500: "56 % lower peak power" for 3B LLM output; "33 % less
  power" with 1.58-bit BitNet.
- Arm Lumex/SME2: "up to 2× energy efficiency", "3× at platform level".
- Google Tensor G5: Gemini Nano "2.6× faster, 2× more efficient".
- Intel Panther Lake NPU: ">15 INT8 TOPS/W" at 3–3.5 W (*secondary sources*);
  a compute figure, not tokens.
- AMD Ryzen AI: tokens/s claims only, no tokens/W found.
- NVIDIA datacenter: tokens/s per MW, all-in utility MW modeled from TDPs.
  Jetson: results by power mode (15 W / MAXN), no tokens/J.

## 5. Standards and regulation

- **ISO/IEC TR 20226:2025:** survey of sustainability metrics, not a procedure.
- **IEEE P7100** (in development): measurement framework for AI training and
  inference environmental impact.
- **Green Software Foundation SCI for AI** (ratified Dec 2025):
  SCI = (E × I + M) per functional unit (per token, per inference), operational
  plus embodied.
- **AFNOR Spec 2314** (Jun 2024): the lifecycle method behind Mistral's LCA.
- **EU AI Act GPAI Code of Practice** (Jul 2025): energy reporting mainly for
  training; inference largely absent.

## 6. Summary table

| Provider | Metric (units) | Scope | Method | On-device? |
|---|---|---|---|---|
| MLPerf Inference Power | samples/J or tokens/J; energy per query | whole system, wall AC | PTDaemon analyzer, ≥60 s, ranging pass | edge yes, phones no |
| MLPerf Client v1.5 | experimental, units unpublished | wall | PTDaemon CLI hook | PC / Mac |
| MLPerf Tiny | µJ per inference | DC to device | EEMBC EnergyRunner | MCU only |
| ML.ENERGY v3 | J/token, J/response, tokens/s/W | GPU only | Zeus / NVML, steady-state window | no |
| HF AI Energy Score v2 | GPU Wh per 1k queries, stars | GPU (CPU/RAM logged) | CodeCarbon, H100, batch 1 | no |
| TokenPowerBench | J/token split prefill/decode | GPU → node → rack | NVML / IPMI / PDU | no |
| SemiAnalysis InferenceX | tokens/s per MW | facility, modeled | summed TDPs | no |
| Google Gemini | Wh, gCO₂e, mL per median prompt | accelerator + host + idle + PUE | fleet telemetry | no |
| Mistral LCA | gCO₂e per 400-token response | full lifecycle | AFNOR LCA | no |
| Vendors (Qualcomm, MediaTek, Arm, Tensor, Intel, AMD) | relative % perf/W, TOPS/W | unstated | internal | phone / PC, unverifiable |
| UL Procyon / Geekbench AI / AnTuTu | performance only | n/a | n/a | yes, no energy |
| MELTing Point (academic) | mWh/token, mAh/token, prompts per charge | battery DC, bypassed | Monsoon 5 kHz | phones, destructive |
| apple-silicon-llm-bench, GreenBench (academic) | J/token, mJ/token | SoC package | powermetrics | Mac |
| Intelligence per Watt (academic) | accuracy per W and per J | accelerator only (GPU line on Mac); iPhone instrument unstated | NVML / ROCm SMI / powermetrics at 50 ms | Mac, iPhone, PC GPUs |

## 7. What this means for pipette

1. **Metric names are settled; scope is not.** Publish J per output token and
   J per prompt token (prefill/decode split, as TokenPowerBench), J per request
   at a declared shape (as ML.ENERGY and AI Energy Score), and average W. Those
   are the units every reader of the leaderboards above already understands.
2. **Every number must carry its measurement boundary.** Google's 2.4× gap
   between accelerator-only and full accounting, GreenBench's implausible
   package figure, and MLPerf's wall-only rule are the same lesson. A SoC
   counter, a battery gauge and a wall meter are three different quantities.
3. **J/token alone misleads; pair it with J/request.** AI Energy Score v2's
   reasoning result is the proof.
4. **Steady-state windows, idle baselines, repetitions, same-run performance.**
   Adopted by every serious method above; none of them is optional.
5. **Phones need the charger accounted for and the governor recorded.**
   MELTing Point's VBUS switch and the DVFS study are the two phone-specific
   rules that no datacenter method covers. Every phone study cuts the charger
   because it measures only the battery side; metering the charger side instead
   satisfies the same rule, and [03-proposal.md](03-proposal.md) takes that
   route.
6. **No other benchmark covers this.** MLPerf does not do
   phones, MLPerf Client's power is experimental and PC-only, the app
   benchmarks have nothing. Whatever pipette publishes will be the first
   cross-platform on-device LLM energy series with a stated method, which is
   why the method has to be written down before the numbers.
