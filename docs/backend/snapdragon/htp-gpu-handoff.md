# Snapdragon HTP v81 — stable mixed-model milestone and GPU hybrid handoff (2026-10-24)

## Mission / result

The milestone is **working, memory-safe as guarded, stable-enough runtime compatibility across different model sizes and precisions on one Android Snapdragon 8 Elite Gen5 device**. Do not present single-run PP fluctuations as actionable speed regressions: Gemma PP128 changed from 642.98 to 1185.77 t/s on subsequent runs after Spark, while TG64 stayed ~33.42–34.82 t/s. User's priority is reliability and model coverage, not microbenchmark tuning.

Repository: `dmnobunaga/ggml-hexagon`.
Milestone branch: `experiment/dsp-va-profiler`. Related predecessor: `experiment/htp-descending-validation`.
Previous upstream reference: `https://github.com/kan-linux/ggml-hexagon/pull/85`.
Full experiment evidence: `docs/backend/snapdragon/dsp-va-window-experiment.md`.

### Proven workload 1 — Spark-X2.5-4B BF16

- GGUF **7.66 GiB**, **4.11B** parameters; requested `-ngl 99`, **actual `effective_ngl=35` out of 36 model layers**. The reported benchmark `ngl=99` is the user request, *not* the actual offload count. Attempting the 36th layer exhausted the applicable memory/allocation budget; adaptive fallback selected 35.
- `registered_prefix=9/9`, **DSP weight RPCMEM = 7443.37 MiB**, payload 7443.33 MiB, DSP workspace 256.00 MiB. `cpu_weight_mmap=1040.20 MiB`, `cpu_weight_anon=0.00 MiB`. These are weight-accounting sizes, not observed RSS.
- Segment cap 1024 MiB: slots 1–9 approximately `0.68, 990.16, 1000.39 × 6, 450.16 MiB`.
- **DSP mapped VA window = 3072 MiB**; much larger registered weights work through lazy per-op mapping/eviction within the bounded DSP VA budget.
- Observed at 35 layers, PP128 **76.62 t/s**, TG64 **4.22 t/s** (one run; not a claim of stable peak speed). At 22 layers, PP128 ~8.47 and TG64 ~4.62 t/s with 1024-MiB segments; nearly full offload dramatically helps prefill, not necessarily decode.

### Proven workload 2 — Gemma 4 E2B Q4_0

- GGUF **2.82 GiB**, **4.65B** parameters, previously verified full 36/36 DSP offload for its model architecture; different precision/architecture from Spark.
- Representative long bench using `-ngl 99`, `--ubatch-size 1024`, prompt sizes 200/500/800/1024: **PP200 1325.42**, **PP500 1746.12**, **PP800 1521.36**, **PP1024 1718.38**, **TG128 34.52 t/s**.
- After the Spark run a PP128/TG64 Gemma run reported **642.98/33.42 t/s**; repeat **1185.77/34.82 t/s**. This strong prefill variability exists on the same device; do **not** infer a code regression from unmatched, single-run PP. TG remained ~34–35 t/s.
- Same runtime continued to execute both models successfully in the reported tests; no observed Gemma TG regression relative to prior results. This does **not** resolve every historical cleanup issue (e.g. `ggml_htp_close` error `0x2c`).

## Current memory architecture

- `GGML_HEXAGON_SEGMENTED_RPCMEM=1`, `GGML_HEXAGON_ADAPTIVE_OFFLOAD=1`, `GGML_HEXAGON_SEGMENT_MB=1024`, `GGML_HEXAGON_MEMORY_RESERVE_MB=1536`, `GGML_HEXAGON_DSP_VA_WINDOW_MB=3072`. Optional `GGML_HEXAGON_DSP_VA_PROFILE=1`.
- Transactional exact-layout reservation: attempt requested offload; if allocation can't fit, roll back and test lower `ngl` plans, preserving the memory guard.
- Weight segments registered as RPCMEM; DSP side lazily `HAP_mmap2`s needed weight slots and evicts via `HAP_munmap2` / LRU under the VA window. Pinned runtime mappings/workspace are separate. Never conflate registered bytes with simultaneously DSP-VA-mapped bytes.
- 22-layer Spark packing: 512-MiB cap = 11 segments / ~4842.11 MiB; 1024-MiB cap = 6 segments / ~4842.09 MiB. A 6-slot, 3072-MiB profiler sample: 64 batches, 384 successful maps, 380 evictions, 104.137 ms cumulative transition time (~1.63 ms/batch). Packing changed churn, but apparent speed differences at PP16 are noisy; stop optimizing microsecond transitions for this milestone.
- Keep protective memory guard and safe `oom_score_adj=800` during BF16 experiments. Do not bypass guard or alter queue synchronization without compelling evidence.

## NEXT TASK — replace remaining CPU model execution with Android GPU

**Goal:** hybrid HTP + GPU inference for models that exceed full DSP offload: send the non-HTP layers/tensors to the phone GPU, preferably via the existing ggml/llama.cpp **Vulkan backend** if actually buildable and supported on target. OpenCL is an alternative if Vulkan is unsuitable. CPU may still be used for orchestration/small unsupported ops, but it must not remain the heavy compute fallback for the last model layers.

**Initial proof target:** Spark BF16, **HTP 35/36 + GPU remaining compatible layer(s)**, maintaining bounded DSP VA and adaptive memory guard. Also useful as secondary test: HTP 22 layers + GPU remaining layers, and full-DSP Gemma regression test.

Work plan:
1. **Inspect the actual repository** before changing code: CMake/build flags for `GGML_VULKAN` or OpenCL; Android driver/device support; registered backends/devices; HTP selection and forced one-device assumption in `src/llama.cpp`, model tensor placement in `src/llama-model.cpp`, buffer types, scheduler/graph splits and cross-backend tensor copies.
2. Identify whether HTP and Vulkan can coexist in a *single* `llama_model` / `ggml_backend_sched` instance, or whether the current HTP adaptive planner overrides standard GPU placement. Do not assume `-ngl 99` assigns any GPU layers; it is currently used for the HTP request. Distinguish **requested ngl** from **effective HTP ngl** in logs.
3. Implement **explicit heterogeneous tensor/layer placement** (HTP first, GPU for overflow; CPU only where unavoidable) with minimal HTP↔GPU crossings. Prefer a clear configurable partition with logging of actual tensors/layers per device. Preserve GPU-compatible embeddings/output/norm handling and correctness rather than assuming that all remaining 1040 MiB are exactly one transformer layer.
4. Reuse existing GGML backend scheduler and necessary cross-device activation transfers where possible; audit tensor lifetime, output/logits path, buffer ownership, model reload/close, and DSP VA pin/evict safety. No unguarded shared pointers across CPU/HTP/GPU device address spaces.
5. First establish a correctness test with identical prompts and deterministic sampling against CPU reference (tolerances for BF16/device arithmetic), then sustained PP/TG smoke under memory guard, then only look at throughput/regression. Measure transfer overhead and fallback behavior.
6. Keep this *next phase* in a new isolated experimental Git branch, leaving the working DSP-only milestone intact.

Relevant current source:
- `src/llama.cpp` — adaptive planner, `make_probe`, `exact-full`/`exact-descending`, currently checks for only one active HTP device.
- `src/llama-model.cpp` — memory guard and tensor/layer buffer placement; `GGML_HEXAGON_SEGMENT_MB` packing.
- `ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp` — segmented RPCMEM, host headroom, AP FastRPC and config.
- `ggml/src/ggml-hexagon/htp/entry.c` + `htp/dsp-ctx.h` — lazy DSP VA map/evict, HMX synchronization and profiling.

**Acceptance:** both Spark and Gemma run through one stable Android build; Spark 35/36 HTP with remaining eligible heavy ops explicitly on GPU (verified from tensor/backend placement and profile, not inferred from throughput); preserve adaptive rollback, DSP VA limit and memory safety; Gemma stays functionally correct without material runtime regression.
