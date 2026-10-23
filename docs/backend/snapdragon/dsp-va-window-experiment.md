# Experimental DSP VA window (FastRPC mempool backend)

This branch is a prototype. The stable `experiment/adaptive-htp-offload` branch is unchanged.
This feature is **disabled by default**. It has been exercised on Snapdragon 8 Elite Gen 5 (Hexagon v81), but remains experimental.

## Purpose

The regular segmented FastRPC backend registers every weight segment with
`HAP_mmap2()` at model load and retains all mappings for the model lifetime.
DSPQueue can use smaller working sets and reclaim ordinary DSP VA mappings.
This opt-in mode keeps all weights in host-side RPCMEM but stores FD/size
metadata on the DSP and lazily maps weight segments just before operations
that consume them. On-demand writable runtime buffers are instead mapped immediately and permanently
pinned for their lifetime. The window accounts for both resident weight mappings
and pinned runtime mappings. When the budget is reached, only inactive
**weight** mappings are evicted by LRU.

An eviction occurs only after suspending/flushing HMX and work queues.
The mode converts segment-based tensor descriptors immediately before
operations, since previously cached DSP pointers may have become stale.

**Safety constraints**
- Only segmented RPCMEM is supported.
- Workspace stays permanently mapped outside the configured window budget.
- Weight-plan reservations are evictable; newly allocated runtime segments are
  pinned and writable. A write into an evictable weight segment fails safely.
- Runtime allocations are conservatively treated as writable even if they
  happen to contain a weight; this may reduce available window capacity.
- The sum of pinned runtime segments and weight segments needed for one op
  must fit the configured window.
- Failure to unmap or map an active segment returns an error; it must not
  silently fall back to stale addresses.
- Session shutdown error `ggml_htp_close() = 0x2c` persists after successful benchmarks. It is tracked separately; the origin of the error is not yet proven.
- DSPQueue DMA64 extended mapping is **not** used by this prototype.
- `-ngl 99` may still be restricted by host RAM headroom; start with fixed `-ngl 16`.

## Build

Build the **FastRPC/mempool** variant:
`-DGGML_HEXAGON_USE_MEMPOOL=ON`. Install matching AP binaries/libraries
and the DSP skeleton `libggml-htp-v81.so` together.
Do not mix this branch's DSP skeleton with an older host library.
The FastRPC IDL `register_rpcmem_segment` signature changed to include
`pinned`; regenerate/rebuild AP stubs and the DSP skeleton together.

## A/B procedure

Run each condition in a **new process**. Keep the same model, `-t`, `-fa`,
`--ubatch-size`, cooling state, and repeated benchmark settings.

### Exact-layout adaptive offload

The adaptive planner first reserves the **exact requested layer layout**.
If all weight segments register successfully, the requested offload is used
immediately (e.g. `effective_ngl=36` and `registered_prefix=4/4`).
There is no intermediate binary search or repeated remapping on this path.

If the full plan only partially registers, the planner rolls it back and
tries smaller `ngl` values in descending order, reserving each candidate's
**own** layout transactionally. Segment feasibility is not assumed monotonic
across `ngl`. Unhealthy mapping state or failed rollback aborts the load.

`GGML_HEXAGON_ADAPTIVE_CANDIDATE_RETRY` is obsolete and ignored; exact-layout
selection is now the default. Diagnostics report `adaptive_strategy=exact-full`
or `adaptive_strategy=exact-descending`.

Regression checks (on-device): verify exact-full chooses all 36 layers for
Gemma 4 E2B Q4_0 with four segments, verify the fallback with an oversized
model, and verify healthy rollback after a rejected candidate. Perform
VA-window ON/OFF runs with the same effective `ngl`. Automated on-device
validation is still required for this branch.

Stable baseline (current FastRPC, no DSP window):
```sh
unset GGML_HEXAGON_DSP_VA_WINDOW_MB
export GGML_HEXAGON_SEGMENTED_RPCMEM=1
export GGML_HEXAGON_ADAPTIVE_OFFLOAD=1
export GGML_HEXAGON_SEGMENT_MB=512
export GGML_HEXAGON_MEMORY_RESERVE_MB=1024
./llama-bench -t 6 --poll 1000 -ngl 15 -fa 1 --ubatch-size 128 -p 128 -n 128 -r 3 -m /sdcard/Spark-X2.5-4B.gguf
```

Experimental window, first at 15 layers (correctness/regression), then 16,
then 24:
```sh
export GGML_HEXAGON_DSP_VA_WINDOW_MB=2560
./llama-bench -t 6 --poll 1000 -ngl 15 -fa 1 --ubatch-size 128 -p 128 -n 128 -r 3 -m /sdcard/Spark-X2.5-4B.gguf
# After verifying valid PP and TG:
./llama-bench -t 6 --poll 1000 -ngl 16 -fa 1 --ubatch-size 128 -p 16 -n 4 -r 1 -m /sdcard/Spark-X2.5-4B.gguf
# After verifying 16 layers and graph correctness:
./llama-bench -t 6 --poll 1000 -ngl 24 -fa 1 --ubatch-size 128 -p 16 -n 4 -r 1 -m /sdcard/Spark-X2.5-4B.gguf
```

For small models (`<small_model>.gguf`), use **both** settings:
```sh
unset GGML_HEXAGON_DSP_VA_WINDOW_MB
./llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 128 -p 128 -n 128 -r 5 -m /sdcard/<small_model>.gguf
export GGML_HEXAGON_DSP_VA_WINDOW_MB=2560
./llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 128 -p 128 -n 128 -r 5 -m /sdcard/<small_model>.gguf
```
A model fully fitting in the window should need no evictions after initial mapping.
Record `pp128` and `tg128` medians/variance separately. Also compare
output tokens with deterministic decoding under `llama-cli`:
benchmark throughput alone does **not** establish numerical correctness.
Vary `-ngl` to match actual offloaded layers when comparing performance.

## Diagnostics

Look for `[DSP-VA] weight mapping window=`, `[RPC-SEG] lazy-weight`,
`[RPC-SEG] eager` with `pinned=1`,
`[DSP-VA] map`, `[DSP-VA] map failed`, and `[DSP-VA] unmap failed`
in `adb logcat`. Host `[HTP-OFFLOAD]` summary prints the selected window
and effective number of layers.

No automated on-device CI currently verifies the implementation. Benchmarks have
confirmed that PP and TG execute; deterministic token-by-token numerical
correctness is **not yet verified**. Test incrementally and retain baseline
binaries for rollback.

## Verified on-device checkpoint — 2026-10-21

Environment: Snapdragon 8 Elite Gen 5, HTP v81, Spark-X2.5-4B BF16
(4.11B parameters, 7.66 GiB GGUF), `-t 6 --poll 1000 -fa 1`,
`--ubatch-size 128`. All measurements in this section were supplied by
the device operator; they are not CI measurements.

**Artifact/build tested**: `f308c7499` (llama-bench build 12481).
This is an earlier experimental branch snapshot, **not** a validation of
subsequent source edits on the branch.

| Configuration | PP | TG | Repeats | Interpretation |
| --- | ---: | ---: | ---: | --- |
| FastRPC, 15 layers, VA window **OFF** | PP128 6.13 ± 1.26 t/s | TG32 3.08 ± 1.12 t/s | 3 | Matched A/B baseline |
| FastRPC, 15 layers, VA window 2560 MiB | PP128 6.44 ± 0.18 t/s | TG32 3.74 ± 0.22 t/s | 3 | Matched A/B experimental |
| FastRPC, 16 layers, VA window 2560 MiB | PP16 4.49 t/s | TG4 3.24 t/s | 1 | PP and TG completed |
| FastRPC, 24 layers, VA window 2560 MiB | PP16 10.52 t/s | TG4 4.60 t/s | 1 | PP and TG completed |
| FastRPC, 24 layers, VA window 2560 MiB | PP128 6.83 ± 0.27 t/s | TG64 4.48 ± 0.20 t/s | 3 | Longer PP/TG run completed |

At 24 layers: `effective_ngl=24`, 12 weight segments registered,
`dsp_weight_mapped=5242.30 MiB` (segment allocations, **not** simultaneous
resident DSP VA), workspace 256 MiB; CPU file-backed weight mapping
3241.27 MiB. The reported DSP VA window is 2560 MiB and excludes
the permanently mapped workspace. Reported `extended_map=0`.

**Direct mapping evidence** from `adb logcat` captured during a PP16/TG4
run: weight slots 1–12 registered `lazy-weight ... pinned=0`; runtime
slots 13–15 registered `eager ... pinned=1` (6.00, 17.00, 8.16 MiB
initially); `[DSP-VA] map` logs show maps and evictions growing.
The 16th logged weight mapping reports `maps=16 evictions=11 active=2032
MiB`. These are **partial counters**, not whole-benchmark totals:
mapping logs stop after the first 16 and only repeat every 128 maps.

The matched 15-layer A/B in this checkpoint did **not** show a throughput
regression from enabling the VA window (6.13→6.44 PP128, 3.08→3.74 TG32),
but run-to-run variance is significant. Older 15-layer FastRPC tests
reported TG ≈5.38–5.62 t/s under different conditions. They are not
directly comparable with this A/B or the 24-layer TG64 result; no global
performance winner has been established.

**Known limitations and follow-ups**:
1. `ggml_htp_close()` consistently reports `0x2c`, including after
   successful PP/TG. Cleanup is deferred to process exit; repeated
   load/unload within one process remains unverified.
2. No deterministic cross-backend logits/token correctness comparison.
3. No per-graph DSP VA `map_count`, `evict_count`, mapping latency, or
   queue synchronization timing; CPU/DSP per-stage profiling is needed.
4. Small-model PP/TG regression suite is pending; verify both default
   mode (VA window unset) and opt-in mode.
5. Large-model memory headroom varied between runs; swapping and device
   thermal state can influence throughput.

**Milestone**: the data demonstrate working BF16 FastRPC inference with a
5.24-GiB weight allocation and active weight remapping inside a 2.5-GiB
DSP VA window. They do **not** prove all weights fit in DSP address
space simultaneously, nor that the close error is harmless.

## Verified AUTO load-mode performance fix — 2026-10-24

Device: Snapdragon 8 Elite Gen 5, HTP v81. Model: Gemma 4 E2B Q4_0,
2.82 GiB GGUF, 36/36 effective HTP layers, four DSP weight segments
(1345.42 MiB mapped), workspace 256 MiB, DSP VA window OFF.
Benchmarks use `-t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024
-p 500,800,1024 -n 128 -r 3`.

An A/B at build `1f4b81c01` isolated a significant performance regression
to the load-mode difference: AUTO had 1756 MiB file-backed CPU weights,
while explicit `-lm none` allocated the same 1756 MiB as anonymous CPU
weights. No change to the effective offload or DSP graph was needed.

| Test (t/s, mean ± reported variation) | AUTO before fix | explicit NONE before fix | AUTO after fix |
| --- | ---: | ---: | ---: |
| pp500 | 844.50 ± 247.12 | 1717.72 ± 59.71 | **1725.79 ± 4.78** |
| pp800 | 857.06 ± 83.97 | 1547.03 ± 78.46 | **1563.38 ± 55.53** |
| pp1024 | 1140.98 ± 94.48 | 1657.28 ± 116.03 | **1566.28 ± 13.20** |
| tg128 | 27.40 ± 0.95 | 33.66 ± 0.35 | **34.72 ± 0.08** |

Fix `100ae215015ade2890b243d1fbfee6ead00aea9e`
(`fix/htp-auto-load-mode`): AUTO again honors backend
`mmap_support=false` on HTP; explicit `-lm mmap` remains available.
The fixed AUTO run is build `100ae2150` (12488), measured by the device
operator, not CI. Measured CPU-weight accounting: mmap 0.00 MiB,
anonymous 1756.00 MiB; average AP graph overhead 1.467 ms vs 5.956 ms
in pre-fix AUTO; DSP graph nodes 1498 and HMX coverage 73.4% in both.

**Scope of conclusion:** the controlled load-mode A/B strongly identifies
the memory-loading policy as the source of the regression, but does not
separately measure major page faults. Historical PR #85 speed numbers were
recorded with a different backend version, so are not an apples-to-apples
benchmark of this code change. Exact-descending fallback and large BF16
behavior on the new AUTO policy are not yet verified.

**Memory safety:** anonymous CPU weights increase resident demand.
Fixed AUTO reported `MemAvailable_start=4855.06 MiB`,
`MemAvailable_loaded=2206.92 MiB`, and `SwapFree` declined by ~988 MiB
during model loading, while process `VmSwap_loaded=0`. Do not infer
that the process itself used swap from the system-wide decline.
Do not use `GGML_MEM_GUARD_WARN_ONLY=1` for risky BF16 benchmarks.
Explicit `-lm mmap` must be used for the initial large-model checks.

Next validation: use a small-model, opt-in synthetic full-plan rejection
to exercise exact-descending rollback and per-candidate layouts safely;
then verify Spark-X2.5-4B BF16 with explicit mmap, a low fixed ngl and
memory headroom before any higher-layer or VA-window trials.
