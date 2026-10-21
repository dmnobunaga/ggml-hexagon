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

Stable baseline (current FastRPC, no DSP window):
```sh
unset GGML_HEXAGON_DSP_VA_WINDOW_MB
export GGML_HEXAGON_SEGMENTED_RPCMEM=1
export GGML_HEXAGON_ADAPTIVE_OFFLOAD=1
export GGML_HEXAGON_SEGMENT_MB=512
export GGML_HEXAGON_MEMORY_RESERVE_MB=1024
export GGML_HEXAGON_ADAPTIVE_CANDIDATE_RETRY=1
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
