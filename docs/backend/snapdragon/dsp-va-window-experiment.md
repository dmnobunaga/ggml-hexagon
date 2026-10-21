# Experimental DSP VA window (FastRPC mempool backend)

This branch is a prototype. The stable `experiment/adaptive-htp-offload` branch is unchanged.
This feature is **disabled by default**, and has not yet been built or run on hardware.

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
- Session shutdown error `ggml_htp_close() = 0x2c` is not addressed here.
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

The current prototype has no on-device CI or verified performance data.
Do **not** interpret a successful model load as proof that PP/TG will execute
correctly. Test incrementally and retain baseline binaries for rollback.
