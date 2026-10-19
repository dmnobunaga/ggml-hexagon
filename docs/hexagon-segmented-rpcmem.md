# FastRPC segmented RPCMEM (experimental)

This is an opt-in allocator experiment for the **FastRPC/invoke** execution backend. It does not use dspqueue to execute graphs. The original monolithic mempool remains the default.

## Configuration

Build using `-DGGML_HEXAGON_USE_MEMPOOL=ON` and rebuild/redeploy **both** `libggml-hexagon.so` and the matching `libggml-htp-v81.so` HTP skel. The IDL and `hex_tensor_desc` wire format changed; mixing old and new shared objects is unsafe.

| Variable | Purpose |
| --- | --- |
| `GGML_HEXAGON_SEGMENTED_RPCMEM=1` | Enable per-buffer registered RPCMEM segments |
| `GGML_HEXAGON_WORKSPACE_MB=256` | Size of the separate FastRPC batch/mirror workspace (64–1024 MiB; default 256) |
| `GGML_HEXAGON_MEMORY_RESERVE_MB=512` | Memory-guard reserve **after** accounting for model CPU + HTP allocations |
| `GGML_MEM_ACCOUNTING=1` | Enable allocation accounting |
| `GGML_MEM_SNAPSHOT=1` | Enable memory snapshots |
| `GGML_MEM_PREFLIGHT_ONLY=1` | Intentionally stop before allocating model weight buffers |

`GGML_HEXAGON_RPCMEM_MB` controls the **old** monolithic mempool. It is ignored when segmented mode is enabled. The new per-buffer mode uses up-to-1024-MiB mapping chunks, plus the separate workspace. On devices supporting DMA64 extended mappings, individual mappings use `FASTRPC_MAP_FD_DELAYED_EXTENDED` and 2 MiB alignment.

## First tests

1. Back up the known-good Android binaries before overwriting them.
2. Rebuild FastRPC and the DSP skel from this branch; deploy the pair.
3. Run the existing Q4 benchmark unchanged except for `GGML_HEXAGON_SEGMENTED_RPCMEM=1 GGML_HEXAGON_WORKSPACE_MB=256`; ensure the logs show `[RPC-SEG]` registration and **actual HTP execution**.
4. Run BF16 with `GGML_MEM_PREFLIGHT_ONLY=1 GGML_HEXAGON_MEMORY_RESERVE_MB=512`. A non-zero test exit may be intentional in preflight mode; inspect `[MEM-GUARD]` and `[MEM-PLAN]`.
5. Only if the preflight passes, unset `GGML_MEM_PREFLIGHT_ONLY`, set `/proc/self/oom_score_adj=800` where permitted, and test `-ngl 15 -p 8 -n 1 -r 1` with HTP enabled. Watch `IonTotalUsed`, `MemAvailable`, and `VmSwap`.
6. If a graph reports a workspace mirror or batch descriptor allocation failure, increase `GGML_HEXAGON_WORKSPACE_MB` incrementally. That increases the total RAM footprint.

To return to the original mode, unset `GGML_HEXAGON_SEGMENTED_RPCMEM` and `GGML_HEXAGON_WORKSPACE_MB`, then use the original `GGML_HEXAGON_RPCMEM_MB` sizing.

## Design

Segment 0 is the small synchronous FastRPC workspace, used for batch descriptors and host tensor mirrors. Segment IDs 1–255 are individual DMA backing buffers, registered through new scalar IDL calls and mapped on the DSP with `HAP_mmap2`; they are unmapped with `HAP_munmap2` on v81. `hex_tensor_desc` carries a segment ID and a relative offset. DSP validation rejects invalid slot/range pairs before dereferencing data. Both buffers and their mappings are freed on buffer destruction. The original `execute_batch` doorbell and execution path are retained.

Quantized weights still undergo their existing repacking on the AP. Models are split into at most 1 GiB contiguous backing chunks to avoid attempting a single >3 GiB allocation. The original monolithic path remains the default.

## Verification state

**Not yet compiled or executed on an Android/HTP device.** This branch is a source-level experiment, not a confirmed working replacement. Verify both Q4 and BF16, data correctness, mapping/unmapping during repeated loads, and comparative throughput before merging.
