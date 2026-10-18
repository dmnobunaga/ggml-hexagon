# Android RPCMEM budget and preflight guard

This branch extends the diagnostic memory snapshots branch. It preserves the
original Hexagon pool-size candidate list when no override is set.

## Controls

- `GGML_HEXAGON_RPCMEM_MB=2048`: request one exact **target** of 2048 MiB.
  Eight MiB are reserved for backend bookkeeping; the allocated FastRPC buffer
  and usable pool are 2040 MiB. The value must be a multiple of 64 between
  256 and 4032. An invalid value or allocation failure aborts initialization;
  it does not silently return to the legacy 3960-MiB allocation.
- `GGML_HEXAGON_MEMORY_RESERVE_MB=1536`: minimum Android
  `MemAvailable` headroom to retain after planned CPU-side weight buffers.
  Defaults to 1536 MiB when Hexagon buffers are present. Set a different
  nonnegative integer (0..8192) to tune the budget. Values too small can
  cause Android LMK pressure: do not lower this simply to force BF16 through.
- `GGML_MEM_SNAPSHOT=1`: existing optional memory snapshots.

When loading on Android with Hexagon buffers:

1. Predict all CPU-side buffers, **including CPU_REPACK**, using
   `ggml_backend_alloc_ctx_tensors_from_buft_size`.
2. Predict aggregate weight allocations for each HTP backend using the
   same allocator's sizing function.
3. Reject if HTP model weights plus a **128 MiB temporary/compute allowance**
   cannot fit within the allocated RPCMEM pool.
4. Read `MemAvailable` after RPCMEM initialization and before model weight
   allocations. Reject if `CPU weights + configured reserve > MemAvailable`.
   If reading `MemAvailable` fails, abort instead of assuming it is safe.

Output goes to the usual llama.cpp model log as `[MEM-GUARD]`.

**Important limitations:** the guard is a conservative preflight for weights,
not an exact forecast of peak system memory. Android process pressure, ION
behavior, KV/cache/compute requirements for large contexts and other concurrent
allocations may still exceed these reserves. A buffer that would exceed HTP
mempool now fails **without creating a CPU fallback allocation**. This may make
previously runnable models fail fast instead of using CPU fallback. CPU
buffers that use file-backed mmap can be overestimated by the preflight.

## Safe verification (Q4_K_M, not BF16)

First test pool configuration and guard rejection with a deliberately small
pool. This allocates only a 1016-MiB RPCMEM buffer and should stop **before
weight allocation**:

```sh
adb shell 'cd /data/local/tmp &&
  export LD_LIBRARY_PATH=/data/local/tmp GGML_HEXAGON_DEVICES=1 &&
  export GGML_HEXAGON_RPCMEM_MB=1024 GGML_MEM_SNAPSHOT=1 &&
  ./llama-bench -v -ngl 15 -p 32 -n 8 -r 1 \
    -m /sdcard/Spark-X2.5-4B-Q4_K_M.gguf'
```

Expected: `[MEM-GUARD] HTP pool too small`. Exit failure is intentional.

Then test a 2048-MiB target / 2040-MiB usable pool:

```sh
adb shell 'cd /data/local/tmp &&
  export LD_LIBRARY_PATH=/data/local/tmp GGML_HEXAGON_DEVICES=1 &&
  export GGML_HEXAGON_RPCMEM_MB=2048 GGML_MEM_SNAPSHOT=1 &&
  ./llama-bench -v -t 6 --poll 1000 -ngl 15 -fa 1 \
    -p 32 -n 8 -r 1 -m /sdcard/Spark-X2.5-4B-Q4_K_M.gguf'
```

Expected: `[MEM-GUARD] PASS`, `rpc mempool: 2040 MiB allocated`,
and a successful benchmark. Capture `adb logcat` for the ION measurements.

**Do not retest BF16 until these guards and the smaller pool are confirmed
to work on the real device.** BF16 -ngl 15 requires ~3441 MiB in HTP plus
~5042 MiB CPU weights; a 2048-MiB pool is not big enough.
