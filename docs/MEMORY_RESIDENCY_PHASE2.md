# Phase 2 — resident memory and transient HTP workspace

Based on `diag/buffer-ownership-20261019`. Purely diagnostic:
**no memory-budget, placement, `-ngl`, FastRPC execution or pool allocation changes**.
Works with any supported model and GGUF format; nothing is keyed on tensor names,
layer counts or model architecture.

## Measurements

The existing `GGML_MEM_SNAPSHOT=1` path (Linux / Android) now logs:

- Process `VmRSS`, `VmHWM`, `RssAnon`, `RssFile`, `RssShmem`, `VmSwap`
  from `/proc/self/status`;
- process `Rss` and `Pss`, `Private_Clean`, `Private_Dirty`,
  `Shared_Clean`, `Shared_Dirty`, `SwapPss` when readable from
  `/proc/self/smaps_rollup`;
- system `MemTotal`, `MemAvailable`, `IonTotalUsed`,
  `IonTotalCache` from `/proc/meminfo` when provided by the kernel;
- **system-wide** DMA-BUF sum from `/sys/kernel/dmabuf/buffers` when
  accessible. This includes other processes and must **never** be treated
  as a per-process usage metric.

The existing `GGML_MEM_ACCOUNTING=1` output additionally reports:

- `buft_id`: distinction between genuinely separate buffer types, even
  when their human-readable `type` name is identical (e.g. CPU-family
  implementations may both advertise `CPU`);
- a per-RPCMEM backing count of newly allocated *persistent weight mirrors*
  (cumulative allocations, **not current live**), the *peak temporary
  mirrors per graph*, and the *peak batch-descriptor memory per graph*.
  Mirrors and descriptors are portions of the existing RPCMEM allocation,
  **not extra physical allocations**.
- snapshots at `context_ready`, `before_context_destroy`,
  `after_context_destroy`, in addition to model allocation/loading and
  the existing process-memory snapshots.

The `bump_hwm` value is a maximal offset in the pool, NOT a direct sum of
mirror/descriptor/weight bytes. It includes alignment and holes. Peak of
child regions and peak bump offset are also different metrics.

**Do not subtract accounted allocations from process RSS as though both
measure the same physical quantity.** Backends can map DMA-BUF pages into
RSS, mmap-backed file pages may be only partially resident, and some buffers
may be resident outside the process' normal RSS. The important first analysis
is the *change across stages*, correlated with alloc/free lifetimes.

Values of `-1` mean the relevant OS interface or metric was unavailable.

## Run on Android

Checkout and rebuild this branch normally. Use the previous successful Q4
run to compare performance, then collect RSS/PSS + ION during the entire load
and compute lifecycle:

```bash
adb shell 'cd /data/local/tmp &&
  export LD_LIBRARY_PATH=/data/local/tmp &&
  export GGML_HEXAGON_DEVICES=1 &&
  export GGML_HEXAGON_RPCMEM_MB=2048 &&
  export GGML_MEM_ACCOUNTING=1 &&
  export GGML_MEM_SNAPSHOT=1 &&
  ./llama-bench -v -t 8 -ngl 15 -fa 1 -p 32 -n 8 -r 1 \
    -m /sdcard/Spark-X2.5-4B-Q4_K_M.gguf' > mem-residency.log 2>&1
```

Log processing (on the development machine):

```sh
python3 tests/check-mem-account-log.py --self-test
python3 tests/check-mem-residency-log.py --self-test
python3 tests/check-mem-account-log.py mem-residency.log
python3 tests/check-mem-residency-log.py mem-residency.log > memory-stages.tsv
```

Observe which memory delta occurs at these stages:

1. `before_rpcmem:HTP0`, `after_rpcmem_alloc:HTP0`,
   `after_rpcmem_register:HTP0` — changes during pool backing creation
   and FastRPC mapping.
2. `before_model_buffers` vs `after_model_buffers` — allocating CPU/HTP
   model buffer storage.
3. `before_load_all_data:...` vs `after_load_all_data:...` — actually
   loading/repacking bytes, page residency and transient staging.
4. `before_context_init` vs `after_context_init`, plus
   `[MEM-ACCOUNT] stage=context_ready` — KV and compute buffers.
5. `before_context_destroy`, `after_context_destroy` — context lifetime.
6. `before_rpcmem_free:HTP0` / `after_rpcmem_free:HTP0` — pool release.

The two benchmark phases may construct/destroy their contexts separately.
Match adjacent snapshots within the **same PID**, stage sequence and timestamps,
not just stage name across two independent executions.

No CPU process is killed, no RSS reclamation is forced, and no memory guard
is changed by this phase.

## Validation and outstanding gaps

- Both parsing scripts have self-contained `--self-test` modes.
- `tests/check-mem-account-log.py` still validates the backing/child
  accounting relationship; the new `buft_id` and additional backing fields
  do not change its accepted prefix.
- GPU memory and mmap page ownership are **not** fully attributed.
- Staging allocations that bypass ggml's buffer API are not individually
  tracked. The process `VmHWM` captures process RSS peaks but cannot
  attribute those peaks to an allocation call site.
- Persistent HTP mirror *allocation totals* are not equal to mirror
  *live bytes* after cache invalidation or model unload.
- `IonTotalUsed` and DMA-BUF heap information may be global and may
  include data not owned by this process. Avoid double counting.
- No build/device execution can be asserted from repository-only checks:
  build and run must be verified on the target Android toolchain.

Next phase, once the deltas are known: minimally instrument any remaining
host staging/allocation call sites which are confirmed to dominate the peak,
instead of introducing a generic memory allocator or model-specific rules.
