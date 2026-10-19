# BF16 physical memory attribution (diagnostic phase 3)

Branch `diag/bf16-memory-attribution-20261019`, based on
`diag/memory-residency-20261019`.

Goal: distinguish **allocation of one large RPCMEM backing**, its mappings,
CPU/CPU_REPACK host buffers, and preflight estimates in the same process.
We do **not** disable or relax the memory guard and do **not** change
execution, repack formats, quantization, thread settings, or offload placement.
The instrumentation is GGML buffer-type based and works for any model.

## Three corrected blind spots

1. **RPCMEM lifetime snapshots are now sent to stderr**, captured by the same
   `adb shell ... > log 2>&1` file as llama-model logs. Previously, the
   Hexagon backend sent these through `GGMLHEXAGON_LOG_ALWAYS`, which on
   Android may log to logcat rather than stdout/stderr. We now have:
   `before_rpcmem:HTP0`, `after_rpcmem_alloc:HTP0`,
   `after_rpcmem_register:HTP0`, `before_rpcmem_free:HTP0`,
   and `after_rpcmem_free:HTP0` if the backend reaches teardown.
2. **The requested buffer type is preserved** when a REPACK allocation
   delegates to a plain CPU buffer allocator. Lines
   `[MEM-ACCOUNT-BUFT] type=CPU_REPACK ... actual_type=CPU` now
   distinguish the caller's intent from the concrete backing type, rather
   than grouping all CPU allocations together. A fallback to CPU physical
   storage is not by itself a duplicate of the same weights: different
   buffer-type groups may own different tensors.
3. **Preflight reports sizes per requested buffer type**, before allocating
   model-weight buffers: `[MEM-PLAN]` lines, followed by
   `memory_preflight_before_guard` and optionally
   `memory_guard_reject_cpu` / `memory_guard_reject_htp` snapshots.
   These planned bytes are storage *requirements*, not new resident pages.
   The existing guard remains unchanged.

`[MEM-RPCMEM-FD]` captures the actual RPCMEM fd, inode, fstat size and
optional matching `/sys/kernel/dmabuf/buffers/<inode>/size` if accessible.
No permission escalation is used. A missing sysfs value is `-1` and does
not imply an allocation failure. Inode/fd identity is useful for identifying
the backing but does **not** prove charged physical-page ownership.

## Controlled Android experiments

Build this branch and run a known-good Q4 first:

```sh
adb shell 'cd /data/local/tmp &&
  export LD_LIBRARY_PATH=/data/local/tmp &&
  export GGML_HEXAGON_DEVICES=1 &&
  export GGML_HEXAGON_RPCMEM_MB=2048 &&
  export GGML_MEM_ACCOUNTING=1 &&
  export GGML_MEM_SNAPSHOT=1 &&
  ./llama-bench -v -t 8 -ngl 15 -fa 1 -p 32 -n 8 -r 1 \
    -m /sdcard/Spark-X2.5-4B-Q4_K_M.gguf   > q4-attribution.log 2>&1'
adb pull /data/local/tmp/q4-attribution.log .
```

Then run BF16 with the **unchanged** safety reserve (2048 MiB). This is
intentionally a guarded preflight: with the previously observed RAM
conditions it is expected to **reject before model-weight allocations**.
No BF16 tensors are loaded if the guard rejects.

```sh
adb shell 'cd /data/local/tmp &&
  export LD_LIBRARY_PATH=/data/local/tmp &&
  export GGML_HEXAGON_DEVICES=1 &&
  export GGML_HEXAGON_RPCMEM_MB=3712 &&
  export GGML_HEXAGON_MEMORY_RESERVE_MB=2048 &&
  export GGML_MEM_ACCOUNTING=1 &&
  export GGML_MEM_SNAPSHOT=1 &&
  ./llama-bench -v -t 8 -ngl 15 -fa 1 -p 32 -n 8 -r 1 \
    -m /sdcard/Spark-X2.5-4B.gguf   > bf16-preflight.log 2>&1'
adb pull /data/local/tmp/bf16-preflight.log .
```

**Do not set memory reserve to zero or bypass the guard merely to obtain a
successful BF16 run**. A host-weight allocation greater than available
capacity can induce Android LMK and terminate other apps.

Analyze both logs:

```sh
python3 tests/check-bf16-attribution-log.py --self-test
python3 tests/check-bf16-attribution-log.py q4-attribution.log
python3 tests/check-bf16-attribution-log.py bf16-preflight.log
python3 tests/check-mem-account-log.py q4-attribution.log
```

The BF16 log may not contain any `model_buffers_allocated` or
`model_weights_loaded` accounting — that is expected on guarded rejection.

## Interpretation and falsifiable questions

**Does allocating RPCMEM actually consume system-wide ION?**
Compare `IonTotalUsedKiB`, `IonTotalCacheKiB`, `MemAvailableKiB`
between `before_rpcmem` and `after_rpcmem_alloc` **in one process**.
An increase by approximately the RPCMEM backing size supports accounting
for the reserved physical allocation even before weights are copied. A
different delta does not establish a free zero-cost pool: cache transfers,
lazy page provisioning, driver charges and concurrent applications must be
considered.

**Does DSP mapping register another physical copy?**
Compare the same counters between `after_rpcmem_alloc` and
`after_rpcmem_register`. A change indicates additional memory charge
or other activity in that interval, not necessarily copied model weights.
A mapping is not inherently a second physical backing.

**Are CPU and CPU_REPACK two copies of the same tensors?**
Check `[MEM-PLAN]` and `[MEM-ACCOUNT-BUFT]` requested vs actual origin.
Distinct buffer allocations alone do not prove duplicated tensor content.
That stronger claim would require per-tensor provenance or a copy
source-to-destination identity trace. This phase does NOT make that claim.

**What is BF16's real failure point?**
If `memory_guard_reject_cpu` occurs, model-weight allocation did not
execute. We have learned the planned host storage and how much RAM/ION
existed *after* creating the HTP backing, but **not** whether allocating
all BF16 host weights would in fact succeed. There is no need to risk an
LMK to distinguish this.

**How much of the pool is actually used?**
Use `[MEM-ACCOUNT-BACKING]` child_live, bump_hwm and
mirror/descriptor peaks from the successful Q4 run. RPCMEM allocation
capacity and HTP virtual mapping are separate issues.

## Limitations

- `IonTotalUsed`, `IonTotalCache` and DMA-BUF global size are kernel/
  system-wide measurements and may be affected by unrelated processes.
- `VmRSS`, `PSS`, anonymous resident pages, mapped GGUF, and dma-buf
  backing are *different views* of the same physical memory; **do not sum
  them**.
- The system might expose no per-buffer dma-buf file information to an
  unprivileged app. Record missing metrics rather than inventing zeros.
- The fd of a pool is not a portable way to infer how many pages are pinned
  or reclaimable inside the DSP domain.
- No actual Android build or device run is asserted by this change; only
  the source patches and parser are provided for target-side validation.
