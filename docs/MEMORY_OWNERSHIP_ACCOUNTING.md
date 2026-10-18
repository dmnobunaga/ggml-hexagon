# Backend memory ownership accounting (phase 1)

Branch: `diag/buffer-ownership-20261019` (based on
`safety/rpcmem-budget-20261019`).

## Purpose

Passive, **model-agnostic** allocation and ownership diagnostics for ggml.
It does not inspect model architecture, layer names, `-ngl`, parameter count,
quantization, or model-specific graph patterns. It **does not change allocation,
offloading, memory policy, or the Hexagon execution path**.

The accounting deliberately distinguishes:

- A **physical backing** such as one FastRPC `rpcmem_alloc2()` allocation.
  Count backing capacity **once**, independently of its address-space mappings.
- A **backend buffer** representing an owned allocation (e.g. CPU, CPU_REPACK)
  or a non-owning view/suballocation (e.g. HTP_REPACK in an RPCMEM pool).
- **Logical child regions** inside a backing, counted as occupied space, not as
  extra physical allocation.
- **FastRPC mapping registrations** (`fastrpc_mmap` /
  `fastrpc_munmap`) counted separately. A delayed registration is not proof
  that pages are currently mapped in the DSP user address space.
- **Bump-pointer high-water** across persistent regions and transient graph
  mirrors/descriptors. The high-water offset is not the sum of live regions
  and may include padding, holes and transient allocations.
- **Backend tensor writes** through `ggml_backend_tensor_set`, a useful
  indicator of initial transfers and format-dependent repack. The byte count
  is write traffic, **not** an additional live allocation or a reliable number
  of distinct copied bytes.

Physical capacity, RSS, pinned physical pages and reclaimable ION cache are
different quantities. `physical_live` reports the sum of tracked owned
backend buffer capacities plus unique registered RPCMEM backings. It is an
**allocation-capacity estimate**, not a kernel-level physical RSS reading.
mmap-backed GGUF file pages, direct mallocs, driver allocations, DSP VTCM,
IOVA/VA mappings and uninstrumented GPU driver allocations may be outside
this accounting. Not every generic backend buffer with no `get_base` can be
classified as an owned physical allocation.

A backend buffer's `backing_id` is not a duplicate allocation. Two buffers
with the same backing represent potentially distinct logical regions but
share one counted physical allocation. A `buffer_id` is a lifecycle ID
(not derived from a model's tensor name).

## Enable

Only enabled when `GGML_MEM_ACCOUNTING=1`. No detailed event tracing or
mutex/aggregation overhead occurs when disabled beyond inexpensive flag
checks. For lifecycle events additionally set `GGML_MEM_ACCOUNT_VERBOSE=1`;
leave this disabled for benchmarking, especially with many tensor writes.

```sh
adb shell 'cd /data/local/tmp &&
  export LD_LIBRARY_PATH=/data/local/tmp &&
  export GGML_HEXAGON_DEVICES=1 &&
  export GGML_HEXAGON_RPCMEM_MB=2048 &&
  export GGML_MEM_ACCOUNTING=1 &&
  ./llama-bench -v -t 8 -ngl 15 -fa 1 -p 32 -n 8 -r 1 \
    -m /sdcard/Spark-X2.5-4B-Q4_K_M.gguf' > mem-account.log 2>&1
```

Key output prefixes:

```text
[MEM-ACCOUNT] stage=model_buffers_allocated physical_live=... physical_peak=... pool_live=...
[MEM-ACCOUNT-BUFT] stage=model_buffers_allocated type=CPU_REPACK buffers=... logical=... physical_unique=... marked_weights=...
[MEM-ACCOUNT-BUFT] stage=model_buffers_allocated type=hexagon-...-REPACK buffers=... logical=... physical_unique=0.00MiB ...
[MEM-ACCOUNT-BACKING] stage=model_buffers_allocated id=... type=HTP_RPCMEM allocated=... children_live=... children_peak=... bump_hwm=... mapping_registrations=...
[MEM-ACCOUNT] stage=model_weights_loaded ...
[MEM-ACCOUNT] stage=before_rpcmem_free ...
```

The **zero** `physical_unique` for HTP suballocations is intentional:
their backing is already counted as `HTP_RPCMEM`. This prevents counting
the same DRAM allocation twice.

Run the log consistency checker:

```sh
python3 tests/check-mem-account-log.py --self-test
python3 tests/check-mem-account-log.py mem-account.log
```

The checker asserts that each snapshot's `physical_live` matches
`sum(unique backend-owned buffers) + sum(registered physical backings)`
(up to printed rounding), that child usage does not exceed its backing,
and that physical peaks are consistent. It cannot certify real RSS.

## Verification order

1. **Q4 with `-ngl 15`, RPCMEM target 2048 MiB**: verify that
   `CPU_REPACK`, `HTP_REPACK`, and pool child sizes are correct, and that
   benchmark performance is unaffected when verbose tracing is off.
2. **Q4 with a different `-ngl`**: verify invariant calculations without
   any model-specific instrumentation.
3. **Optional BF16 preflight-only**: if the existing RAM guard rejects,
   examine `before_rpcmem_free` and the memory snapshots. **Do not lower
   the RAM guard simply to obtain a successful BF16 run**.

This phase does NOT implement the DSP address-space window manager or
changes inspired by dspqueue. Address-space mapping lifecycle and the
physical owner of weight data must be studied independently. In particular,
neither `-ngl 99` nor an HTP version number should imply a 4 GiB physical
allocation in a future memory planner.

## Next instrumentation candidates

- Track transient mirror/descriptor regions as independent child **regions**
  with lifetimes, not only bump-pointer high-water.
- Instrument CPU_REPACK's format transformation and staging workspace
  allocations (separately from tensor writes).
- Correlate alloc/free events with `/proc/self/smaps_rollup`,
  `/proc/meminfo`, and ION/dma-buf heap metrics to estimate actual
  resident/charged memory and temporary duplication.
- Introduce explicit backing identity for externally sourced mmap and
  backend buffers, rather than inferring physical ownership from interfaces.

These require additional instrumentation and/or kernel-side evidence;
the current counters are intentionally not labeled as exact physical RAM.
