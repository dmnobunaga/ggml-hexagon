# Android memory snapshots (opt-in)

The diagnostic branch adds a lightweight, read-only snapshot at RPCMEM lifecycle,
large HTP allocations, model-buffer allocation, each weight-upload group, large
synchronous staging copies, and context initialization.

## Enable

Set `GGML_MEM_SNAPSHOT=1` in the **llama-bench process** and use `-v` to
include llama.cpp's INFO-level snapshots. HTP-side snapshots also appear in
Android Logcat under the `ggml-hexagon` tag. When the variable is unset,
no snapshot files are read.

Example inside `adb shell` (adjust binary and model path):

```sh
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp
export GGML_HEXAGON_DEVICES=1
export GGML_MEM_SNAPSHOT=1
./llama-bench -v -t 6 -ngl 99 -fa 1 -p 32 -n 8 -r 1 -m /sdcard/Spark-X2.5-4B-Q4_K_M.gguf
```

To verify the diagnostic code without allocating BF16 weights, use the
**existing local** `GGML_HEXAGON_ALLOC_DRY_RUN=1` implementation, if available,
on the BF16 model; this is not part of this branch.

## Markers

- `before_rpcmem:HTP0`, `after_rpcmem_alloc:HTP0`, `after_rpcmem_register:HTP0`
- `before_model_buffers`, `after_model_buffer:<type>`, `after_model_buffers`
- `before_load_all_data`, `before_load_all_data:<type>`,
  `after_load_all_data:<type>`, `after_load_all_data`
- `before_staging:<tensor>`, `after_staging_alloc:<tensor>`,
  `after_staging_read:<tensor>`, `after_staging_upload:<tensor>`
  for synchronous staged tensors of 128 MiB or larger
- `before_context_init`, `after_context_init`
- `before_large_htp_alloc`, `after_large_htp_alloc` for requests >= 256 MiB
- `before_cpu_fallback`, `after_cpu_fallback` on the oversized-HTP allocation path
- `before_rpcmem_free`, `after_rpcmem_free`

Each `[MEM]` entry carries a monotonic `t_ms`, PID, memory metrics in KiB,
and optional system-wide DMA-BUF metrics. `-1` means the kernel interface
was unavailable or unreadable. `dmaBufGlobalKiB` is **system-wide**, **not**
memory owned by llama-bench; it must never be added directly to `VmRSSKiB`.
`PssKiB` is from `/proc/self/smaps_rollup`, if available.

For context near an Android low-memory kill:

```sh
adb logcat -b all -v threadtime -s 'ggml-hexagon:I' 'lowmemorykiller:I' 'lmkd:I' 'ActivityManager:W' '*:S'
```

## Caveats

Memory snapshots **are not an OOM guard**. They intentionally do not change
allocation decisions. Test with a working quantized model first; do not use
the original 7.66 GiB BF16 `-ngl 99` workload without a separately validated
safety guard. The 3960 MiB RPCMEM pool may be initially lazy-backed, and
`MemAvailable` / RSS are snapshots, not a precise peak-residency measure.
Buffer lifetime peaks between markers may still be missed.
