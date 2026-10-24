# HTP + OpenCL Android hybrid experiment

Experimental branch: `experiment/htp-opencl-hybrid`. Based on `experiment/htp-vulkan-hybrid`, with isolated build and deployment paths.

## Scope

- Retains existing FastRPC, RPCMEM, DSP VA mapping, and HTP adaptive reservation implementation.
- Routes upper decoder blocks and optionally output to HTP; routes decoder blocks below the HTP offload boundary to GPUOpenCL instead of CPU.
- Selects OpenCL only when `GGML_HEXAGON_HYBRID_OPENCL=1`; Vulkan remains available under its existing separate environment flag.
- Protects the known-good DSP-only build and existing Vulkan hybrid; builds to `out/ggmlopencl-hybrid-android` and deploys to `/data/local/tmp/ggml-opencl-hybrid` by default.
- `llama-bench` device-list syntax: `--device HTP0/GPUOpenCL` (slash joins devices in **one** benchmark).
- `llama-cli` device-list syntax: `--device HTP0,GPUOpenCL` (comma).

## Build and deploy

```bash
git switch experiment/htp-opencl-hybrid
./scripts/build-run-ggmlopencl-hybrid-android.sh build
./scripts/build-run-ggmlopencl-hybrid-android.sh devices
```

The wrapper enables `GGML_OPENCL=ON`, Adreno kernels, and embedded OpenCL kernels; uses the OpenCL headers / ICD library already prepared by the original Android script. Run `update_fastrpc_libs` to redeploy a built runtime without rebuilding:

```bash
./scripts/build-run-ggmlopencl-hybrid-android.sh update_fastrpc_libs
```

Confirm `GPUOpenCL` is listed. If it is not, verify that the device exposes the vendor OpenCL implementation, e.g. `adb shell 'ls -l /vendor/lib64/libOpenCL.so'`. The locally built cross-compile ICD library is **not** proof that the phone's OpenCL GPU driver is available.

## Smoke test — Gemma E2B Q4_0

Use PowerShell:

```powershell
$cmd = 'cd /data/local/tmp/ggml-opencl-hybrid; export LD_LIBRARY_PATH=/data/local/tmp/ggml-opencl-hybrid:/vendor/lib64 ADSP_LIBRARY_PATH=/data/local/tmp/ggml-opencl-hybrid GGML_HEXAGON_SEGMENTED_RPCMEM=1 GGML_HEXAGON_ADAPTIVE_OFFLOAD=1 GGML_HEXAGON_HYBRID_OPENCL=1 GGML_HEXAGON_SEGMENT_MB=1024 GGML_HEXAGON_MEMORY_RESERVE_MB=1536 GGML_HEXAGON_DSP_VA_WINDOW_MB=3072; unset GGML_HEXAGON_HYBRID_VULKAN GGML_HEXAGON_TEST_FORCE_DESCENDING GGML_MEM_GUARD_WARN_ONLY GGML_MEM_PREFLIGHT_ONLY; echo 800 > /proc/self/oom_score_adj; ./llama-bench --device HTP0/GPUOpenCL -t 6 --poll 1000 -ngl 24 -fa 1 --ubatch-size 128 -p 128 -n 16 -r 1 -lm mmap -v -m /sdcard/gemma-4-E2B-it-Q4_0.gguf'
adb shell $cmd 2>&1 | Tee-Object gemma-htp-opencl.log
```

## Spark BF16 check

```powershell
$cmd = 'cd /data/local/tmp/ggml-opencl-hybrid; export LD_LIBRARY_PATH=/data/local/tmp/ggml-opencl-hybrid:/vendor/lib64 ADSP_LIBRARY_PATH=/data/local/tmp/ggml-opencl-hybrid GGML_HEXAGON_SEGMENTED_RPCMEM=1 GGML_HEXAGON_ADAPTIVE_OFFLOAD=1 GGML_HEXAGON_HYBRID_OPENCL=1 GGML_HEXAGON_SEGMENT_MB=1024 GGML_HEXAGON_MEMORY_RESERVE_MB=1536 GGML_HEXAGON_DSP_VA_WINDOW_MB=3072; unset GGML_HEXAGON_HYBRID_VULKAN GGML_HEXAGON_TEST_FORCE_DESCENDING GGML_MEM_GUARD_WARN_ONLY GGML_MEM_PREFLIGHT_ONLY; echo 800 > /proc/self/oom_score_adj; ./llama-bench --device HTP0/GPUOpenCL -t 6 --poll 1000 -ngl 31 -fa 1 --ubatch-size 128 -p 32 -n 8 -r 1 -lm mmap -v -m /sdcard/Spark-X2.5-4B.gguf'
adb shell $cmd 2>&1 | Tee-Object spark-htp-opencl.log
```

OpenCL currently converts BF16 tensors to FP16 **on upload**, retaining two bytes per weight, but this is not numerically lossless. Some operations/tensors may still fall back to CPU, and the current HTP planner does not jointly budget OpenCL allocations before choosing HTP offload. Do not weaken the RAM guard for the first test.

These commands have not been run on a physical phone by the authoring environment.
