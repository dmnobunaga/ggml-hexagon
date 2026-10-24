# Experimental HTP + Vulkan hybrid inference

**Status:** first implementation, not yet built or device-tested. Opt-in feature in `experiment/htp-vulkan-hybrid`; stable milestone `a7b4a6dc0c22701918999d721aeea3eb362eb33d` unchanged.

## Placement and limitations

The standard `-ngl` count includes an optional output layer. For Spark's 36 repeating blocks with `effective_ngl=35`, planned placement is:
- Vulkan: blocks 0–1.
- HTP: blocks 2–35 and output.
- CPU: input/embeddings and unsupported ops.

This is **not** 35 transformer blocks on HTP. We preserve the descending HTP weight layout from the adaptive planner; tensor-level fallbacks and graph execution placement require separate confirmation.

With `GGML_HEXAGON_HYBRID_VULKAN=1`, select exactly **HTP first, Vulkan second**, with `-dev` or `--device` (autodetection may skip Adreno because HTP announces itself as GPU and Vulkan may be IGPU). The existing transactional HTP reservation probes only the HTP weight chunks.

## Android cross-build and isolated deployment

On a Linux build host, NDK, Hexagon SDK and LunarG SDK are downloaded or reused by the existing FastRPC setup workflow:

```bash
# Reuses the existing NDK / Hexagon SDK download and deployment pipeline:
./scripts/build-run-ggmlhybrid-android.sh build
# After rebuilding manually, redeploy without recompiling:
./scripts/build-run-ggmlhybrid-android.sh update_fastrpc_libs
./scripts/build-run-ggmlhybrid-android.sh devices
```

The wrapper calls the existing `build-run-ggmlhexagon-android.sh` for NDK/Hexagon setup, FastRPC skel generation and synchronized AP deployment; it downloads the same LunarG SDK as the Vulkan-only script and adds `-DGGML_VULKAN=ON`. The `update_fastrpc_libs` alias uploads matched AP libraries, Vulkan backend, DSP skeletons and executables from the hybrid `bin/` without recompiling or using the old `out/ab-test` snapshots. It does not overwrite the known-good `out/ggmlhexagon-android`, `out/ab-test`, or `/data/local/tmp` files unless `GGML_REMOTE_PATH` is explicitly overridden. Hybrid artifacts use `out/ggmlhybrid-android` and `/data/local/tmp/ggml-hybrid`.

Verify both devices before loading a model:

```bash
./scripts/build-run-ggmlhybrid-android.sh devices
```

**Important: device-list syntax differs across executables.** In `llama-bench`, a comma separates *independent benchmark configurations* while a slash combines devices into one configuration. Use `--device HTP0/Vulkan0` (not `HTP0,Vulkan0`). In `llama-cli` and other executables using `common/arg.cpp`, the device-list delimiter is a comma: `--device HTP0,Vulkan0`.

For the first smoke test, replace `HTP0/Vulkan0` with the device names actually printed by `--list-devices`:

```bash
adb shell '
cd /data/local/tmp/ggml-hybrid
export LD_LIBRARY_PATH=/data/local/tmp/ggml-hybrid
export ADSP_LIBRARY_PATH=/data/local/tmp/ggml-hybrid
export GGML_HEXAGON_SEGMENTED_RPCMEM=1
export GGML_HEXAGON_ADAPTIVE_OFFLOAD=1
export GGML_HEXAGON_SEGMENT_MB=1024
export GGML_HEXAGON_MEMORY_RESERVE_MB=1536
export GGML_HEXAGON_DSP_VA_WINDOW_MB=3072
export GGML_HEXAGON_HYBRID_VULKAN=1
export GGML_SCHED_DEBUG=2
./llama-bench -dev HTP0/Vulkan0 -ngl 35 -t 6 --poll 1000 -fa 1 \
  --ubatch-size 128 -p 16 -n 4 -r 1 -m /sdcard/Spark-X2.5-4B.gguf
' 2>&1 | tee hybrid-spark-35.log
```

Start with fixed `-ngl 35`, then try `-ngl 99` (adaptive descending) and `-ngl 22` (larger Vulkan remainder).

## Acceptance checks

1. Both HTP and Vulkan initialize in one process.
2. Logs `[HTP-HYBRID]` report layer[0..1]=Vulkan, layer[2..35]=HTP, output=HTP. This proves **planned assignment**, not actual GPU computation.
3. `[MEM-GUARD]` reports `planned_gpu_other`, retains the RAM reserve, and the DSP VA mapping window is still bounded.
4. `GGML_SCHED_DEBUG=2` proves that Vulkan actually executes `MUL_MAT` nodes for overflow blocks. Inspect any CPU `MUL_MAT` nodes; CPU-fallback heavy matmul is not yet forbidden.
5. Compare deterministic short-generation against reference, then run PP/TG and Gemma full-HTP regression.

## Boundaries

- `src/llama.cpp`: two-device validation, existing exact-layout transactional planner preserved.
- `src/llama-model.cpp`: explicit overflow Vulkan placement, plus conservative GPU/other weight accounting in Android memory guard.
- `ggml/src/ggml-backend.cpp`: **unchanged**; standard GGML graph split and tensor copies used.
- Opt-in only; DSP-only behavior unaffected when `GGML_HEXAGON_HYBRID_VULKAN` is unset.

Guard accounts model weights, but not every transient compute/KV allocation or GPU driver overhead. Keep `GGML_HEXAGON_MEMORY_RESERVE_MB=1536`; do not disable the guard. Driver cache coherence and Vulkan BF16 compatibility must be validated on-device.
