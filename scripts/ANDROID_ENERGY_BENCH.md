# Android battery-energy benchmark and repeat-init stress test

This fork adds `--energy` to `llama-bench` and a device-side loop, `scripts/android-energy-stress.sh`.

## What is measured

- `energy_mwh`: mean **device battery energy** consumed by one timed PP/TG repetition. Model loading, warmup, and `llama-bench` startup are **not included**.
- `energy_w`: average device battery power (watts) over the timed repetitions with valid readings.
- `energy_mwh_per_1k_tokens`: battery energy normalized to 1,000 prompt + generated tokens.
- `energy_source`: `battery_energy_now` (preferred), `battery_charge_counter` (+ mean battery voltage), `battery_current_integral` (sampled ~100 ms), or `unavailable`.

All fields are shown only with `--energy`, including in `-o csv` and `-o json`. On non-Android platforms the readings are unavailable. If counters are unreadable, missing, unchanged at their reporting resolution, or the phone is charging, energy values are left empty (JSON `null`) rather than invented.

**This is battery discharge for the whole device, not isolated DSP/CPU power.** Screen, radios, thermal behavior, other processes, and the fuel gauge influence readings. It is not an instantaneous rail measurement or a hardware power profiler. For comparisons, unplug the charger, turn the screen off, keep temperature/state of charge similar, leave radios/background work unchanged, and repeat/reorder the tests. Short tests may be dominated by fuel gauge quantization; use more tokens or `-r 5`.

## Single benchmark on the device

Build and deploy the fork's Android `llama-bench` and runtime libraries normally. This example uses the repository's default `/data/local/tmp` deployment:

```bash
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench --energy -o csv -r 5 -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200 -n 128 -m /sdcard/gemma-4-E2B-it-Q4_0.gguf' > htp-energy.csv
```

CPU baseline (same model and workload):

```bash
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench --energy -o csv -r 5 -t 6 -ngl 0 -fa 1 --ubatch-size 1024 -p 200 -n 128 -m /sdcard/gemma-4-E2B-it-Q4_0.gguf' > cpu-energy.csv
```

For FastRPC vs dspqueue, deploy each backend's corresponding libraries and DSP skeleton, or stage independent runtime directories (`/data/local/tmp/fastrpc` and `/data/local/tmp/dspqueue`) containing the matching binary, libraries, and runtime files. Point `LD_LIBRARY_PATH` at the matching directory, and use the same model, quantization, and test flags. Record which runtime/commit was deployed; `-ngl 99` alone does **not** select between FastRPC and dspqueue implementations.

## Device-side Spark-4B stress test

The upstream project's model alias `spark-4b` maps to `/sdcard/Spark-X2.5-4B.gguf` (large, approximately 8 GB). Place it on the phone first.

```bash
adb push scripts/android-energy-stress.sh /data/local/tmp/
adb shell 'sh /data/local/tmp/android-energy-stress.sh 5 /sdcard/Spark-X2.5-4B.gguf /data/local/tmp /data/local/tmp/energy-test-5 fastrpc'
adb pull /data/local/tmp/energy-test-5 ./energy-test-5
```

Once five runs are stable, increase to 20:

```bash
adb shell 'STRESS_DELAY=30 sh /data/local/tmp/android-energy-stress.sh 20 /sdcard/Spark-X2.5-4B.gguf /data/local/tmp /data/local/tmp/energy-test-20 fastrpc'
adb pull /data/local/tmp/energy-test-20 ./energy-test-20
```

Each iteration starts a **fresh `llama-bench` process** (`-r 1`, `-p 200`, `-n 32`) and takes pre/post memory, memory PSI, battery temperature, and fuel-gauge snapshots. The script writes one per-run CSV, one stderr log, and a `summary.csv`, with a default battery temperature stop threshold of 45°C (raw `temp` typically in deci°C). Set `STRESS_MAX_TEMP_DECIC` for a different threshold, `STRESS_NGL=0` for a CPU baseline, or `STRESS_PP`/`STRESS_TG` to change workload. The output directory must be writable by `adb shell`.

Because the loop is **on device**, the ADB connection is not involved between repetitions. If the device reboots, the shell process itself is terminated; previously flushed logs remain on the phone. After a reboot check `adb shell getprop sys.boot.reason`, `adb shell dumpsys dropbox`, `adb logcat -b all -d`, and pstore when accessible. A reboot or LMK cannot be conclusively identified from a missing ADB connection alone.

To verify that the Hexagon backend actually ran, inspect each run's stderr and CSV: `-ngl 99` alone is not proof if backend selection/offload fails. For reproducible system comparisons, collect a control without multiple probes and a probe-enabled build, testing both under comparable thermal and memory conditions.

## Interpretation

Battery energy per 1,000 tokens is suitable for comparing the **same** PP or TG workload among FastRPC, dspqueue, and CPU. Compare PP with PP and TG with TG, not a combined `pp+tg` run against a TG-only run. The reported energy excludes load/initialization, so use the device-side snapshots/logs to study OOM during loading; do not confuse a low per-token estimate with low total startup energy.
