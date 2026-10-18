#!/system/bin/sh
# Run on the Android device (adb push, then adb shell sh ...).
# Every iteration starts a new llama-bench process and reinitializes the backend.
set -u

RUNS="${1:-20}"
MODEL="${2:-/sdcard/Spark-X2.5-4B.gguf}"
RUNTIME="${3:-/data/local/tmp}"
OUT="${4:-/data/local/tmp/energy-stress-$(date +%Y%m%d-%H%M%S)}"
LABEL="${5:-hexagon}"
DELAY="${STRESS_DELAY:-20}"
PP="${STRESS_PP:-200}"
TG="${STRESS_TG:-32}"
NGL="${STRESS_NGL:-99}"
THREADS="${STRESS_THREADS:-6}"
MAX_TEMP="${STRESS_MAX_TEMP_DECIC:-450}"

case "$RUNS" in ''|*[!0-9]*) echo "RUNS must be a positive integer" >&2; exit 2;; esac
[ "$RUNS" -gt 0 ] || exit 2
[ -x "$RUNTIME/llama-bench" ] || { echo "llama-bench not executable: $RUNTIME" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "GGUF not found: $MODEL" >&2; exit 2; }

mkdir -p "$OUT" || exit 2
export LD_LIBRARY_PATH="$RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_HEXAGON_OPPOLL=1

cat > "$OUT/config.txt" <<EOF
label=$LABEL
model=$MODEL
runtime=$RUNTIME
runs=$RUNS
n_gpu_layers=$NGL
threads=$THREADS
prompt_tokens=$PP
generation_tokens=$TG
delay_seconds=$DELAY
max_battery_temp_deci_c=$MAX_TEMP
boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
EOF
printf 'run,label,status,exit_code,elapsed_s,temp_before_deci_c,temp_after_deci_c,boot_before,boot_after\n' > "$OUT/summary.csv"

snapshot() {
    file="$1"
    {
        date
        cat /proc/meminfo
        cat /proc/pressure/memory 2>/dev/null
        cat /sys/class/power_supply/battery/status 2>/dev/null
        cat /sys/class/power_supply/battery/capacity 2>/dev/null
        cat /sys/class/power_supply/battery/temp 2>/dev/null
        cat /sys/class/power_supply/battery/charge_counter 2>/dev/null
        cat /sys/class/power_supply/battery/energy_now 2>/dev/null
        cat /sys/class/power_supply/battery/voltage_now 2>/dev/null
    } > "$file" 2>&1
}

i=1
while [ "$i" -le "$RUNS" ]; do
    id="$(printf '%03d' "$i")"
    pretemp="$(cat /sys/class/power_supply/battery/temp 2>/dev/null || echo -1)"
    case "$pretemp" in ''|*[!0-9]*) pretemp=-1;; esac
    if [ "$pretemp" -ge "$MAX_TEMP" ]; then
        echo "Battery temperature ${pretemp}/10 C exceeds limit ${MAX_TEMP}/10 C. Stopping." >&2
        break
    fi

    before="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
    snapshot "$OUT/run-${id}-before.txt"
    start="$(date +%s)"
    echo "[$i/$RUNS] $LABEL model=$MODEL start=$(date)"
    "$RUNTIME/llama-bench" \
        -m "$MODEL" -t "$THREADS" -ngl "$NGL" \
        --poll 1000 -fa 1 --ubatch-size 1024 \
        -r 1 -p "$PP" -n "$TG" --energy -o csv \
        > "$OUT/run-${id}-bench.csv" 2> "$OUT/run-${id}-stderr.log"
    rc=$?
    end="$(date +%s)"
    elapsed=$((end - start))
    after="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
    posttemp="$(cat /sys/class/power_supply/battery/temp 2>/dev/null || echo -1)"
    case "$posttemp" in ''|*[!0-9]*) posttemp=-1;; esac
    snapshot "$OUT/run-${id}-after.txt"

    status=PASS
    if [ "$rc" -ne 0 ]; then
        status=FAIL
    elif ! grep -q 'energy_mwh' "$OUT/run-${id}-bench.csv"; then
        status=NO_ENERGY_BUILD
    elif grep -q '"unavailable"' "$OUT/run-${id}-bench.csv"; then
        status=NO_ENERGY_COUNTERS
    fi
    if [ -n "$before" ] && [ -n "$after" ] && [ "$before" != "$after" ]; then
        status=REBOOT
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$i" "$LABEL" "$status" "$rc" "$elapsed" "$pretemp" "$posttemp" "$before" "$after" >> "$OUT/summary.csv"
    echo "[$i/$RUNS] status=$status exit=$rc time=${elapsed}s temp=${posttemp}/10C"

    if [ "$status" = REBOOT ]; then
        break
    fi
    i=$((i + 1))
    if [ "$i" -le "$RUNS" ]; then
        sleep "$DELAY"
    fi
done
echo "Completed. Logs: $OUT"
