#!/system/bin/sh

cd /data/local/tmp || exit 1

export LD_LIBRARY_PATH=/data/local/tmp
export GGML_HEXAGON_DEVICES=1
export GGML_HEXAGON_SEGMENTED_RPCMEM=1
export GGML_HEXAGON_WORKSPACE_MB=256
export GGML_HEXAGON_MEMORY_RESERVE_MB=1536
export GGML_MEM_GUARD_WARN_ONLY=1
export GGML_MEM_ACCOUNTING=1
export GGML_MEM_SNAPSHOT=1
unset GGML_HEXAGON_RPCMEM_MB
unset GGML_MEM_PREFLIGHT_ONLY

LOG=fastrpc-bf16-run.log
MEM=fastrpc-bf16-mem.log
RESULT=fastrpc-bf16-result.txt

sample() {
    printf 'phase=%s uptime_s=%s pid=%s ' \
        "$1" "$(cut -d ' ' -f1 /proc/uptime)" "$2"
    awk '
    $1=="MemAvailable:" ||
    $1=="MemFree:" ||
    $1=="Cached:" ||
    $1=="SwapFree:" ||
    $1=="IonTotalUsed:" ||
    $1=="IonTotalCache:" ||
    $1=="VmRSS:" ||
    $1=="VmHWM:" ||
    $1=="VmSwap:" ||
    $1=="RssAnon:" ||
    $1=="Pss:" ||
    $1=="Private_Dirty:" {
        key=$1
        sub(/:$/, "", key)
        printf "%s=%s ", key, $2
    }
    END { print "" }
    ' /proc/meminfo /proc/"$2"/status /proc/"$2"/smaps_rollup 2>/dev/null
}

AVAILABLE_KIB=$(awk '$1=="MemAvailable:" {print $2; exit}' /proc/meminfo)
MIN_AVAILABLE_KIB=8388608
if [ -z "$AVAILABLE_KIB" ] || [ "$AVAILABLE_KIB" -lt "$MIN_AVAILABLE_KIB" ]; then
    echo "SKIPPED: MemAvailable=${AVAILABLE_KIB:-unknown} KiB; need at least 8388608 KiB (8192 MiB)."
    echo 'Do not force BF16 at low available memory.'
    exit 2
fi

sample BEFORE $$ > "$MEM"

(
    echo 800 > /proc/self/oom_score_adj 2>/dev/null || true
    exec ./llama-bench -v \
        -m /sdcard/Spark-X2.5-4B.gguf \
        -t 8 -ngl 15 -fa 1 -p 8 -n 1 -r 1
) > "$LOG" 2>&1 &
BENCH_PID=$!
echo "benchmark_pid=$BENCH_PID"

(
    while kill -0 "$BENCH_PID" 2>/dev/null; do
        sample RUNNING "$BENCH_PID"
        avail=$(awk '$1=="MemAvailable:" {print $2; exit}' /proc/meminfo)
        if [ -n "$avail" ] && [ "$avail" -lt 786432 ]; then
            echo "EMERGENCY_STOP: MemAvailable=${avail}KiB < 768MiB; terminating benchmark" >> "$MEM"
            kill -TERM "$BENCH_PID" 2>/dev/null
            break
        fi
        sleep 0.5
    done
) >> "$MEM" &
MONITOR_PID=$!

wait "$BENCH_PID"
RC=$?
wait "$MONITOR_PID"
sample AFTER $$ >> "$MEM"
printf 'exit_code=%s\n' "$RC" > "$RESULT"
echo "exit_code=$RC"
echo "Logs: $LOG $MEM $RESULT"
