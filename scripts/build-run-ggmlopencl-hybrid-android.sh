#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export GGML_HYBRID_OPENCL=1
unset GGML_HYBRID_VULKAN
export GGML_LOCAL_BUILD_DIR="${GGML_LOCAL_BUILD_DIR:-${ROOT}/out/ggmlopencl-hybrid-android}"
export GGML_REMOTE_PATH="${GGML_REMOTE_PATH:-/data/local/tmp/ggml-opencl-hybrid}"

case "${1:-build}" in
    build)
        "$ROOT/scripts/build-run-ggmlhexagon-android.sh" build
        "$ROOT/scripts/build-run-ggmlhexagon-android.sh" update_hybrid_libs
        ;;
    update_fastrpc_libs|update_hybrid_libs|update)
        "$ROOT/scripts/build-run-ggmlhexagon-android.sh" update_hybrid_libs
        ;;
    clean)
        "$ROOT/scripts/build-run-ggmlhexagon-android.sh" clean
        ;;
    devices)
        adb shell "cd '${GGML_REMOTE_PATH}' && export LD_LIBRARY_PATH='${GGML_REMOTE_PATH}:/vendor/lib64' && export ADSP_LIBRARY_PATH='${GGML_REMOTE_PATH}' && ./llama-bench --list-devices"
        ;;
    *)
        echo "Usage: $0 {build|update_fastrpc_libs|devices|clean}" >&2
        exit 2
        ;;
esac
