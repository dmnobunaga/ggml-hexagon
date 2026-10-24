#!/usr/bin/env bash
# Opt-in wrapper for the existing FastRPC Android build/deployment pipeline.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export GGML_HYBRID_VULKAN=1
export GGML_LOCAL_BUILD_DIR="${GGML_LOCAL_BUILD_DIR:-${ROOT}/out/ggmlhybrid-android}"
export GGML_REMOTE_PATH="${GGML_REMOTE_PATH:-/data/local/tmp/ggml-hybrid}"

case "${1:-build}" in
    build)
        "$ROOT/scripts/build-run-ggmlhexagon-android.sh" build
        for binary in llama-bench llama-completion llama-cli; do
            if [ -f "${GGML_LOCAL_BUILD_DIR}/bin/${binary}" ]; then
                adb push "${GGML_LOCAL_BUILD_DIR}/bin/${binary}" "${GGML_REMOTE_PATH}/"
                adb shell "chmod 755 '${GGML_REMOTE_PATH}/${binary}'"
            fi
        done
        ;;
    clean)
        "$ROOT/scripts/build-run-ggmlhexagon-android.sh" clean
        ;;
    devices)
        adb shell "cd '${GGML_REMOTE_PATH}' && export LD_LIBRARY_PATH='${GGML_REMOTE_PATH}' && export ADSP_LIBRARY_PATH='${GGML_REMOTE_PATH}' && ./llama-bench --list-devices"
        ;;
    *)
        echo "Usage: $0 {build|clean|devices}" >&2
        exit 2
        ;;
esac
