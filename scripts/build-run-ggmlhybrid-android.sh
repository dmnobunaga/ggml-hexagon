#!/usr/bin/env bash
# Isolated experimental FastRPC + Vulkan Android build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GGML_HYBRID_BUILD_DIR:-${ROOT}/out/ggmlhybrid-android}"
REMOTE="${GGML_HYBRID_REMOTE_DIR:-/data/local/tmp/ggml-hybrid}"
ACTION="${1:-build}"

case "$ACTION" in
build)
    : "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to the Android NDK directory}"
    : "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT to the Hexagon SDK directory}"
    : "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT to the Hexagon tools directory}"
    : "${VULKAN_SDK:?Set VULKAN_SDK to the host LunarG Vulkan SDK directory}"

    GLSLC="${Vulkan_GLSLC_EXECUTABLE:-${VULKAN_SDK}/bin/glslc}"
    SPIRV="${SPIRV_HEADERS_DIR:-${VULKAN_SDK}/share/cmake/SPIRV-Headers}"
    VK_LIBRARY="${Vulkan_LIBRARY:-${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/34/libvulkan.so}"
    for required in \
        "${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
        "${VULKAN_SDK}/include/vulkan/vulkan.hpp" \
        "$GLSLC" "$VK_LIBRARY" "$SPIRV"; do
        if [[ ! -e "$required" ]]; then
            echo "Missing dependency: $required" >&2
            exit 1
        fi
    done

    FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE -I${VULKAN_SDK}/include"
    cmake -S "$ROOT" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
        -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
        -DGGML_HEXAGON=ON -DGGML_HEXAGON_USE_MEMPOOL=ON \
        -DGGML_VULKAN=ON -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF \
        -DGGML_LLAMAFILE=ON -DGGML_CCACHE=ON \
        -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
        -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" \
        -DPREBUILT_LIB_DIR=android_aarch64 \
        -DVulkan_INCLUDE_DIR="${VULKAN_SDK}/include" \
        -DVulkan_LIBRARY="$VK_LIBRARY" \
        -DVulkan_GLSLC_EXECUTABLE="$GLSLC" \
        -DSPIRV-Headers_DIR="$SPIRV" \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_SUBPROCESS=ON \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF
    cmake --build "$BUILD" --parallel "${JOBS:-4}"
    echo "Build complete: $BUILD"
    ;;
push)
    : "${ADB:=adb}"
    [[ -f "${BUILD}/bin/libggml-hexagon.so" ]] || { echo "Missing Hexagon AP library" >&2; exit 1; }
    [[ -f "${BUILD}/bin/libggml-vulkan.so" ]] || { echo "Missing Vulkan library" >&2; exit 1; }
    shopt -s nullglob
    skels=("${BUILD}"/bin/libggml-htp-v*.so)
    (( ${#skels[@]} > 0 )) || { echo "Missing FastRPC DSP skeleton" >&2; exit 1; }

    "$ADB" shell "mkdir -p '$REMOTE'"
    for f in "${BUILD}"/bin/*.so "${BUILD}"/bin/llama-bench "${BUILD}"/bin/llama-cli "${BUILD}"/bin/llama-completion; do
        [[ -f "$f" ]] || continue
        "$ADB" push "$f" "${REMOTE}/$(basename "$f")"
    done
    "$ADB" shell "chmod 755 '$REMOTE'/llama-*"
    echo "Isolated hybrid runtime pushed to: $REMOTE"
    echo "Set LD_LIBRARY_PATH=$REMOTE when running it."
    ;;
*)
    echo "Usage: $0 {build|push}" >&2
    exit 2
    ;;
esac
