# FastRPC `ggml_htp_close()` — invalid-handle diagnostic (2026-10-21)

## Status

**NOT FIXED OR VERIFIED.** The existing fast path completes BF16 PP/TG,
but the AP logs `error 0x2c: failed to close ggmlop dsp handle`, including
with DSP VA window disabled. In Qualcomm AEEStdErr, 0x2c is
`AEE_EBADHANDLE`. The precise source of the error remains unknown.

This branch is based on `experiment/dsp-va-window`. The checkpoint of the
successful 24-layer VA experiment is recorded separately on that branch:
`docs/backend/snapdragon/dsp-va-window-experiment.md`.

## Instrumentation

AP (opt in with `GGML_HEXAGON_CLOSE_DIAG=1`):

- Logs `ggmlop_handle`, number of tracked segments, workspace and domain
  before close.
- Unregisters segments as before.
- Performs one harmless `ggml_htp_execute_batch(handle, 0, 0xFFFB)`
  after segment teardown and before closing. This requires the
  original DSP workspace mapping and reports whether the handle is still
  invocable just before `ggml_htp_close()`.
- Logs the close return code without masking errors or retrying.
- All extra AP RPC calls are disabled unless the environment variable is set.

DSP (close path only):

- Logs entry, segment cleanup, work/DMA cleanup, HMX cleanup, compute
  resource release, and function exit.
- On the base `experiment/fastrpc-close-handle` diagnostic branch,
  teardown operations are unchanged.
- On the descendant `experiment/fastrpc-close-workspace-unmap` branch,
  the close function also calls `HAP_munmap2` on the main DSP workspace
  after worker/HMX queues are destroyed and before releasing compute
  resources. Failure is logged and returned as `AEE_EFAILED`.

**Do not treat `AEE_EBADHANDLE` as success.** Some FastRPC versions can
invalidate local handle bookkeeping on a failed close attempt; blindly
retrying or freeing host RPCMEM could cause use-after-free.

## Build and run

Build `-DGGML_HEXAGON_USE_MEMPOOL=ON` from this branch and install the
matching AP and DSP artifacts. The inherited DSP VA branch has an updated
IDL `register_rpcmem_segment(..., pinned)`; do not mix old and new stubs.

In a fresh process on the device (short run is sufficient to exercise close):

```sh
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp
export GGML_HEXAGON_SEGMENTED_RPCMEM=1
export GGML_HEXAGON_ADAPTIVE_OFFLOAD=1
export GGML_HEXAGON_SEGMENT_MB=512
export GGML_HEXAGON_MEMORY_RESERVE_MB=1024
export GGML_HEXAGON_DSP_VA_WINDOW_MB=2560
export GGML_HEXAGON_CLOSE_DIAG=1
unset GGML_HEXAGON_ADAPTIVE_CANDIDATE_RETRY
./llama-bench -t 6 --poll 1000 -ngl 15 -fa 1 --ubatch-size 128 \
  -p 16 -n 4 -r 1 -m /sdcard/Spark-X2.5-4B.gguf
```

Capture all logcat buffers *while the benchmark runs*:

```sh
adb logcat -b all -c
adb logcat -b all -v time > dsp-close.log
# Stop capture with Ctrl-C after the benchmark exits.
grep -E 'CLOSE-DIAG|remote_handle.*close|bad handle|ggmlop dsp handle' dsp-close.log
```

On PowerShell replace `grep` with
`Select-String -Path dsp-close.log -Pattern 'CLOSE-DIAG|remote_handle.*close|bad handle|ggmlop dsp handle'`.

## How to interpret

1. **AP pre-close invoke fails**: local or remote handle was lost before close,
   or session/transport is already unusable. Compare the preceding
   `unregister_rpcmem_segment`/memory unmap logs.
2. **Pre-close invoke succeeds but no DSP close entry appears**: problem
   arises while dispatching the close call, potentially in FastRPC
   handle-table or DSP skeleton dispatch. Collect `adsprpc` logs.
3. **DSP close entry appears but no exit**: narrow to the last logged
   DSP cleanup phase, especially HMX/work queues and compute resources.
4. **DSP close exits successfully but AP returns 0x2c**: prioritize host
   FastRPC close bookkeeping / returned remote handle status. Do **not**
   infer that host-side release is safe solely from the DSP trace.
5. **Both exit successfully**: repeat load/run/close with VA window
   OFF and ON and repeat model loads in the same process, checking
   resource ownership and absence of memory leaks.

The legacy DSP `ggml_htp_close` returns 0 unconditionally on its normal
path; that fact alone does not prove the function was reached on failing runs.

## Follow-up once the failure boundary is known

- If DSP cleanup is implicated, repair release sequencing in a **separate**
  source commit, keeping host error checking intact.
- If FastRPC handle close bookkeeping is implicated, inspect the device
  libcdsprpc behavior / logcat and the generated QAIC stub/skeleton
  contract; do not assume the open remote handle is safe to reuse.
- Compare the diagnostic-only branch against the workspace-unmap branch.
  A successful workspace unmap is a cleanup correctness improvement,
  but does not establish that it fixes `AEE_EBADHANDLE` until tested.

## Two-stage test to isolate the patch

1. First test this **workspace-unmap** branch with the short command above.
   Look for `[CLOSE-DIAG] DSP workspace unmapped` and the final AP
   `close returned=...` line.
2. If the close error persists, test **diagnostic-only**
   `experiment/fastrpc-close-handle` with matching AP/DSP artifacts,
   to determine whether explicit workspace unmapping changes the error.
3. Only after a successful close, test two model load/run/unload cycles
   **in the same process** to detect retained handle state and resource leaks.

Neither branch has been compiled or run on hardware from this session.
Do not deploy over the known-good `f308c7499` binaries without backup.
