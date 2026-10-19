#!/usr/bin/env python3
"""Compare preflight memory and system counters around RPCMEM lifetime.

All values are observational. MemAvailable, RSS, ION and DMA-BUF are not
additive; ION and DMA-BUF may represent system-wide rather than process usage.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MEM_RX = re.compile(r"\[MEM\] stage=(\S+)\s+t_ms=(\d+)\s+pid=(\d+)\s+(.*)")
PLAN_RX = re.compile(r"\[MEM-PLAN\] requested=(\S+) device=(\S+) domain=(\S+) bytes=(\d+) MiB=([\d.]+)")
FD_RX = re.compile(r"\[MEM-RPCMEM-FD\] stage=(\S+) device=(\d+) fd=(\d+) backing_bytes=(\d+)\s+(.*)")
ACCOUNT_RX = re.compile(r"\[MEM-ACCOUNT-BUFT\] stage=(\S+) type=(\S+) buft_id=(\d+) actual_type=(\S+) actual_buft_id=(\d+)\s+(.*)")

MEASURES = ("MemAvailableKiB", "VmRSSKiB", "PssKiB", "IonTotalUsedKiB",
            "IonTotalCacheKiB", "RssAnonKiB", "VmSwapKiB", "dmaBufGlobalKiB")
PHASES = ("before_rpcmem", "after_rpcmem_alloc", "after_rpcmem_register",
          "memory_preflight_before_guard", "memory_guard_reject_cpu",
          "before_model_buffers", "after_load_all_data", "after_context_init",
          "before_rpcmem_free", "after_rpcmem_free")


def fields_from(raw: str) -> dict[str, int]:
    result = {}
    for token in raw.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            result[key] = int(value)
        except ValueError:
            pass
    return result


def parse(log: str) -> dict:
    samples, plans, fd, buffers = [], [], [], []
    for line in log.splitlines():
        m = MEM_RX.search(line)
        if m:
            samples.append({"stage": m[1], "time_ms": int(m[2]),
                            "pid": int(m[3]), **fields_from(m[4])})
        m = PLAN_RX.search(line)
        if m:
            plans.append({"requested": m[1], "device": m[2], "domain": m[3],
                          "bytes": int(m[4])})
        m = FD_RX.search(line)
        if m:
            fd.append({"stage": m[1], "device": int(m[2]),
                       "fd": int(m[3]), "backing_bytes": int(m[4]),
                       "fields": m[5]})
        m = ACCOUNT_RX.search(line)
        if m:
            buffers.append({"stage": m[1], "requested": m[2],
                            "actual": m[4], "details": m[6]})
    return {"samples": samples, "plans": plans, "fd": fd, "buffers": buffers}


def format_mib(value: int | None) -> str:
    return f"{value / 1024:.2f}" if value is not None and value >= 0 else "n/a"


def report(parsed: dict) -> None:
    samples = parsed["samples"]
    if not samples:
        raise SystemExit("No [MEM] rows; enable GGML_MEM_SNAPSHOT=1.")
    first_pid = samples[0]["pid"]
    if len({s["pid"] for s in samples}) > 1:
        print("NOTE: more than one PID; do not compare across process boundaries")
    print(f"Memory chronology: {len(samples)} rows; first PID={first_pid}")
    print("stage\tt_ms\t" + "\t".join(x.removesuffix("KiB") + "MiB" for x in MEASURES))
    for s in samples:
        if any(s["stage"].startswith(x) for x in PHASES):
            print("\t".join((s["stage"], str(s["time_ms"])) +
                tuple(format_mib(s.get(k)) for k in MEASURES)))
    print("\nRPCMEM interval deltas, same PID and device, compared at adjacent boundaries")
    for name, from_phase, to_phase in (
        ("allocation", "before_rpcmem:", "after_rpcmem_alloc:"),
        ("registration", "after_rpcmem_alloc:", "after_rpcmem_register:"),
        ("release", "before_rpcmem_free:", "after_rpcmem_free:"),
    ):
        before = next((x for x in samples if x["stage"].startswith(from_phase)), None)
        after = next((x for x in samples if x["stage"].startswith(to_phase)
                      and before and x["pid"] == before["pid"]
                      and x["time_ms"] >= before["time_ms"]), None)
        if before is None or after is None:
            print(f"  {name}: unavailable (one boundary was not captured)")
            continue
        values = []
        for key in MEASURES:
            a, b = before.get(key), after.get(key)
            if a is not None and b is not None and a >= 0 and b >= 0:
                values.append(f"{key}={((b-a)/1024):+.2f} MiB")
        print(f"  {name}: " + ", ".join(values))

    if parsed["plans"]:
        print("\nPlanned buffers (pre-allocation; not evidence of resident memory)")
        for p in parsed["plans"]:
            print(f"  requested={p['requested']} device={p['device']} "
                  f"domain={p['domain']} planned={p['bytes'] / 1048576:.2f} MiB")
        for domain in sorted({p["domain"] for p in parsed["plans"]}):
            total = sum(p["bytes"] for p in parsed["plans"] if p["domain"] == domain)
            print(f"  domain={domain} total_planned={total / 1048576:.2f} MiB")
    if parsed["fd"]:
        print("\nRPCMEM fd identity (not proof of exclusive physical-page ownership)")
        for x in parsed["fd"]:
            print(f"  {x['stage']} fd={x['fd']} backing={x['backing_bytes'] / 1048576:.2f} MiB {x['fields']}")
    if parsed["buffers"]:
        print("\nBuffer origin -> actual allocation type")
        seen = set()
        for b in parsed["buffers"]:
            key = (b["stage"], b["requested"], b["actual"])
            if key not in seen:
                print(f"  {b['stage']}: {b['requested']} -> {b['actual']}")
                seen.add(key)
    print("\nNOTE: no sum of IonTotalUsed, VmRSS, dmaBufGlobal or allocated capacity is a valid RSS estimate.")


def self_test() -> None:
    log = """[MEM] stage=before_rpcmem:HTP0 t_ms=10 pid=123 MemAvailableKiB=1000000 VmRSSKiB=1000 IonTotalUsedKiB=4000
[MEM] stage=after_rpcmem_alloc:HTP0 t_ms=11 pid=123 MemAvailableKiB=900000 VmRSSKiB=1000 IonTotalUsedKiB=1052576
[MEM] stage=after_rpcmem_register:HTP0 t_ms=12 pid=123 MemAvailableKiB=900000 VmRSSKiB=2000 IonTotalUsedKiB=1052576
[MEM-PLAN] requested=CPU_REPACK device=CPU domain=CPU bytes=4096 MiB=0.00
[MEM-RPCMEM-FD] stage=after_rpcmem_to_fd device=0 fd=3 backing_bytes=1073741824 stat_ok=1 inode=42
[MEM-ACCOUNT-BUFT] stage=model_buffers_allocated type=CPU_REPACK buft_id=5 actual_type=CPU actual_buft_id=2 buffers=1 logical=1.00MiB
"""
    data = parse(log)
    assert len(data["samples"]) == 3
    assert data["samples"][0]["IonTotalUsedKiB"] == 4000
    assert data["samples"][1]["IonTotalUsedKiB"] == 1052576
    assert data["plans"][0]["requested"] == "CPU_REPACK"
    assert data["fd"][0]["backing_bytes"] == 1073741824
    assert data["buffers"][0]["actual"] == "CPU"
    print("BF16 attribution log parser: self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.logfile:
        report(parse(args.logfile.read_text(errors="replace")))
    else:
        parser.error("provide a logfile or --self-test")
