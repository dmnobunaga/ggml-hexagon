#!/usr/bin/env python3
"""Validate model-agnostic ggml MEM-ACCOUNT snapshots without device access."""

import re
import sys
from pathlib import Path

HEADER = re.compile(
    r"^\[MEM-ACCOUNT\] stage=(\S+) physical_live=([0-9.]+)MiB "
    r"physical_peak=([0-9.]+)MiB pool_live=([0-9.]+)MiB"
)
BUFT = re.compile(
    r"^\[MEM-ACCOUNT-BUFT\] stage=(\S+) type=(\S+) .*physical_unique=([0-9.]+)MiB"
)
BACKING = re.compile(
    r"^\[MEM-ACCOUNT-BACKING\] stage=(\S+) id=(\d+) type=(\S+) "
    r"allocated=([0-9.]+)MiB children_live=([0-9.]+)MiB "
    r"children_peak=([0-9.]+)MiB bump_hwm=([0-9.]+)MiB"
)
EPSILON_MIB = 0.10  # The logger rounds all measurements to 0.01 MiB.


def validate(contents: str) -> list[str]:
    snapshots = []
    active = None
    for line in contents.splitlines():
        header = HEADER.match(line)
        if header:
            if active:
                snapshots.append(active)
            active = {
                "stage": header[1], "physical": float(header[2]),
                "peak": float(header[3]), "pool": float(header[4]),
                "owned": 0.0, "backings": 0.0, "ids": set(), "errors": [],
            }
            continue
        if active is None:
            continue
        buft = BUFT.match(line)
        if buft and buft[1] == active["stage"]:
            active["owned"] += float(buft[3])
        backing = BACKING.match(line)
        if backing and backing[1] == active["stage"]:
            backing_id = int(backing[2])
            if backing_id in active["ids"]:
                active["errors"].append("duplicate backing ID")
            active["ids"].add(backing_id)
            allocated = float(backing[4])
            child_live = float(backing[5])
            child_peak = float(backing[6])
            active["backings"] += allocated
            if child_live > allocated + EPSILON_MIB:
                active["errors"].append(f"backing {backing_id}: child region usage exceeds allocation")
            if child_peak + EPSILON_MIB < child_live:
                active["errors"].append(f"backing {backing_id}: peak child usage less than live")
    if active:
        snapshots.append(active)
    if not snapshots:
        return ["no [MEM-ACCOUNT] snapshots; check GGML_MEM_ACCOUNTING=1"]

    errors = []
    for snap in snapshots:
        prefix = snap["stage"]
        errors.extend(f"{prefix}: {error}" for error in snap["errors"])
        if abs(snap["physical"] - snap["owned"] - snap["backings"]) > EPSILON_MIB:
            errors.append(
                f"{prefix}: double counting or missing ownership: physical={snap['physical']:.2f} "
                f"but unique buffers + backings={snap['owned'] + snap['backings']:.2f} MiB"
            )
        if abs(snap["pool"] - snap["backings"]) > EPSILON_MIB:
            errors.append(f"{prefix}: pool_live and backing totals disagree")
        if snap["peak"] + EPSILON_MIB < snap["physical"]:
            errors.append(f"{prefix}: physical_peak less than physical_live")
    return errors


def self_test() -> None:
    good = """[MEM-ACCOUNT] stage=test physical_live=1280.00MiB physical_peak=1340.00MiB pool_live=1024.00MiB active_pool_suballoc_peak=512.00MiB buffers=2 allocs=2 frees=0 writes=0.00MiB calls=0
[MEM-ACCOUNT-BUFT] stage=test type=CPU buffers=1 logical=256.00MiB physical_unique=256.00MiB marked_weights=256.00MiB
[MEM-ACCOUNT-BUFT] stage=test type=HTP-REPACK buffers=1 logical=512.00MiB physical_unique=0.00MiB marked_weights=512.00MiB
[MEM-ACCOUNT-BACKING] stage=test id=2 type=HTP_RPCMEM allocated=1024.00MiB children_live=512.00MiB children_peak=512.00MiB bump_hwm=800.00MiB mapping_registrations=1
"""
    assert not validate(good), validate(good)
    bad = good.replace("physical_live=1280.00", "physical_live=1792.00")
    assert any("double counting" in e for e in validate(bad))
    bad = good.replace("children_live=512.00", "children_live=1200.00")
    assert any("exceeds allocation" in e for e in validate(bad))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        print("MEM-ACCOUNT log validator: self-test passed")
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 tests/check-mem-account-log.py <llama-bench.log>")
    errs = validate(Path(sys.argv[1]).read_text(errors="replace"))
    for e in errs:
        print("FAIL:", e)
    if errs:
        raise SystemExit(1)
    print("MEM-ACCOUNT snapshots: ownership and backing totals consistent")
