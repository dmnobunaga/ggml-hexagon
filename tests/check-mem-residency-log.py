#!/usr/bin/env python3
"""Inspect model-agnostic Linux/Android memory snapshots written by ggml."""

import argparse
import re
from pathlib import Path

RX = re.compile(r"^\[MEM\] stage=(\S+) t_ms=(\d+) pid=(\d+) (.*)$")
FIELDS = (
    "VmRSSKiB", "VmHWMKiB", "PssKiB", "PrivateDirtyKiB",
    "RssAnonKiB", "RssFileKiB", "RssShmemKiB", "VmSwapKiB",
    "SwapPssKiB", "MemAvailableKiB", "IonTotalUsedKiB",
    "IonTotalCacheKiB", "dmaBufGlobalKiB",
)


def read_rows(log: str) -> list[dict]:
    rows = []
    for line in log.splitlines():
        match = RX.match(line)
        if not match:
            continue
        row = {"stage": match[1], "time_ms": int(match[2]), "pid": int(match[3])}
        for part in match[4].split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                row[key] = int(value)
            except ValueError:
                pass
        rows.append(row)
    return rows


def mib(row: dict, field: str) -> str:
    n = row.get(field, -1)
    return f"{n / 1024:.1f}" if n >= 0 else "n/a"


def report(rows: list[dict]) -> None:
    print(
        "stage\tpid\tt_ms\tVmRSS MiB\tVmHWM MiB\tPSS MiB\t"
        "PrivateDirty MiB\tRssAnon MiB\tRssFile MiB\tRssShmem MiB\t"
        "VmSwap MiB\tSwapPss MiB\tMemAvailable MiB\t"
        "IonUsed MiB\tIonCache MiB\tGlobalDmaBuf MiB"
    )
    for row in rows:
        print("\t".join(
            [str(row["stage"]), str(row["pid"]), str(row["time_ms"])]
            + [mib(row, field) for field in FIELDS]
        ))

    if not rows:
        raise SystemExit("No [MEM] snapshots; set GGML_MEM_SNAPSHOT=1 before running.")


def self_test() -> None:
    example = (
        "[MEM] stage=after_context_init t_ms=100 pid=42 "
        "VmRSSKiB=102400 VmHWMKiB=110592 PssKiB=99000 "
        "PrivateDirtyKiB=20480 RssAnonKiB=60000 RssFileKiB=40000 "
        "RssShmemKiB=2400 VmSwapKiB=0 SwapPssKiB=-1 "
        "MemAvailableKiB=5000000 IonTotalUsedKiB=450000 "
        "IonTotalCacheKiB=600000 dmaBufGlobalKiB=-1\n"
    )
    rows = read_rows(example)
    assert len(rows) == 1
    assert rows[0]["stage"] == "after_context_init"
    assert rows[0]["PssKiB"] == 99000
    assert mib(rows[0], "VmRSSKiB") == "100.0"
    assert mib(rows[0], "SwapPssKiB") == "n/a"
    assert read_rows("not a memory snapshot") == []
    print("Memory residency parser: self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.logfile is None:
        parser.error("logfile is required unless --self-test is used")
    else:
        report(read_rows(args.logfile.read_text(errors="replace")))
