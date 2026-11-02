#!/usr/bin/env python3
"""Assemble SM100a PTX and check each entry's registers, spills and occupancy.

Explicit stack frames for compiler temporaries are reported, not called spills.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from check_ptx import KERNELS


def occupancy(registers: int) -> float:
    if not 1 <= registers <= 255:
        raise ValueError("invalid register allocation")
    per_warp = ((registers * 32 + 255) // 256) * 256
    blocks = min(8, 65536 // (per_warp * 8))
    return blocks / 8


def parse_ptxas(log: str) -> tuple[list[int], int, int]:
    regs = [int(n) for n in re.findall(r"\bUsed\s+(\d+)\s+registers?\b", log, re.I)]
    stores = sum(int(n) for n in re.findall(r"\b(\d+)\s+bytes?\s+spill\s+stores?\b", log, re.I))
    loads = sum(int(n) for n in re.findall(r"\b(\d+)\s+bytes?\s+spill\s+loads?\b", log, re.I))
    return regs, stores, loads


def check_resources(log: str, limit: int = 128, required=KERNELS) -> list[str]:
    errors = []
    if re.search(r"ptxas\s+(?:fatal|error)", log, re.I):
        errors.append("ptxas reported an error")
    starts = list(re.finditer(r"Compiling entry function '([^']+)' for '([^']+)'", log))
    if not starts:
        return errors + ["missing per-kernel ptxas reports"]
    seen = set()
    for i, match in enumerate(starts):
        name, arch = match.groups()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(log)
        body = log[match.end():end]
        reg = re.search(r"\bUsed\s+(\d+)\s+registers?\b", body)
        spills = re.search(r"(\d+)\s+bytes?\s+stack\s+frame,\s*(\d+)\s+bytes?\s+spill\s+stores?,\s*(\d+)\s+bytes?\s+spill\s+loads?", body)
        if name in seen or arch != "sm_100a" or reg is None or spills is None:
            errors.append(f"incomplete, duplicate or wrong-target report: {name}")
            continue
        seen.add(name)
        registers = int(reg[1])
        stack, stores, loads = map(int, spills.groups())
        if registers < 1 or registers > limit:
            errors.append(f"register cap {limit} exceeded/invalid in {name}: {registers}")
        if stores or loads:
            errors.append(f"spills in {name}: stores={stores} bytes loads={loads} bytes")
        if 1 <= registers <= 255 and occupancy(registers) < 0.25:
            errors.append(f"register-limited occupancy below 25% in {name}")
    for symbol in required:
        if symbol not in seen:
            errors.append(f"missing ptxas report for {symbol}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--output", type=Path, default=Path("/tmp/causal_softmax_sm100a.cubin"))
    parser.add_argument("--log", type=Path, help="save the raw ptxas resource report")
    args = parser.parse_args()
    command = [args.ptxas, "-arch=sm_100a", "--verbose", "--warn-on-spills", "-o", str(args.output), str(args.ptx)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        print(f"FAIL cannot run ptxas: {exc}", file=sys.stderr)
        return 1
    log = result.stdout + result.stderr
    if args.log:
        args.log.write_text(log)
    print(log)
    if result.returncode:
        print(f"FAIL ptxas exit {result.returncode}", file=sys.stderr)
        return 1
    errors = check_resources(log)
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    if errors:
        return 1
    registers, stores, loads = parse_ptxas(log)
    print(f"PASS ptxas: {len(KERNELS)} entries max_regs={max(registers)} spills={stores + loads} occupancy_floor={occupancy(max(registers)):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
