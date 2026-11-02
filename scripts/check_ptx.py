#!/usr/bin/env python3
"""Check each SM100a entry for its required GPU instructions.

PTX local scratch is not the same as a register spill. The assembler's explicit
spill counters are the authoritative zero-spill gate in check_resources.py.
"""
import argparse
import re
import sys
from pathlib import Path

KERNELS = (
    "probe",
    "causal_softmax_baseline",
    "causal_softmax_warp",
    "causal_softmax_tma",
    "causal_softmax_tma_pipelined",
    "causal_softmax_tma_online",
)
TMA = KERNELS[3:]
TMA_REQUIRED = {
    "2D TMA load": r"cp\.async\.bulk\.tensor\.2d\.shared(?:::cta|::cluster)?\.global",
    "2D TMA store": r"cp\.async\.bulk\.tensor\.2d\.global\.shared(?:::cta|::cluster)?",
    "mbarrier init": r"mbarrier\.init",
    "mbarrier expected bytes": r"mbarrier\.arrive\.expect_tx",
    "mbarrier wait": r"mbarrier\.try_wait",
    "asynchronous proxy fence": r"fence\.proxy\.async",
    "TMA store commit": r"cp\.async\.bulk\.commit_group",
    "TMA store wait": r"cp\.async\.bulk\.wait_group",
    "FP32 exponential": r"ex2\.approx(?:\.ftz)?\.f32",
}
G2S_EVENT = r"cp\.async\.bulk\.tensor\.2d\.shared(?:::cta|::cluster)?\.global"
EXPECT_EVENT = r"mbarrier\.arrive\.expect_tx"


def entries(ptx: str) -> dict[str, str]:
    ptx = re.sub(r"/\*.*?\*/", "", ptx, flags=re.S)
    ptx = re.sub(r"//[^\n]*", "", ptx)
    if not re.search(r"(?m)^\s*\.target\s+sm_100a\b", ptx):
        raise ValueError("wrong SM100a target")
    found = {}
    for match in re.finditer(r"\.entry\s+([^\s(]+)\s*\(", ptx):
        start = ptx.find("{", match.end())
        if start < 0:
            raise ValueError("unterminated PTX entry")
        depth, end = 1, start + 1
        while end < len(ptx) and depth:
            depth += (ptx[end] == "{") - (ptx[end] == "}")
            end += 1
        if depth:
            raise ValueError("unterminated PTX body")
        if match[1] in found:
            raise ValueError(f"duplicate PTX entry {match[1]}")
        found[match[1]] = ptx[start:end]
    if not found:
        raise ValueError("no PTX entry functions")
    return found


def check_ptx(ptx: str) -> list[str]:
    try:
        found = entries(ptx)
    except ValueError as exc:
        return [str(exc)]
    problems = []
    for symbol in KERNELS:
        body = found.get(symbol)
        if body is None:
            problems.append(f"missing {symbol} entry")
            continue
        if symbol != "probe" and not re.search(r"ex2\.approx(?:\.ftz)?\.f32", body):
            problems.append(f"missing FP32 exponential in {symbol}")
        if symbol in TMA:
            for label, pattern in TMA_REQUIRED.items():
                if not re.search(pattern, body):
                    problems.append(f"missing {label} in {symbol}")
            # Textual guard against copy-before-expect_tx, which can race
            # the TMA completion against registration of expected bytes.
            unpaired = 0
            for event in re.findall(rf"{EXPECT_EVENT}|{G2S_EVENT}", body):
                if event.startswith("mbarrier"):
                    unpaired += 1
                elif unpaired == 0:
                    problems.append(f"TMA load before expected bytes in {symbol}")
                    break
                else:
                    unpaired -= 1
            if unpaired:
                problems.append(f"unmatched TMA expected-byte registration in {symbol}")
        if symbol == "causal_softmax_tma_online":
            for label, pattern in {
                "parity wait": r"mbarrier\.try_wait\.parity",
                "warp butterfly": r"shfl\.sync\.bfly\.b32",
                "FP16 load conversion": r"cvt\.f32\.f16",
                "FP16 rounded output": r"cvt\.rn\.f16\.f32",
            }.items():
                if not re.search(pattern, body):
                    problems.append(f"missing {label} in {symbol}")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    args = parser.parse_args()
    problems = check_ptx(args.ptx.read_text())
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"PASS PTX: sm_100a, {len(KERNELS)} entries, per-entry 2D TMA/barriers/FP32")
