#!/usr/bin/env python3
"""Convert GPU/CPU benchmark key=value logs into typed, reviewable JSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_record(line: str) -> dict | None:
    line = line.strip()
    if line.startswith("BENCH "):
        source = "gpu"
        tokens = line.split()[1:]
    elif line.startswith("model="):
        source = "cpu"
        tokens = line.split()
    else:
        return None

    record: dict[str, str | float | int] = {"source": source}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or not value:
            continue
        if key in {"rows", "cols", "n", "streams", "queued", "calls_per_stream"}:
            record[key] = int(value)
        elif key.endswith(("_ms", "_GBps", "_per_s", "_pct", "_speedup")) or key == "scale" or key.startswith("speedup_"):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid numeric metric {key}={value}")
            record[key] = number
        else:
            record[key] = value

    if not ("median_ms" in record or "batch_p50_ms" in record):
        return None
    if source == "gpu" and "category" not in record:
        raise ValueError("GPU timing lacks category")
    if "kernel" not in record:
        raise ValueError("timing lacks kernel identifier")
    if record.get("rows", 0) <= 0 or record.get("cols", 0) <= 0:
        raise ValueError("timing lacks positive dimensions")
    return record


def parse_log(content: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(content.splitlines(), 1):
        try:
            row = parse_record(line)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("no timing records found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parsed = parse_log(args.log.read_text(encoding="utf-8"))
    result = {
        "source_file": args.log.name,
        "record_count": len(parsed),
        "cpu_only": all(row["source"] == "cpu" for row in parsed),
        "records": parsed,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
