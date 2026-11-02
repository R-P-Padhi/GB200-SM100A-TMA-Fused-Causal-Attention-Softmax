#!/usr/bin/env python3
"""Hash the checked-in sources so CI artifacts identify their exact inputs."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = [root / name for name in [
    "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "README.md", "LICENSE",
    "THIRD_PARTY_NOTICES.md", ".gitignore", ".cargo/config.toml", ".github/workflows/ci.yml",
]]
for directory in ["gpu", "host", "reference", "scripts", "docs"]:
    files.extend(p for p in (root / directory).rglob("*")
                 if p.is_file() and not any(part in {"target", "__pycache__", ".git"} for part in p.parts))
manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files))}
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, help="write manifest to this file")
args = parser.parse_args()
result = json.dumps(manifest, indent=2) + "\n"
if args.output:
    args.output.write_text(result)
else:
    print(result, end="")
