#!/usr/bin/env python3
"""Optional Linux-only, driverless CUDA-13 static-build setup in .tools/."""
import concurrent.futures
import hashlib
import json
import platform
import shlex
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
REVISION = "99e571f49baaefe4cd69157950c89ac9d4012728"


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def download(item):
    path = TOOLS / "downloads" / item["url"].rsplit("/", 1)[1]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
        return path
    temporary = path.with_suffix(".download")
    with urllib.request.urlopen(item["url"], timeout=180) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
        temporary.unlink()
        raise RuntimeError(f"CUDA package SHA-256 mismatch: {item['package']}")
    temporary.replace(path)
    return path


def main():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise SystemExit("Use Linux x86_64 or GitHub Actions; Windows needs no local toolchain.")
    (TOOLS / "downloads").mkdir(parents=True, exist_ok=True)
    packages = json.loads((ROOT / "scripts/cuda-packages.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        archives = list(workers.map(download, packages))
    for archive in archives:
        run("dpkg-deb", "-x", str(archive), str(TOOLS / "cuda"))
    run("python3", "-m", "pip", "install", "--target", str(TOOLS / "clang"), "libclang==18.1.1")
    run("rustup", "toolchain", "install", "1.99.1", "--profile", "minimal", "--component", "rust-src,rustc-dev,llvm-tools,rustfmt,clippy")
    run("cargo", "+1.99.1", "install", "--git", "https://github.com/NVIDIA/cuda-rust.git",
        "--rev", REVISION, "--locked", "--force", "cargo-oxide")
    gcc_include = subprocess.check_output(["gcc", "-print-file-name=include"], text=True).strip()
    if not (Path(gcc_include) / "stddef.h").is_file():
        raise RuntimeError("GCC development headers not found")
    toolkit = TOOLS / "cuda/usr/local/cuda-13.0"
    settings = {
        "CUDA_TOOLKIT_PATH": str(toolkit),
        "CUDA_HOME": str(toolkit),
        "LIBCLANG_PATH": str(TOOLS / "clang/clang/native"),
        "BINDGEN_EXTRA_CLANG_ARGS": f"-isystem {gcc_include}",
        "RUSTC_BOOTSTRAP": "1",
    }
    lines = [f"export {name}={shlex.quote(value)}" for name, value in settings.items()]
    lines.append(f'export PATH={shlex.quote(str(toolkit / "bin"))}:"$PATH"')
    (TOOLS / "env.sh").write_text("\n".join(lines) + "\n")
    print("Ready: source .tools/env.sh && cd gpu && cargo oxide build --arch sm_100a")


if __name__ == "__main__":
    main()
