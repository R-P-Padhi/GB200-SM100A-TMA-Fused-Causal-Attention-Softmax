# Static code generation evidence

Generated on 2026-11-02 with Rust 1.99.1, official NVIDIA cuda-oxide v0.3.0
(`99e571f49baaefe4cd69157950c89ac9d4012728`), CUDA 13.0 `ptxas`,
target `sm_100a`. The Linux build machine had **no Blackwell GPU**.

- `sm100a.ptx`: merged source's PTX for six entries (probe plus five variants).
- `sm100a.cubin`: assembled `sm_100a` device image; not executed.
- `sm100a.ptxas.log`: raw assembler report, including register and spill counts.
- `source-sha256.json`: hashes of the final Rust, Python, pinned manifest and documentation inputs.

Reproduce with `cd gpu && cargo oxide build --arch sm_100a`, then from
the repository root:

```bash
python3 scripts/check_ptx.py gb200_tma_causal_softmax.ptx
python3 scripts/check_resources.py gb200_tma_causal_softmax.ptx \
  --output sm100a.cubin --log sm100a.ptxas.log
```

PTX local scratch and assembler stack frames (32/40 bytes for TMA variants)
are **not zero**, even though every entry reports **zero spill stores and zero
spill loads**. These instructions may have runtime costs; this evidence proves
assembly/resource constraints only, not correctness, speed or race freedom.
