# AGENTS.md — Laguna XS 2.1 NVFP4 SM121 vLLM

## Project

Serve **poolside/Laguna-XS-2.1-NVFP4** (33B MoE, 3B active, 256 experts top-8) on NVIDIA GB10 / SM121 using vLLM v0.24.0 built from source.

## Current Phase

- **Phase 1 complete** — baseline benchmark recorded (c1–c32, 100% success)

## Architecture Constraints

- **Native paths only** — no Marlin, no emulation, no fallback
- **SM121** = GB10 compute capability (12.1). Compile with `TORCH_CUDA_ARCH_LIST=12.1`
- **MAX_JOBS=6 + FLASHINFER_NVCC_THREADS=2** required at runtime — unbounded JIT freezes the host (97 CUTLASS MoE objects × 20 cores = OOM)

## Container Image

`ghcr.io/r0b0tlab/sm121-vllm-v0240-nvfp4:kv-exp` (SHA `b14ed31d0ca8`, 15.4 GB)

Built from: vLLM v0.24.0 + FlashInfer PR #3684 + vLLM PR #46329 (SM121 NVFP4 KV guard lift) + CUDA 13.0 + JIT deps (build-essential, ninja, cuda-nvcc, cuda-libraries-dev-13-0)

## Key Upstream PRs

| PR | Status | Purpose |
|----|--------|---------|
| FlashInfer #3684 | ✅ Applied | NVFP4 KV cache support |
| vLLM #46329 | ✅ Applied | Lift SM100-only NVFP4 KV guard for SM121 |

## Benchmark Protocol

1. Start container with model at `/models/model:ro`
2. Wait for JIT compilation (~15 min cold, ~3 min cached)
3. Run `scripts/benchmark_nvfp4_kv.py` against `http://127.0.0.1:18080`
4. Collect: ramp c1–c32, prefill measurements, sanity suite
5. Store results in `results/`

## File Layout

```
.
├── docker/
│   ├── Dockerfile.kv-exp       # CUDA 13.0 + JIT deps + NVFP4 KV
│   └── vllm-pr46329.diff       # SM100 guard lift patch
├── scripts/
│   ├── entrypoint.sh           # Container entrypoint + JIT guard
│   ├── audit_runtime.py        # SM121 verification
│   ├── benchmark_nvfp4_kv.py   # Benchmark harness
│   └── public_safety_scan.py   # Pre-commit secret scan
├── results/
│   ├── laguna_xs21_nvfp4_baseline.html
│   └── benchmark_laguna_xs_nvfp4.json
├── README.md
├── AGENTS.md
└── PRIVACY.md
```

## Publish Workflow

- GitHub: `r0b0tlab/laguna-xs2-nvfp4-sm121-vllm`
- Push directly to `main`
- Run `scripts/public_safety_scan.py` before every commit
