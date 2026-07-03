# Laguna XS 2.1 NVFP4 on GB10 / SM121 — vLLM v0.24.0 Native Serving

Reproducibility pack for serving **poolside/Laguna-XS-2.1-NVFP4** (33B MoE, 3B active, 256 experts top-8) on NVIDIA GB10 / Blackwell SM121.

## Phase 1 Baseline Results

| Concurrency | Output tok/s | Output tokens | Success |
|---:|---:|---:|---:|
| 1 | **43.79** | 44 | 1/1 ✅ |
| 2 | 91.46 | 88 | 2/2 ✅ |
| 4 | 180.80 | 176 | 4/4 ✅ |
| 8 | 198.12 | 397 | 8/8 ✅ |
| 16 | 606.04 | 704 | 16/16 ✅ |
| 32 | **1,095.08** | 1,408 | 32/32 ✅ |

**Prefill:** 1,749 tok/s peak (734-token prompt)

**100% success rate across all concurrency levels.** Zero Marlin/emulation — pure FLASHINFER_CUTLASS MoE.

## Configuration

| Setting | Value |
|---------|-------|
| Model | poolside/Laguna-XS-2.1-NVFP4 |
| Architecture | LagunaForCausalLM (MoE) |
| Total / Active params | 33B / 3B |
| Layers | 40 (10 global + 30 SWA w=512) |
| Experts | 256 + 1, top-8 routing |
| Max context | 256K (served at 32K) |
| Quantization | compressed-tensors nvfp4-pack-quantized |
| MoE backend | FLASHINFER_CUTLASS |
| GEMM backend | FlashInferCutlassNvFp4LinearKernel |
| Attention | FlashInfer (global + sliding window) |
| KV cache | FP8 (baked in config) |
| Spec decode | None (Phase 1 baseline) |
| Tool call parser | poolside_v1 |

## Container

### Run

```bash
docker run -d --gpus all --ipc=host --name sm121-laguna \
  -p 18080:8000 \
  -v /path/to/Laguna-XS-2.1-NVFP4:/models/model:ro \
  -e SERVED_MODEL_NAME="Laguna-XS-2.1-NVFP4" \
  -e MAX_MODEL_LEN=32768 \
  -e TOOL_CALL_PARSER=poolside_v1 \
  -e REASONING_PARSER=poolside_v1 \
  -e GPU_MEMORY_UTILIZATION=0.72 \
  -e MAX_JOBS=6 \
  -e FLASHINFER_NVCC_THREADS=2 \
  ghcr.io/r0b0tlab/sm121-vllm-v0240-nvfp4:kv-exp
```

### Critical: FlashInfer JIT Parallelism Guard

Without `MAX_JOBS=6` + `FLASHINFER_NVCC_THREADS=2`, ninja defaults to all 20 CPU cores during runtime MoE kernel compilation (97 CUTLASS objects), exhausting 121 GiB RAM and freezing the host. This is baked into `entrypoint.sh`.

**Why this happens:** Laguna's `compressed-tensors` nvfp4-pack format routes to `FLASHINFER_CUTLASS` which JIT-compiles 97 SM121-specific MoE kernels on first run.

### Startup Timeline

| Phase | Duration |
|-------|----------|
| Checkpoint load (5 shards, 20 GiB) | 131.7s |
| torch.compile (mode 3) | 25.5s |
| MoE profiling/warmup | 630.0s |
| FlashInfer JIT (first run) | ~10 min |
| **Total cold start** | **~15 min** |
| Cached restart | ~3 min |

## Phase 2 — DFlash Implementation

**Goal:** Enable DFlash speculative decoding (up to 15 tokens/step).

**Status:** DFlash draft model downloaded (`Laguna-XS-2.1-DFlash`, 5-layer Llama-style speculator). The `dflash.py` module exists in vLLM v0.24.0, but `DFlashLagunaForCausalLM` architecture is not registered — needs PR #46853.

**Steps:**
1. Apply vLLM PR #46853 to register DFlash architecture
2. Mount DFlash draft model as second volume
3. Configure `SPECULATIVE_CONFIG={"method":"dflash","model_path":"/models/dflash"}`
4. Benchmark DFlash vs Phase 1 baseline (same ramp c1–c32)
5. Generate comparison report

## Files

| Path | Description |
|------|-------------|
| `docker/Dockerfile.kv-exp` | CUDA 13.0 + JIT deps + NVFP4 KV cache |
| `docker/vllm-pr46329.diff` | Lift SM100-only NVFP4 KV guard for SM121 |
| `scripts/entrypoint.sh` | Container entrypoint + JIT parallelism guard |
| `scripts/benchmark_nvfp4_kv.py` | Benchmark harness with power/decode/prefill telemetry |
| `scripts/audit_runtime.py` | SM121 runtime gate verification |
| `results/laguna_xs21_nvfp4_baseline.html` | Phase 1 HTML report |
| `results/benchmark_laguna_xs_nvfp4.json` | Raw benchmark data |

## Environment

- **Hardware:** NVIDIA GB10 (DGX Spark), SM121, 128 GB unified memory
- **vLLM:** v0.24.1.dev (ee0da84) built from source for SM121
- **CUDA:** 13.0
- **Driver:** 580.159.03
- **FlashInfer:** 0.6.12 with PR #3684
- **Key upstream PRs:** vLLM #46329 (NVFP4 KV guard lift), vLLM #46853 (DFlash — pending)

## License

Scripts and documentation: MIT. Model weights are not redistributed; follow upstream poolside/Laguna-XS-2.1 license.
