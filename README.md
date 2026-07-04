# Laguna XS 2.1 NVFP4 on GB10 / SM121 — vLLM v0.24.0 Native Serving

Reproducibility pack for serving **poolside/Laguna-XS-2.1-NVFP4** (33B MoE, 3B active, 256 experts top-8) on NVIDIA GB10 / Blackwell SM121.

## Benchmark Results

Measured with **vLLM `bench serve`** (official tool) and **lm-eval-harness** (EleutherAI).

### Decode Ramp — vLLM bench serve

`--dataset-name random --random-input-len 512 --random-output-len 256 --num-prompts 20 --ignore-eos`

| C | Output tok/s | Total tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | OK |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **43.79** | 131.38 | 120.5ms | 127.7ms | 22.4ms | 27.0ms | 20/20 |
| 2 | 84.81 | 254.43 | 81.3ms | 101.7ms | 23.3ms | 24.8ms | 20/20 |
| 4 | 143.94 | 431.81 | 115.3ms | 124.2ms | 27.5ms | 29.0ms | 20/20 |
| 8 | 203.22 | 609.65 | 128.7ms | 130.2ms | 34.7ms | 37.1ms | 20/20 |
| 16 | **266.95** | 800.84 | 155.8ms | 208.9ms | 46.4ms | 49.5ms | 20/20 |
| 32 | **495.45** | 1,486.34 | 1,808.8ms | 2,383.1ms | 55.5ms | 69.3ms | 32/32 |

### Prefill — vLLM bench serve

`--random-input-len 2048 --random-output-len 1`

| Input tokens | TTFT p50 | Prefill rate |
|---:|---:|---:|
| 2,048 | 228.2ms | **8,975 tok/s** |

### Reasoning — lm-eval-harness

Full GSM8K was run through the OpenAI-compatible **completions** endpoint to keep the model answer in visible text for scoring.

| Task | N-shot | Metric | Score | Samples | Artifacts |
|------|---:|-------|---:|---:|---|
| GSM8K | 0 | exact_match (flexible) | **40.11%** ±1.35% | 1,319 | `results/gsm8k-full-0shot-20260703/` |
| GSM8K | 0 | exact_match (strict) | 0.00% | 1,319 | `results/gsm8k-full-0shot-20260703/` |

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

## Container

```bash
docker run -d --gpus all --ipc=host --name sm121-laguna \
  -p 18080:8000 \
  -v /path/to/Laguna-XS-2.1-NVFP4:/models/model:ro \
  -e SERVED_MODEL_NAME="Laguna-XS-2.1-NVFP4" \
  -e MAX_MODEL_LEN=32768 \
  -e TOOL_CALL_PARSER=poolside_v1 \
  -e GPU_MEMORY_UTILIZATION=0.72 \
  -e MAX_JOBS=6 \
  -e FLASHINFER_NVCC_THREADS=2 \
  ghcr.io/r0b0tlab/sm121-vllm-v0240-nvfp4:kv-exp
```

### Critical: FlashInfer JIT Parallelism Guard

Without `MAX_JOBS=6` + `FLASHINFER_NVCC_THREADS=2`, ninja defaults to all 20 CPU cores during runtime MoE kernel compilation (97 CUTLASS objects), exhausting 121 GiB RAM and freezing the host.

## Benchmark Methodology

**Tools:** vLLM `bench serve` (decode/prefill), lm-eval-harness (quality). No custom scripts.

## Files

| Path | Description |
|------|-------------|
| `results/laguna_xs21_corrected.html` | HTML report (vllm bench + lm-eval) |
| `results/corrected_results.json` | Raw benchmark JSON |
| `docker/Dockerfile.kv-exp` | CUDA 13.0 + JIT deps + NVFP4 KV cache |
| `scripts/entrypoint.sh` | Container entrypoint + JIT guard |
| `scripts/benchmark_nvfp4_kv.py` | Benchmark harness |

## Appendix: Methodology & Testing Notes

This section documents methodology decisions and testing issues encountered during benchmark development. These notes explain how the final numbers were derived but are not part of the results themselves.

### A.1 — Why Published Tools Over Custom Scripts

The initial benchmark used a custom Python script that produced only 44-token outputs per request — too short to isolate decode throughput from prefill and scheduling overhead. The reported throughput conflated TTFT, first-token latency, and actual decode rate into a single misleading number.

**Correction:** vLLM `bench serve` (vLLM's official benchmark) with `--ignore-eos` and 256-token forced outputs provides clean steady-state decode measurement, along with standard percentiles (TTFT/ITL/TPOT at p50/p99).

### A.2 — GSM8K Strict vs Flexible Matching

GSM8K in lm-eval-harness ships with two scoring variants:

- **strict-match** — requires the model to emit `#### N` (exact format). Our score: **0%**
- **flexible-extract** — regex-sweeps the response for any number matching the gold answer. Our current full-set score: **40.11%**

The 0% strict is a formatting artifact — Laguna XS 2.1 writes answers in prose (e.g., "Final Answer: 9") rather than the `#### 9` format. The 40.11% flexible-extract reflects actual math reasoning capability in the latest full-set completions-endpoint run.

### A.3 — Few-Shot Attempt and Tokenizer Mismatch

An initial 5-shot GSM8K run produced near-random results. Root cause: lm-eval-harness's internal tokenizer length-checking conflicted with the model's tokenizer, causing prompt truncation in the few-shot examples. Zero-shot configuration avoids this and produces clean results.

## License

MIT. Model weights not redistributed; follow upstream poolside license.
