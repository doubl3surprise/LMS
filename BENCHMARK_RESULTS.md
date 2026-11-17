# Mem0 + SGLang LoCoMo 10 Benchmark Results

## Cross-Platform Evaluation

### DGX (NVIDIA Spark, GB10, TP=1)
- **Overall**: 123/152 = **80.9%**
- temporal: 33/37 = 89.2%
- single-hop: 68/82 = 82.9%
- open-domain: 10/13 = 76.9%
- multi-hop: 12/20 = 60.0%

Timestamp: 2026-11-16, Run ID: locomo_results_20261116_212630

### 5880ada (4x RTX 5880 Ada, TP=4)
- **Overall**: 126/152 = **82.9%**
- temporal: 33/37 = 89.2%
- single-hop: 59/70 = 84.3%
- open-domain: 10/13 = 76.9%
- multi-hop: 24/32 = 75.0%

Timestamp: 2026-11-17, Run ID: locomo_eval_full_20261117_182108

## Performance

| Metric           | 5880ada | DGX    | Ratio |
|------------------|---------|--------|-------|
| Ingestion (419)  | 13m24s  | ~13m   | ~1x   |
| Eval (152 Q)     | 6m19s   | ~72m   | ~11x faster |
| Throughput       | 2.5 Q/s | 0.2 Q/s | **11x** |

> DGX: Qwen3.8-27B on single GB10 chip, bandwidth-limited.
> 5880ada: 4x RTX 5880 Ada TP=4, faster per-token throughput.

## Setup

Both platforms use identical stack:
- Mem0 memory layer (FastEmbed + Qdrant)
- SGLang serving endpoint
- Same locomo10.json dataset (419 conversation chunks, 152 questions)
