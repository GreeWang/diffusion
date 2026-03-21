# Benchmark Report

- Generated at: 2026-03-21T09:36:54+00:00
- Device: NVIDIA A100-SXM4-40GB
- Python/Torch/Diffusers: 3.12.7 / 2.5.0+cu124 / unavailable

## Aggregate Metrics

| resolution | hidiffusion | avg latency (s) | avg throughput (img/s) | avg peak VRAM (MB) | successes | failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 0 | 0.555 | 1.803 | 1322.822 | 6 | 0 |
| 1024x1024 | 0 | 0.696 | 1.436 | 1991.290 | 6 | 0 |
| 1536x1536 | 0 | 0.932 | 1.073 | 3105.402 | 6 | 0 |

## Failure Stats

- No failures recorded.

## Conclusion

This run generated a complete comparison table. Re-run on a GPU-backed environment to replace dry-run metrics with real latency and VRAM numbers.
