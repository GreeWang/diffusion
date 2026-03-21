# Benchmark Report

- Generated at: 2026-03-21T09:36:53+00:00
- Device: NVIDIA A100-SXM4-40GB
- Python/Torch/Diffusers: 3.12.7 / 2.5.0+cu124 / unavailable

## Aggregate Metrics

| resolution | hidiffusion | avg latency (s) | avg throughput (img/s) | avg peak VRAM (MB) | successes | failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 1 | 0.527 | 1.898 | 1164.084 | 6 | 0 |
| 1024x1024 | 1 | 0.661 | 1.512 | 1752.335 | 6 | 0 |
| 1536x1536 | 1 | 0.764 | 1.308 | 2732.753 | 6 | 0 |

## Failure Stats

- No failures recorded.

## Conclusion

This run generated a complete comparison table. Re-run on a GPU-backed environment to replace dry-run metrics with real latency and VRAM numbers.
