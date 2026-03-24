# Ablation Report

- Run directory: `results/ablation/runs/ablation_sd15_p0_20260324_merged`
- Total rows: 144
- Successful rows: 134

## Aggregate Metrics

| priority | exp_id | resolution | mean latency (s) | 95% CI | mean throughput (img/s) | mean peak VRAM (MB) | success rate | successful runs | total runs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | base+offload | 1024x1024 | 4.156 | 0.422 | 0.242 | 3190.1 | 100.00% | 6 | 6 |
| P0 | base+tiling | 1024x1024 | 2.295 | 0.139 | 0.437 | 3262.4 | 100.00% | 6 | 6 |
| P0 | base+xformers | 1024x1024 | 2.397 | 0.437 | 0.427 | 5081.2 | 100.00% | 6 | 6 |
| P0 | baseline | 1024x1024 | 2.211 | 0.412 | 0.462 | 5081.2 | 100.00% | 6 | 6 |
| P0 | hi+xformers | 1024x1024 | 2.359 | 1.084 | 0.469 | 5081.2 | 100.00% | 6 | 6 |
| P0 | hi+xformers+offload | 1024x1024 | 4.893 | 1.337 | 0.215 | 3190.1 | 100.00% | 6 | 6 |
| P0 | hi+xformers+tiling | 1024x1024 | 2.514 | 1.089 | 0.436 | 3262.4 | 100.00% | 6 | 6 |
| P0 | hidiffusion | 1024x1024 | 1.991 | 0.411 | 0.518 | 5081.2 | 100.00% | 6 | 6 |
| P0 | base+offload | 1536x1536 | 7.039 | 0.364 | 0.142 | 6231.3 | 100.00% | 6 | 6 |
| P0 | base+tiling | 1536x1536 | 5.279 | 0.231 | 0.190 | 3822.9 | 100.00% | 6 | 6 |
| P0 | base+xformers | 1536x1536 | 5.974 | 1.131 | 0.172 | 8122.5 | 100.00% | 6 | 6 |
| P0 | baseline | 1536x1536 | 4.886 | 0.121 | 0.205 | 8122.5 | 100.00% | 6 | 6 |
| P0 | hi+xformers | 1536x1536 | 4.257 | 1.214 | 0.249 | 8122.5 | 100.00% | 6 | 6 |
| P0 | hi+xformers+offload | 1536x1536 | 6.350 | 6.638 | 0.159 | 6231.3 | 33.33% | 2 | 6 |
| P0 | hi+xformers+tiling | 1536x1536 | 4.819 | 1.084 | 0.216 | 3824.9 | 100.00% | 6 | 6 |
| P0 | hidiffusion | 1536x1536 | 3.271 | 0.293 | 0.307 | 8122.5 | 100.00% | 6 | 6 |
| P0 | base+offload | 2048x2048 | 14.074 | 1.200 | 0.071 | 10489.1 | 100.00% | 6 | 6 |
| P0 | base+tiling | 2048x2048 | 12.302 | 0.228 | 0.081 | 4729.2 | 100.00% | 6 | 6 |
| P0 | base+xformers | 2048x2048 | 11.945 | 0.889 | 0.084 | 12380.2 | 100.00% | 6 | 6 |
| P0 | baseline | 2048x2048 | 11.205 | 0.077 | 0.089 | 12380.2 | 100.00% | 6 | 6 |
| P0 | hi+xformers | 2048x2048 | 6.556 | 1.074 | 0.155 | 12380.2 | 100.00% | 6 | 6 |
| P0 | hi+xformers+offload | 2048x2048 | nan | nan | nan | nan | 0.00% | 0 | 6 |
| P0 | hi+xformers+tiling | 2048x2048 | 7.282 | 1.290 | 0.140 | 4729.2 | 100.00% | 6 | 6 |
| P0 | hidiffusion | 2048x2048 | 5.815 | 0.518 | 0.173 | 12380.2 | 100.00% | 6 | 6 |

## Main Findings

- 1024x1024: HiDiffusion relative to baseline improved average latency by 10.0% with peak VRAM delta 0.0 MB.
- 1024x1024: adding xformers on top of HiDiffusion changed latency by -18.5%.
- 1536x1536: HiDiffusion relative to baseline improved average latency by 33.1% with peak VRAM delta 0.0 MB.
- 1536x1536: adding xformers on top of HiDiffusion changed latency by -30.2%.
- 2048x2048: HiDiffusion relative to baseline improved average latency by 48.1% with peak VRAM delta 0.0 MB.
- 2048x2048: adding xformers on top of HiDiffusion changed latency by -12.7%.

## Failure Breakdown

- `timeout_xformers_cpu_offload`: 10

## Recommendation

- Recommended configuration: `hidiffusion` at 1024x1024 with mean latency 1.991s and peak VRAM 5081.2 MB.
