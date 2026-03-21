# HiDiffusion Benchmark Project

## Resume Bullet

- HiDiffusion inference benchmarking and adaptive configuration toolkit (`PyTorch`, `diffusers`): built a reproducible benchmark runner that compares baseline diffusion inference against HiDiffusion across multiple resolutions and seeds, records latency, throughput, peak VRAM, and failure rate into CSV/Markdown artifacts, and saves sample outputs for quick inspection.
- Engineering tooling: implemented `auto_config.py` to recommend `HiDiffusion`/`xformers`/CPU offload/VAE tiling/step-count strategies from a VRAM budget and target resolution, with reproducible metadata capture for faster debugging and reruns.

## Interview Talking Points

### 1. How did you keep the benchmark fair?

- Fixed prompt, seed, scheduler, steps, and guidance scale; only switched the HiDiffusion patch on or off.
- Used CUDA synchronization before and after inference timing to avoid async measurement bias.
- Kept warmup runs out of the final CSV and recorded software versions in metadata for reproducibility.

### 2. Why choose latency, throughput, peak VRAM, and failure rate?

- Those metrics map directly to product constraints in generative inference systems: responsiveness, throughput capacity, GPU cost, and reliability.
- They also align with HiDiffusion's value proposition: improving high-resolution generation efficiency without retraining.

### 3. Why is `auto_config` rule-based instead of learned?

- The two-week scope favors an explainable and reproducible policy over a fragile overfit heuristic.
- The thresholds can be calibrated from the benchmark CSV, and the JSON schema leaves room for future learned or probing-based policies.

## Two-Minute Project Story

I took the original HiDiffusion package and turned it into an interview-ready engineering project. The first piece is a benchmark runner that compares baseline diffusion inference to HiDiffusion under fixed seeds and prompts, then emits raw CSV, a Markdown report, and sample images. The second piece is an `auto_config` tool that turns a VRAM budget and target resolution into a practical inference strategy using switches like HiDiffusion, xformers, CPU offload, VAE tiling, and step count. The result is a small but complete workflow that emphasizes fairness, reproducibility, and operational trade-offs instead of just generating pretty demo images.
