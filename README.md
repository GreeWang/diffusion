# HiDiffusion 项目最终报告

## 1. 项目定位

本项目不是训练新的扩散模型，也不是提出新的生成算法，而是基于开源项目 `HiDiffusion` 完成一套可复现、可量化、可直接用于简历与面试展示的工程化推理评测工具。

项目核心目标有两部分：

1. 实现一个可复现的 benchmark runner，对比 baseline diffusion inference 与开启 HiDiffusion 之后的推理性能。
2. 实现一个 auto config 工具，根据显存预算与目标分辨率给出推理策略建议。

因此，这个项目的本质是：

- 复现开源方法的真实推理效果
- 将论文/开源项目包装成工程化工具链
- 用真实实验数据总结 HiDiffusion 的速度收益与显存代价

## 2. 修改内容

### 2.1 Benchmark Runner

我实现了 `scripts/benchmark/run_benchmark.py`，用于公平对比 baseline 与 HiDiffusion。

支持内容包括：

- 指定 `model_id`
- 指定分辨率 `height/width`
- 指定 `steps`
- 指定 `seeds`
- 指定 `scheduler`
- baseline / `enable_hidiffusion` 对比
- 输出 `raw.csv`
- 输出 `report.md`
- 输出 `metadata.json`
- 输出样例图 `samples/*.png`
- 支持 `dry_run` / smoke test
- 支持真实 GPU 推理 benchmark

对比原则保持一致：

- 固定 prompt
- 固定 seed
- 固定 steps
- 固定 scheduler
- 只切换 `enable_hidiffusion`

采集指标包括：

- `latency_sec`
- `throughput_img_s`
- `peak_vram_mb`
- `success`
- `fail_reason`
- 运行环境元数据

### 2.2 Auto Config 工具

我实现了 `scripts/benchmark/auto_config.py`，用于根据显存预算和目标分辨率输出推荐策略 JSON。

输出字段包括：

- `enable_hidiffusion`
- `enable_xformers`
- `enable_cpu_offload`
- `enable_vae_tiling`
- `steps`

这部分采用规则型策略，而不是学习型策略，目的在于：

- 可解释
- 易复现
- 适合本科生项目交付
- 可基于 benchmark 结果进一步校准

### 2.3 Prompt 集、文档与测试

我补充了以下工程化内容：

- `scripts/benchmark/promptset_v1.json`
- `tests/test_smoke.py`
- `docs/resume_and_interview.md`
- `README.md` 中的 benchmark / auto config 使用说明
- `requirements.txt` 中的补充依赖

同时修复了真实运行时的脚本导入问题，使 `run_benchmark.py` 可以直接导入本地 `hidiffusion` 包并完成真实实验。

## 3. 环境与模型

本次真实实验运行环境：

- GPU: `NVIDIA A100-SXM4-40GB`
- Python: `3.12.7`
- Torch: `2.5.0+cu124`
- Diffusers: `0.37.0`

真实 benchmark 使用模型：

- `runwayml/stable-diffusion-v1-5`

调度器：

- `DDIM`

## 4. 实验设置

为了保证对比公平，真实实验统一使用如下设置：

- 模型相同：`runwayml/stable-diffusion-v1-5`
- scheduler 相同：`ddim`
- steps 相同：`10`
- prompts 数量相同：`2`
- seeds 相同：`0,1`
- 仅切换 `enable_hidiffusion=0/1`

真实实验覆盖分辨率：

- `512x512`：smoke test
- `1024x1024`
- `1536x1536`
- `2048x2048`

其中，`1024/1536/2048` 为主要分析结果。

## 5. 真实实验结果

### 5.1 512x512 smoke test

这个配置主要用于验证真实推理链路，而不是用于证明 HiDiffusion 优势。

结果：

- baseline: `0.233s`, `4.299 img/s`, `3256 MB`
- HiDiffusion: `0.255s`, `3.926 img/s`, `5902 MB`

结论：

- 在非常小的分辨率和很少的步数下，HiDiffusion 没有体现优势
- 这说明小规模 smoke test 只适合验证链路，不适合作为最终结论依据

### 5.2 1024x1024

平均结果：

- baseline: `1.436s`, `0.699 img/s`, `5081 MB`
- HiDiffusion: `1.202s`, `0.835 img/s`, `7732 MB`

结论：

- 平均延迟下降约 `16.3%`
- 吞吐提升
- 峰值显存上升

### 5.3 1536x1536

平均结果：

- baseline: `4.230s`, `0.237 img/s`, `8122 MB`
- HiDiffusion: `2.588s`, `0.387 img/s`, `9445 MB`

结论：

- 平均延迟下降约 `38.8%`
- 吞吐明显提升
- 峰值显存上升，但增幅相对可控

### 5.4 2048x2048

平均结果：

- baseline: `10.579s`, `0.095 img/s`, `12380 MB`
- HiDiffusion: `4.899s`, `0.204 img/s`, `15025 MB`

结论：

- 平均延迟下降约 `53.7%`
- 吞吐提升非常明显
- 峰值显存上升到约 `15.0GB`

## 6. 核心结论

到目前为止，这个项目最重要的实验结论是：

1. HiDiffusion 的优势不是在低分辨率小样本场景下体现，而是在更高分辨率下变得明显。
2. 分辨率越高，HiDiffusion 的加速收益越明显：
   - 1024: `16.3%`
   - 1536: `38.8%`
   - 2048: `53.7%`
3. HiDiffusion 并不是“白送收益”，它带来了更高的显存占用。
4. 因此，HiDiffusion 更适合在高分辨率生成场景下作为速度优化手段使用，而不是在低分辨率小任务里默认开启。

一句话总结：

> HiDiffusion 在高分辨率推理场景下可以显著降低延迟、提升吞吐，但代价是更高的显存占用。

## 7. 这个项目的价值

这个项目的价值不在于“提出新模型”，而在于：

- 能把一个论文/开源项目真正复现到本机 GPU 上
- 能做公平 benchmark，而不是只跑官方 demo
- 能把结果整理成结构化产物：CSV、Markdown、样例图、策略 JSON
- 能从实验中总结 trade-off，而不是只说“效果好”
- 能把研究型方法包装成面向工程与面试展示的可交付项目

## 8. 当前可以写进简历的内容

可以概括为：

- 基于 HiDiffusion 开源项目完成复现、性能评测与推理配置工具开发
- 搭建扩散模型推理 benchmark 工具链，自动输出 latency、throughput、peak VRAM、失败率、样例图与 Markdown 报告
- 在 A100 40GB 上完成 `1024/1536/2048` 多分辨率真实 benchmark，HiDiffusion 相比 baseline 分别实现约 `16.3% / 38.8% / 53.7%` 的推理加速
- 观察到 HiDiffusion 在高分辨率场景下速度收益明显，但伴随显存占用上升

## 9. 当前局限与后续改进

当前项目仍然有一些局限：

- 真实 benchmark 目前主要在 `Stable Diffusion v1.5` 上完成
- prompt 数量和 seed 数量仍然不算特别大
- `auto_config` 的阈值规则还可以继续根据更多实验进一步校准
- 还没有覆盖 `sdxl`、`inpainting`、`controlnet` 等更复杂场景

如果继续扩展，优先级建议是：

1. 增加更多 prompt 和 seed
2. 在 `SDXL` 上复现同类实验
3. 根据真实 benchmark 数据进一步校准 `auto_config`
4. 补充可视化图表和更正式的总报告

## 10. 最终结论

本项目已经完成从“开源方法复现”到“工程化 benchmark 工具”的落地。

与仅仅跑通官方 demo 相比，我已经完成了：

- 真实 GPU 推理验证
- 多分辨率 benchmark
- 自动化报告输出
- 推理策略配置工具开发
- 简历与面试可复用材料整理


> 基于开源扩散模型方法完成复现、推理 benchmark、自动配置与工程化评测工具开发，并在真实 GPU 环境下完成多分辨率性能验证。
