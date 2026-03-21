

# HiDiffusion Benchmark Toolkit

一个基于开源 `HiDiffusion` 的工程化推理评测工具，用于在真实 GPU 环境下对比 baseline diffusion inference 与启用 HiDiffusion 后的推理表现，并根据显存预算与目标分辨率输出推荐推理策略。

本项目不训练新的扩散模型，也不提出新的生成算法。它的重点是将开源方法复现、封装并整理为一套可复现、可量化、可展示的推理 benchmark 与配置辅助工具，适合作为扩散模型推理优化方向的工程化项目。

---

## 项目目标

本项目围绕两个核心目标展开：

1. 提供一个可复现的 benchmark runner，用于公平对比 baseline diffusion inference 与启用 HiDiffusion 后的性能差异。
2. 提供一个 auto config 工具，根据显存预算和目标分辨率，输出适合实际部署的推理策略建议。

项目整体关注的是扩散模型推理阶段的工程问题，包括：

* 如何以统一条件进行性能对比
* 如何记录结构化实验结果
* 如何总结速度与显存之间的 trade-off
* 如何将开源方法整理为更适合复现、展示和二次扩展的工具链

---

## 项目特点

### 1. 面向推理阶段的工程化 benchmark

项目提供统一的 benchmark 脚本，用于在相同模型、相同 prompt、相同 seed、相同步数和相同 scheduler 条件下，对比 baseline 与 HiDiffusion 的推理表现。

重点关注的不是“是否能出图”，而是：

* 延迟
* 吞吐
* 峰值显存
* 成功率
* 失败原因
* 运行环境元数据

这使得项目更接近实际工程中的性能验证流程，而不是简单运行 demo。

### 2. 可复现的结构化输出

benchmark 运行后会输出多种结构化产物，便于分析、复盘和展示，包括：

* `raw.csv`：逐次运行的原始性能数据
* `report.md`：自动生成的结果摘要
* `metadata.json`：运行环境与实验配置
* `samples/*.png`：样例生成图

这些输出既便于后续可视化分析，也适合用于简历、面试或项目展示材料整理。

### 3. 面向部署决策的 auto config

除了 benchmark，本项目还提供一个规则型 auto config 工具，用于根据显存预算与目标分辨率给出推荐推理配置。

输出内容包括：

* 是否启用 HiDiffusion
* 是否启用 xFormers
* 是否启用 CPU offload
* 是否启用 VAE tiling
* 推荐 steps

这部分采用规则型策略而不是学习型策略，主要考虑：

* 结果可解释
* 行为易复现
* 更适合作为本科阶段的工程项目交付
* 后续可直接结合 benchmark 数据进行阈值校准

---

## 功能概览

### Benchmark Runner

`scripts/benchmark/run_benchmark.py`

用于执行 baseline 与 HiDiffusion 的对比测试，支持：

* 指定 `model_id`
* 指定分辨率 `height` / `width`
* 指定 `steps`
* 指定 `seeds`
* 指定 `scheduler`
* baseline / `enable_hidiffusion` 对比
* 输出 `raw.csv`
* 输出 `report.md`
* 输出 `metadata.json`
* 输出样例图 `samples/*.png`
* 支持 `dry_run` / smoke test
* 支持真实 GPU 推理 benchmark

### Auto Config

`scripts/benchmark/auto_config.py`

用于根据输入条件生成推荐配置，输出 JSON 结果，字段包括：

* `enable_hidiffusion`
* `enable_xformers`
* `enable_cpu_offload`
* `enable_vae_tiling`
* `steps`

### 配套工程内容

项目同时包含一组用于复现与展示的辅助内容：

* `scripts/benchmark/promptset_v1.json`
* `tests/test_smoke.py`
* `docs/resume_and_interview.md`
* `README.md` 使用说明
* `requirements.txt` 补充依赖

此外，项目还处理了本地脚本导入问题，使 `run_benchmark.py` 可以直接导入本地 `hidiffusion` 包并完成真实实验。

---

## 评测原则

为了让 benchmark 更接近公平对比，baseline 与 HiDiffusion 之间只切换 `enable_hidiffusion` 开关，其余条件保持一致，包括：

* 固定 prompt
* 固定 seed
* 固定 steps
* 固定 scheduler
* 固定模型

这种设置的目的是尽可能减少无关变量干扰，使实验结果更能反映 HiDiffusion 在推理阶段带来的实际影响。

---

## 采集指标

benchmark 主要记录以下指标：

* `latency_sec`
* `throughput_img_s`
* `peak_vram_mb`
* `success`
* `fail_reason`
* 运行环境元数据

这些指标覆盖了扩散模型推理评测中最常见的几个维度：速度、吞吐、显存占用与可运行性。

---

## 项目结构

```text
.
├── scripts/
│   └── benchmark/
│       ├── run_benchmark.py
│       ├── auto_config.py
│       └── promptset_v1.json
├── tests/
│   └── test_smoke.py
├── docs/
│   └── resume_and_interview.md
├── outputs/
│   └── ...
├── README.md
└── requirements.txt
```

---

## 运行环境

本项目已在真实 GPU 环境下完成验证，示例环境如下：

* GPU: `NVIDIA A100-SXM4-40GB`
* Python: `3.12.7`
* Torch: `2.5.0+cu124`
* Diffusers: `0.37.0`

示例 benchmark 模型：

* `runwayml/stable-diffusion-v1-5`

示例调度器：

* `DDIM`

---

## 使用方式

### 1. 运行 benchmark

可以通过 `run_benchmark.py` 对 baseline 与 HiDiffusion 进行统一条件下的对比测试。

典型流程如下：

1. 指定模型、分辨率、步数、scheduler 和 seed
2. 运行 baseline
3. 运行启用 HiDiffusion 的配置
4. 自动收集性能指标
5. 输出结构化结果文件

该脚本既可用于小规模 smoke test，也可用于真实 GPU 环境下的多分辨率 benchmark。

### 2. 生成推荐配置

可以通过 `auto_config.py` 输入显存预算和目标分辨率，获得推荐推理策略。

这适用于如下场景：

* 在资源受限环境中快速选择推理参数
* 在不同分辨率下决定是否启用 HiDiffusion
* 为后续部署策略或实验脚本提供初始配置

---

## 项目价值

这个项目的价值不在于提出新模型，而在于把一个开源扩散方法真正落地为可运行、可对比、可展示的工程化工具。

它体现的能力主要包括：

* 开源方法复现能力
* 扩散模型推理流程理解
* benchmark 设计与公平对比意识
* 结构化实验产物整理能力
* 工程脚本开发与工具化封装能力
* 对速度与显存 trade-off 的分析能力

相比只运行官方示例，本项目更强调：

* 可重复验证
* 可量化比较
* 可扩展
* 可用于简历与面试表达

---

## 适用场景

本项目适合以下用途：

* 扩散模型推理优化方向的课程项目或个人项目
* 简历中的生成式 AI / diffusion 工程项目展示
* 面试中介绍 benchmark、性能评测和推理优化能力
* 后续扩展到 SDXL、inpainting、ControlNet 等场景的基础框架

---

## 当前范围与后续扩展

当前项目主要围绕 `Stable Diffusion v1.5` 的推理 benchmark 展开，并提供基础的自动配置能力。后续可以继续扩展的方向包括：

* 增加更多 prompt 与 seed
* 扩展到 `SDXL`
* 支持 `inpainting`、`ControlNet` 等场景
* 基于真实 benchmark 数据进一步校准 auto config 阈值
* 增加图表可视化与更完整的分析报告

---

## 总结

HiDiffusion Benchmark Toolkit 关注的是扩散模型推理阶段的工程化验证，而不是新模型训练。项目将开源方法复现、benchmark、自动配置和结果整理整合为一套统一工具链，用于分析高分辨率生成场景下的性能收益与资源代价，并为后续扩展和项目展示提供基础。

> 一个面向扩散模型推理优化的工程化 benchmark 与配置工具，用于复现 HiDiffusion、评测其性能表现，并生成可复现的结构化结果与推荐策略。
