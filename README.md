

# HiDiffusion Benchmark Toolkit

这是一个围绕开源 `HiDiffusion` 搭建的小型推理评测工具。项目主要关注扩散模型在推理阶段的性能表现，对比普通推理流程和启用 HiDiffusion 之后在速度、显存占用和稳定性上的差异，并在此基础上提供一个简单的配置推荐脚本。

这个仓库不涉及新模型训练，也不是去改造扩散模型本身。更准确地说，它是在已有开源方法的基础上，把推理测试、结果记录和配置选择整理成一套比较完整、能重复跑起来的工具链。

---

## 项目内容

关心的问题通常是：

* 同样的模型和参数下，开启某种优化后到底快了多少
* 显存占用有没有明显变化
* 在高分辨率下还能不能稳定跑通
* 不同优化选项组合起来之后，收益和代价分别是什么

项目的重点在“推理流程跑起来表现”。

项目主要有两部分：

1. 一个 benchmark 脚本，用来在统一条件下对比 baseline 和 HiDiffusion
2. 一个 auto config 脚本，根据显存预算和目标分辨率给出一组较保守的推荐配置

---

## 目前包含的内容

### 1. Benchmark Runner

`scripts/benchmark/run_benchmark.py`

这个脚本负责执行实际的推理测试。
在 baseline 和 HiDiffusion 两种设置下，尽量保持其他条件一致，例如：

* 相同模型
* 相同 prompt
* 相同 seed
* 相同步数
* 相同 scheduler

这样做的目的很简单，就是尽量把变量控制住，让结果更接近“是否启用 HiDiffusion”本身带来的差异，而不是混入别的影响因素。

运行后会记录一些比较常用的指标，包括：

* `latency_sec`
* `throughput_img_s`
* `peak_vram_mb`
* `success`
* `fail_reason`

除此之外，还会保存运行配置、环境信息以及生成样例，方便后面回头看。

---

### 2. Auto Config

`scripts/benchmark/auto_config.py`

这个脚本做的事情比较简单：
输入显存预算和目标分辨率，输出一份推荐配置。

当前输出的字段包括：

* `enable_hidiffusion`
* `enable_xformers`
* `enable_cpu_offload`
* `enable_vae_tiling`
* `steps`

这里没有做成学习型方法，而是先用规则来判断。原因主要是现阶段这个项目更偏工具化和可复现，规则型策略更容易解释，也更容易根据后续 benchmark 数据继续手动调整。

---

## 项目结构

```text
.
├── scripts/
│   └── benchmark/
│       ├── run_benchmark.py
│       ├── auto_config.py
│       └── promptset_v1.json
├── scripts/
│   └── ablation/
│       ├── gen_manifest.py
│       ├── run_one.py
│       ├── run_manifest.py
│       ├── aggregate.py
│       └── plot.py
├── tests/
│   └── test_smoke.py
├── docs/
│   └── ...
├── outputs/
│   └── ...
├── README.md
└── requirements.txt
```

---

## 输出结果

benchmark 跑完之后，会生成一组结构化结果，主要包括：

* `raw.csv`：每次运行的原始记录
* `report.md`：自动汇总的简要结果
* `metadata.json`：环境和实验配置
* `samples/*.png`：生成样例图

我保留这些输出，主要是为了后续分析方便。
比如想画图、做统计，或者查某次运行为什么失败，直接看这些文件就够了。

---

## 评测时的基本原则

为了尽量让对比更公平，baseline 和 HiDiffusion 之间只切换 `enable_hidiffusion` 这个开关，其他条件尽量保持不变。

也就是说，测试时会固定：

* prompt
* seed
* steps
* scheduler
* model

这样虽然不能保证完全消除所有波动，但至少可以把实验控制在一个比较清楚的范围内。

---

## 运行环境

这个项目已经在真实 GPU 环境下跑通过。
其中一组示例环境如下：

* GPU: `NVIDIA A100-SXM4-40GB`
* Python: `3.12.7`
* Torch: `2.5.0+cu124`
* Diffusers: `0.37.0`

示例测试模型：

* `runwayml/stable-diffusion-v1-5`

示例调度器：

* `DDIM`

---

## 怎么使用

## 1. 运行 benchmark

可以直接使用 `run_benchmark.py` 对 baseline 和 HiDiffusion 做对比测试。

一个比较典型的流程是：

1. 指定模型、分辨率、步数、scheduler 和 seed
2. 跑 baseline
3. 跑启用 HiDiffusion 的版本
4. 自动记录性能指标
5. 输出结果文件

它既可以用来做简单的 smoke test，也可以在真实 GPU 环境下做更完整的 benchmark。

---

## 2. 生成推荐配置

如果只是想先快速得到一组可用的推理设置，可以用 `auto_config.py`。

它更适合下面这种情况：

* 显存预算比较明确，但还没决定具体开哪些选项
* 不同分辨率下想先拿一组初始配置试跑
* 后面准备把推荐结果继续接到其他实验脚本里

---

## Ablation Matrix Workflow

除了 baseline 和 HiDiffusion 的直接对比，这个仓库里现在也整理了一套消融实验流程，用来系统测试不同优化选项组合的效果。

目前覆盖的组合主要包括：

* baseline
* HiDiffusion
* xformers
* cpu_offload
* vae_tiling


### 相关文件

* `experiments/ablation_manifest.yaml`：实验矩阵定义
* `scripts/ablation/gen_manifest.py`：生成不同 profile 的 manifest
* `scripts/ablation/run_one.py`：单配置、单进程执行
* `scripts/ablation/run_manifest.py`：批量展开并运行整个实验矩阵
* `scripts/ablation/aggregate.py`：聚合结果，输出 `ablation.csv`、`ablation_report.md`、`recommended_strategy.json`
* `scripts/ablation/plot.py`：根据聚合结果画图

### 快速验证

```bash
python scripts/ablation/gen_manifest.py \
  --profile smoke \
  --model_id "runwayml/stable-diffusion-v1-5" \
  --out experiments/ablation_manifest.yaml

python scripts/ablation/run_manifest.py \
  --manifest experiments/ablation_manifest.yaml \
  --dry_run 1

python scripts/ablation/aggregate.py \
  --run_dir results/ablation/runs/<run_id>

python scripts/ablation/plot.py \
  --csv results/ablation/ablation.csv
```

### 正式 P0 实验

```bash
python scripts/ablation/gen_manifest.py \
  --profile p0 \
  --model_id "runwayml/stable-diffusion-v1-5" \
  --out experiments/ablation_manifest.yaml

python scripts/ablation/run_manifest.py \
  --manifest experiments/ablation_manifest.yaml

python scripts/ablation/aggregate.py \
  --run_dir results/ablation/runs/<run_id>

python scripts/ablation/plot.py \
  --csv results/ablation/ablation.csv
```
