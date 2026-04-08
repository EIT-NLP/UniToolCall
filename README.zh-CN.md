<p align="center">
  <img src="images/logo.png" alt="UniToolCall logo" width="240" />
</p>

# UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents

**语言 / Languages:** [English](README.md) | **简体中文**

## 🚀概述

工具调用能力使大语言模型智能体能够通过结构化工具调用与外部系统交互。然而，现有研究在交互表示上不一致，对工具使用轨迹的结构分布关注不足，且评测基准互不兼容。本文提出 **UniToolCall**，一个统一工具学习的框架，从工具集构建、数据集生成到评测对全流程进行标准化。

该框架构建包含 **22k+** 工具的大规模工具池，以及 **390k+** 条实例的混合训练语料：融合 **10** 个标准化公开数据集与结构可控的合成轨迹，覆盖 **单跳、多跳、单轮与多轮** 交互，并显式建模 **串行与并行** 执行模式；针对多轮交互引入 **Anchor Linkage** 机制以强化跨轮依赖。我们还将 **7** 个公开基准转换为统一的 **Query-Action-Observation-Answer（QAOA）** 表示，并在函数调用、轮次与会话层级开展细粒度评测。

<p align="center">
  <img src="images/framework.png" alt="UniToolCall 框架示意图" width="92%" />
</p>

## 实验结果

在干扰项较多的 **Hybrid-20** 设定下，于本数据上微调的 Qwen3-8B 在工具使用上取得显著提升：UniToolCall 达到 **93.0%** 的单轮 Strict Precision，较 Qwen3-32B 高出 **20.3** 个百分点，验证了框架有效性。下图为评测结果。

<p align="center">
  <img src="images/performance.png" alt="UniToolCall 评测结果" width="92%" />
</p>

## 数据集

训练数据由 **两部分** 组成：(1) **Public转换**，(2) **Pipeline生成**。

Public转换数据集的完整内容见 Hugging Face：

**[huggingface.co/datasets/EIT-NLP/UniToolCall](https://huggingface.co/datasets/EIT-NLP/UniToolCall)**

Pipeline生成的完整数据集见本仓库：

`multi-hop_pipeline/data/`, `multi-turn_pipeline/data/`和 `single-hop_pipeline/data/`

## 工具集

构建训练数据集的工具列表见：

`tool_set/apis/toolset.json`

## 环境依赖

- **Python** `>=3.9`（见 [`pyproject.toml`](pyproject.toml)）。
- 第三方库清单见根目录 [`requirements.txt`](requirements.txt)，并与 [`pyproject.toml`](pyproject.toml) 里 `[project]` / `dependencies` 一致。

## 安装

```bash
cd UniToolCall
pip install -r requirements.txt
pip install -e .
```

## 推理与评估

以下脚本位于 `test_set/scripts/metrics/`。请先配置 **API 密钥**（环境变量）。

### 推理

在仓库根目录执行：

```bash
cd test_set/scripts/metrics
python generate_with_qwen_server_list.py
```

常用示例：

```bash
# api1 = SiliconFlow；api2 = OpenAI 官方；api3 = Anthropic；api4 = Gemini；server/sft = 本地 vLLM
python generate_with_qwen_server_list.py --mode api1 \
  --inputfile /path/to/benchmark_json_dir \
  --outputfile /path/to/predictions_dir
```

### 评估

```bash
cd test_set/scripts/metrics
python all_evaluation.py
```

批量评估：

```bash
python all_evaluation.py \
  --inputfile /path/to/predictions_dir \
  --outputfile /path/to/eval_results_dir \
  --gtfile /path/to/gt_json_dir
```

**数据构建Pipeline**的代码分别在 `multi-hop_pipeline/scripts/`、`multi-turn_pipeline/scripts/`、`single-hop_pipeline/scripts/` 下，请在对应目录执行，例如：`cd multi-hop_pipeline/scripts && python generate_via_api.py`。

## 目录结构

| 路径 | 说明 |
|------|-------------|
| `multi-hop_pipeline/` | 多跳轨迹生成、质检、增强、**标准化** |
| `multi-turn_pipeline/` | 多轮生成与**标准化** |
| `single-hop_pipeline/` | 单跳数据工具 |
| `test_set/` | 基准评测 |
| `tool_set/` | 工具语料 |
| `train_set/` | 训练数据准备 |
| `src/uni_toolcall/` | 公共模块（`paths`、`prompts`、`secrets`） |

**Pipeline目录**（`*_pipeline/`）通常含 `prompts/`、`outputs/`。`multi-hop_pipeline/` 还可含 `db/`（如用于采样的 `usage_stats.json`）。

`train_set/scripts/` 与 `test_set/scripts/` 下用 `convert/`、`analysis/`、`metrics/`、`toollist/` 等按用途分子目录。

## 许可证

见仓库根目录 [`LICENSE`](LICENSE) 文件（Apache License 2.0）。
