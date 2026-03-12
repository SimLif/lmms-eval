# Qwen3.5 医学评估分析报告

## 概述

本文档记录 Qwen3.5-2B/4B/9B 在 `med_eval`（17 个医学任务）上的全量评估结果及 9B 模型异常表现的根因分析。

## 评估配置

| 配置项 | 值 |
|--------|-----|
| 任务集 | `med_eval`（17 个医学任务） |
| GPU | 8 × A800 80GB |
| batch_size | 1 |
| max_pixels | 3211264 (576 × 28 × 28) |
| enable_thinking | False（三个模型一致） |
| max_new_tokens | 1024（任务级配置） |
| transformers | 5.2.0 |

## 评估结果

### Multimodal VQA

| Task | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B |
|------|-----------|-----------|-----------|
| MMMU-Med | 45.33 | 48.67 | 46.00 |
| PMC-VQA | 52.80 | 55.25 | 55.35 |
| OmniMedVQA | 76.51 | 76.58 | **55.64** |
| MedXpertQA | 24.40 | 33.30 | 35.95 |
| **Avg** | **49.76** | **53.45** | **48.23** |

### Text QA

| Task | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B |
|------|-----------|-----------|-----------|
| MMLU-Med | 62.27 | 77.34 | 75.73 |
| PubMedQA | 73.20 | 76.60 | 76.40 |
| MedMCQA | 45.71 | 57.78 | 61.10 |
| MedQA | 48.08 | 60.17 | **24.43** |
| MedBullets-4 | 44.81 | 53.90 | **25.97** |
| MedBullets-5 | 41.88 | 44.16 | **19.48** |
| MedXpertQA | 10.45 | 13.71 | 18.86 |
| SuperGPQA | 18.98 | 30.24 | 34.30 |
| **Avg** | **43.17** | **51.74** | **42.04** |

### Report Generation

| Task | Metric | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B |
|------|--------|-----------|-----------|-----------|
| IU-XRay | BLEU4 | 11.84 | 11.81 | 9.21 |
| IU-XRay | RaTEScore | 64.50 | 64.80 | 64.15 |
| MIMIC-CXR | BLEU4 | 8.39 | 8.71 | 8.13 |
| MIMIC-CXR | RaTEScore | 57.34 | 57.36 | 58.11 |

## 9B 异常表现分析

### 现象

Qwen3.5-9B 在多个选择题任务上分数异常低，部分接近随机水平（MedQA 24.43% ≈ 随机 25%），与预期的"更大模型更强"趋势相悖。

### 根因：9B 在 non-thinking 模式下的 instruction following 缺陷

#### 1. 输出格式对比

prompt 中明确要求 "Answer with the option's letter from the given choices directly."，三个模型行为差异显著：

```
# 2B / 4B 的典型输出
B

# 9B 的典型输出（2000-4000 字符）
Based on the clinical presentation, this patient is experiencing...
[长段落推理过程]
...Therefore, the resident's primary responsibility is to inform...
B
```

缓存数据统计（medqa_usmle, rank0, 前 20 个样本）：

| 模型 | 响应长度 > 50 字符 | 典型长度 |
|------|-------------------|---------|
| 4B | 0/20 | 1 字符 |
| 9B | 19/20 | 2000-4000 字符 |

#### 2. 确认 enable_thinking=False 已正确生效

排查过程：

1. **代码验证**：`qwen3_5.py` 中 `enable_thinking` 默认为 `False`，eval 脚本未传入该参数，三个模型都使用默认值
2. **Chat template 验证**：三个模型的 tokenizer chat template 完全一致。`enable_thinking=False` 时在 prompt 末尾插入空的 `<think>\n\n</think>\n\n` 块，告诉模型"思考已结束，直接输出答案"
3. **Tokenizer 配置验证**：`<think>`/`</think>` token 在三个模型中配置一致（`special=False`）
4. **输出验证**：9B 缓存中 0/320 个样本包含 `<think>` 标签

结论：**thinking 模式确实已关闭，9B 没有进入 thinking mode。**

#### 3. 问题本质

9B 模型在 non-thinking 模式下**自发生成冗长推理文本**（不带 `<think>` 标签），这是模型本身的行为特性，不是配置问题。属于 **instruction following 能力缺陷**。

#### 4. 推理过程导致准确率下降

9B 不仅格式不对，推理过程本身也降低了准确率（overthinking）：

- 长推理中讨论多个选项，最终选错
- 分析过程中改变主意，最终答案与初始直觉不同
- MedQA 24.43% 接近随机水平（25%），说明推理过程实质上是噪声

#### 5. 答案提取逻辑验证

```
模型输出
  → take_first filter（默认）
  → process_results()
    → parse_reasoning_answer(strict=False)
      → 尝试 <answer> 或 \boxed{} 标签 → 找不到 → 返回原始全文
    → parse_multi_choice_response()
      → 在全文中搜索 (A)/(B)/(C)/(D)
      → 返回最后出现的选项字母
```

提取逻辑本身没有 bug。**分数低是因为模型给出的答案本身就是错的**，不是提取错误。

## 运行时间

| 模型 | 推理开始 | 推理结束 | 总耗时 |
|------|---------|---------|-------|
| Qwen3.5-2B | 03-03 20:26 | 03-03 22:25 | ~2h |
| Qwen3.5-4B | 03-03 22:28 | 03-04 07:01 | ~8.5h |
| Qwen3.5-9B | 03-04 07:05 | 03-05 08:29* | ~25h |

*9B 推理完成后 metric 计算卡住超过 2.5h，kill 后利用缓存重跑 metric 完成。

## 运维记录

### tqdm 进度条导致 Claude Code 冻死

**问题**：执行 `tail -N` 读取包含 tqdm 进度条的日志文件时，Claude Code 界面完全冻住，Ctrl+C 无效。

**原因**：tqdm 用 `\r`（回车符）更新进度，重定向到文件后，一个 `\n` 分隔的"行"包含成千上万次进度条更新，单行可达数 MB。`tail -80` 按行读取，实际输出可能几十 MB。

**解决方案**：
```bash
# 按字节数读取，不按行数
timeout 3 tail -c 500 logfile.log
# 过滤进度条后读取
timeout 3 grep -v '█' logfile.log | tail -20
# 只查看文件元数据
ls -la logfile.log
```

### 9B metric 计算卡死

**现象**：9B 推理 100% 完成后，进程仍在 R 状态运行超过 2.5 小时，但日志文件不再更新，结果文件未生成。

**对比**：2B/4B 的 metric 计算几乎瞬间完成。

**解决方案**：kill 进程后利用缓存（232 个文件，29 个任务 × 8 rank，完整无缺）重跑，metric 计算在几分钟内完成。

## 后续建议

1. **9B 评估结果不可靠**：由于 instruction following 问题，9B 在选择题任务上的分数不反映其真实能力
2. **可能的改进方向**：
   - 使用 thinking mode（`enable_thinking=True`）+ 从 `<think>` 块后提取答案
   - 在 prompt 中使用 `\boxed{}` 格式要求（已有 `think` 模式的 post_prompt 配置）
   - 降低 `max_new_tokens` 限制冗长输出（但可能截断答案）
3. **Qwen3.5 系列特性**：较大模型在 non-thinking 模式下仍倾向详细推理，这是模型训练特性，非配置可解决

## 参考资料

### Qwen3.5 官方文档

- [Qwen3.5-9B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-9B) - 包含部署命令、推理参数推荐
- [Qwen3.5-4B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-2B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-2B) - 提到 2B 在 thinking 模式下更容易进入思维循环
- [Qwen3.5 官方博客: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5) - Qwen3.5 系列架构与设计理念
- [Qwen3 官方博客: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/) - Qwen3 系列的 thinking/non-thinking 双模式设计说明
- [Qwen 官方 Quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html) - 包含 `enable_thinking=False` 的 API 调用示例

### Thinking 模式相关 Issues 和讨论

- [vLLM #35574: Qwen3.5 Can not close thinking by "enable_thinking": false](https://github.com/vllm-project/vllm/issues/35574) - vLLM 下 Qwen3.5 关闭 thinking 的 bug 报告，后续评论确认可以正常工作
- [vLLM #17327: Qwen3 Usage Guide](https://github.com/vllm-project/vllm/issues/17327) - vLLM 官方的 Qwen3 系列使用指南，含 `enable_thinking` 配置方法
- [Ray #52979: Qwen3 models "enable_thinking: False" still returns thinking content](https://github.com/ray-project/ray/issues/52979) - Ray Serve 下 Qwen3 关闭 thinking 仍输出推理内容
- [ms-swift #5836: enable_thinking=False ignored on vLLM 0.8](https://github.com/modelscope/ms-swift/issues/5836) - ms-swift 框架下 thinking 关闭不生效的讨论
- [QwenLM/Qwen3 #1300: how to set enable_thinking=False in vllm deploy](https://github.com/QwenLM/Qwen3/discussions/1300) - 社区讨论如何在部署时关闭 thinking
- [Ollama #14502: Qwen3.5 instruct (Non-Thinking) versions](https://github.com/ollama/ollama/issues/14502) - 社区请求 Qwen3.5 的 non-thinking 专用版本，反映较大模型在 non-thinking 模式下仍倾向推理

### 社区分析与讨论

- [Reddit: Qwen3.5 4B: overthinking to say hello](https://www.reddit.com/r/LocalLLaMA/comments/1rj8x1q/qwen35_4b_overthinking_to_say_hello/) - 即使 4B 也有 overthinking 问题的讨论
- [Reddit: Qwen 3.5 Non-thinking Mode Benchmarks?](https://www.reddit.com/r/LocalLLaMA/comments/1riy5x6/qwen_35_nonthinking_mode_benchmarks/) - 社区请求 non-thinking 模式的 benchmark 数据
- [Reddit: Qwen3.5 9B and 4B benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1rirtyy/qwen35_9b_and_4b_benchmarks/) - 9B 与 4B 的性能对比讨论
- [Reddit: How to turn off thinking in Qwen 3](https://www.reddit.com/r/LocalLLaMA/comments/1ka67wo/heres_how_to_turn_off_thinking_in_qwen_3_add_no/) - `/no_think` 和 `/think` 动态切换指令的使用方法
- [Kaitchup: Qwen3 Instruct "Thinks" — When Token Budgets Silently Skew Benchmark Scores](https://kaitchup.substack.com/p/qwen3-instruct-thinks-when-token) - 深入分析 Qwen3 Instruct 在 non-thinking 模式下的 verbose 输出如何影响 benchmark 分数

### 部署与配置指南

- [Unsloth: Qwen3.5 How to Run Locally Guide](https://unsloth.ai/docs/models/qwen3.5) - 包含各尺寸模型的 thinking 默认行为说明：**0.8B/2B/4B/9B 默认关闭 reasoning，27B 及以上默认开启**
- [vLLM Discuss: Deployment example for Qwen3 with hybrid thinking](https://discuss.vllm.ai/t/deployment-example-for-a-qwen3-model-with-hybrid-thinking/1462) - vLLM 部署 Qwen3 hybrid thinking 模式的示例
- [Fireworks: Qwen3 Instruct vs Thinking vs Coder Model Selection Guide](https://fireworks.ai/blog/qwen-3-decoded) - Qwen3 系列不同变体的选型指南
- [LearnOpenCV: Qwen3 Run a Thinking LLM Locally](https://learnopencv.com/qwen3/) - Qwen3 本地运行教程，含 thinking 模式说明
- [Alibaba Cloud: Deep Thinking Documentation](https://www.alibabacloud.com/help/en/model-studio/deep-thinking) - 阿里云 API 下的 reasoning_content 参数说明

### 技术深度分析

- [Sebastian Raschka: Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch) - Qwen3 架构的深度技术解析
- [Qwen3-Omni Technical Report (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1) - Qwen3-Omni 技术报告，包含多模态训练与 thinking 模式的技术细节
