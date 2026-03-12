# Qwen-VL 系列医学评估综合分析

## 一、模型架构演进概览

| 维度 | Qwen2.5-VL (2025.01) | Qwen3-VL (2025.09-10) | Qwen3.5 (2025.12) |
|------|----------------------|----------------------|-------------------|
| **架构范式** | ViT + MLP merger + LLM（三模块拼接） | ViT + MLP merger + LLM（三模块拼接，升级 ViT） | ViT + LLM（仍有独立视觉编码器，但 Early Fusion 联合预训练） |
| **Vision Encoder** | 从零训练的动态分辨率 ViT，Window Attention | SigLIP-2 架构，DeepStack 多层特征融合 | 独立 ViT（24 层，patch_size=16），但视觉和文本从预训练开始联合优化 |
| **位置编码** | MRoPE（多维旋转位置编码） | Interleaved MRoPE（交错频率，更平衡的时空建模） | 继承 Qwen3-Next 架构 |
| **LLM 基座** | Qwen2.5 系列 | Qwen3 系列（~36T tokens 预训练） | Qwen3-Next 架构，Gated DeltaNet + Gated Attention 混合注意力 |
| **MoE 支持** | 无（仅 dense） | 有（30B-A3B, 235B-A22B） | 有（35B-A3B, 122B-A10B, 397B-A17B） |
| **上下文长度** | 131K | 256K（YaRN 扩展至 1M） | 256K（Plus 版本 1M） |
| **训练目标** | 视觉-语言对齐 + 指令微调 | 视觉-语言对齐 + SFT + thinking 双模式 | Agent 导向，工具调用、UI 交互、长程规划 |
| **patch_size** | 14 | 16 | — |

**关键架构变化**：Qwen2.5-VL → Qwen3-VL 是同范式内的**渐进升级**（换更强 ViT、加 DeepStack、改位置编码）；Qwen3-VL → Qwen3.5 则是**范式转换**（从"视觉模块拼接到语言模型"转变为"原生多模态联合预训练"）。

### InternVL 系列参照

| 维度 | InternVL2.5 (2024.12) | InternVL3 (2025.04) | InternVL3.5 (2025.08) |
|------|----------------------|---------------------|----------------------|
| **架构范式** | ViT-MLP-LLM（传统拼接） | ViT-MLP-LLM + Native Multimodal Pre-Training | ViT-MLP-LLM + Native Multimodal Pre-Training |
| **LLM 基座** | Qwen2.5 系列 | Qwen2.5 系列（但联合预训练后文本能力反超 Qwen2.5） | GPT-OSS / Qwen2.5 系列 |
| **后训练** | SFT | SFT + Mixed Preference Optimization (MPO) | **Cascade RL**（离线 RL + 在线 RL）+ Visual Consistency Learning (ViCO) |
| **特色** | 基础 VLM 能力 | 原生多模态预训练 + 工具使用、GUI Agent | 动态视觉分辨率路由 (ViR) + 推理强化 |

### 多模态训练与文本能力的 Trade-off：InternVL3 vs Qwen3-VL

多模态训练是否会损害文本能力，是 VLM 训练的核心问题之一。InternVL3 和 Qwen3-VL 对此给出了不同的回答和应对策略。

#### InternVL3：联合预训练增强文本能力

InternVL3 ([arXiv:2504.10479](https://arxiv.org/abs/2504.10479)) 提出 Native Multimodal Pre-Training，将语言预训练和多模态对齐合并为单一阶段，在预训练过程中交错混合多模态数据和纯文本数据，联合优化 ViT + MLP + LLM 全部参数。论文的关键发现是——多模态联合训练不但没有损害文本能力，反而增强了它：

> *"We compare InternVL3 with Qwen2.5 Chat models, whose corresponding pre-trained base models are employed as the initialization of the language component in InternVL3. **Benefitting from Native Multimodal Pre-Training, the InternVL3 series achieves even better overall text performance than the Qwen2.5 series.**"*

> *"This observed enhancement in language capabilities primarily arises from several factors, including **the integration of approximately 25% pure-language data, joint parameter optimization during native multimodal pre-training**, and the extensive use of high-quality textual corpora during the subsequent post-training stage. Such an approach not only strengthens multimodal comprehension but also **significantly enhances language proficiency**. Consequently, even when derived from identical pre-trained base models, the integrated multimodal and pure-text training strategy employed by InternVL3 results in **substantially improved performance in language capabilities compared to the specialized training pipeline designed for pure-text tasks** used by the Qwen2.5 chat models."*

#### Qwen3-VL：工程手段缓解文本能力下降

Qwen3-VL ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)) 仍采用传统的分阶段预训练范式（warm-up → 8K → 32K → 256K 逐步扩展上下文），但通过两个关键工程手段来平衡多模态和文本能力：

**1. Square-root Reweighting（预训练阶段）**：将 per-sample loss 替换为 square-root-normalized per-token loss，平衡文本和多模态数据在训练中的贡献：

> *"To balance text-only and multimodal learning objectives, we apply **square-root reweighting**, which moves from a per-sample loss to a **square-root-normalized per-token loss**. This approach better balances the contributions of text and multimodal data during training, **boosting multimodal performance without compromising text capabilities**."*

**2. On-Policy Distillation / oPD（后训练阶段）**：从多个 RL 增强的领域专家教师模型蒸馏知识，补偿多模态训练中可能损失的文本能力：

> *"Post-training comprises three phases: (i) supervised fine-tuning on long chain-of-thought data, (ii) **knowledge distillation from stronger teacher models**, and (iii) reinforcement learning."*

引用文献 Video-OPD ([arXiv:2602.02994](https://arxiv.org/abs/2602.02994)) 对 oPD 机制的描述：

> *"Recent large-scale models, including MiMo-V2-Flash, Qwen3, and **Qwen3-VL**, adopt **on-policy distillation** to align a student model with multiple RL-enhanced, domain-specialized teachers through **token-level reverse KL divergence** computed over trajectories sampled from the student's current policy. This design enables **efficient transfer of multi-domain expertise while mitigating off-policy distributional mismatch**."*

最终 Qwen3-VL 也实现了"VLM 文本能力不弱于纯文本模型"的目标：

> *"By optimizing the training corpus and training strategy, we **preserve the underlying LLM's language proficiency during vision-language (VL) training**... Qwen3-VL **outperforms most VLMs across a broad set of multimodal tasks and surpasses its text-only counterpart on the majority of language benchmarks**."*

#### 策略对比

| 维度 | InternVL3 | Qwen3-VL |
|------|-----------|----------|
| **对问题的态度** | 多模态训练**可以增强**文本能力 | 多模态训练**会损害**文本能力，需要缓解 |
| **预训练策略** | Native Multimodal Pre-Training（ViT+MLP+LLM 全参数联合训练，混入 ~25% 纯文本数据） | 传统分阶段预训练，用 **square-root reweighting** 平衡多模态和文本 loss 贡献 |
| **后训练策略** | SFT + MPO（Mixed Preference Optimization） | SFT + **oPD**（on-policy distillation，从多个 RL-enhanced 领域专家蒸馏）+ RL |
| **最终结论** | VLM 文本能力 > 同基座纯文本模型 | VLM 文本能力 ≈ 或略 > 同基座纯文本模型 |

两者的最终结果都是"VLM 文本能力不弱于甚至强于纯文本模型"，但路径不同：InternVL3 靠**范式改变**（联合预训练，认为多模态数据与文本数据存在正向迁移），Qwen3-VL 靠**工程优化**（loss 设计 + 蒸馏，承认 trade-off 并用技术手段补偿）。

---

## 二、评估结果汇总

### 评估配置

- **任务集**: `med_eval`（17 个医学任务：8 多模态 VQA + 8 纯文本 QA + 2 报告生成，外加 open-ended 子集的 LLM Judge）
- **GPU**: 8 × A800/H800 80GB
- **max_pixels**: 3,211,264 (576 × 28 × 28)
- **LLM Judge**: `claude-sonnet-4-6`（Open-ended VQA 评估）

### Multimodal VQA

| Model | MMMU-Med | VQA-RAD | SLAKE | PathVQA | PMC-VQA | VQA-Med | OmniMedVQA | MedXpertQA | **Avg** |
|-------|---------|---------|-------|---------|---------|---------|------------|------------|---------|
| Qwen2.5-VL-3B | 52.67 | 59.20 | 51.58 | 32.23 | 53.70 | 14.25 | 63.51 | 20.45 | 43.45 |
| Qwen2.5-VL-7B | 54.67 | 58.98 | 53.77 | 35.19 | 52.60 | 15.92 | 65.81 | 22.45 | 44.92 |
| Qwen3-VL-2B | 40.00 | 52.99 | 48.85 | 34.52 | 49.90 | 14.11 | 69.26 | 20.45 | 41.26 |
| Qwen3-VL-4B | 53.33 | 52.33 | 55.21 | 37.29 | 53.55 | 16.79 | 76.95 | 22.70 | 46.02 |
| Qwen3-VL-8B | 58.67 | 56.10 | 57.88 | 36.18 | 55.75 | 18.60 | 75.64 | 24.90 | 47.96 |
| Qwen3-VL-30B-A3B | 66.67 | 64.52 | 58.60 | 38.74 | 55.45 | 20.41 | 79.49 | 30.00 | 51.73 |
| Qwen3-VL-32B | 67.33 | 61.86 | 60.27 | 39.15 | 60.50 | 21.64 | 77.44 | 30.60 | 52.35 |
| Qwen3.5-2B | 45.33 | 61.64 | 51.10 | 36.84 | 52.80 | 18.52 | 76.51 | 24.40 | 45.89 |
| Qwen3.5-4B | 48.67 | 65.41 | 56.69 | 41.19 | 55.25 | 20.91 | 76.58 | 33.30 | 49.75 |
| Qwen3.5-9B | 46.00 | 64.75 | 58.83 | 40.96 | 55.35 | 22.58 | **55.64**† | 35.95 | 47.51 |
| InternVL2.5-8B | 57.33 | 56.54 | 57.07 | 37.45 | 50.70 | 24.17 | 83.27 | 22.10 | 48.58 |
| InternVL3-8B | 60.67 | 60.98 | 66.28 | 40.30 | 53.35 | 19.97 | 77.11 | 22.80 | 50.18 |
| **InternVL3.5-8B** | **62.67** | **61.20** | **63.99** | **43.23** | **58.95** | **24.89** | **88.70** | **25.30** | **53.62** |

† Qwen3.5-9B 在 OmniMedVQA 上异常低，源于 instruction following 缺陷（详见发现 6）

### Text QA

| Model | MMLU-Med | PubMedQA | MedMCQA | MedQA | MedBullets-4 | MedBullets-5 | MedXpertQA | SuperGPQA | **Avg** |
|-------|---------|---------|---------|-------|-------------|-------------|------------|-----------|---------|
| Qwen2.5-VL-3B | 69.80 | 72.80 | 50.78 | 51.69 | 44.48 | 37.66 | 11.39 | 26.21 | 45.60 |
| Qwen2.5-VL-7B | 74.13 | 76.20 | 54.03 | 58.05 | 48.05 | 37.34 | 12.69 | 27.70 | 48.52 |
| Qwen3-VL-2B | 62.32 | 69.60 | 42.60 | 41.95 | 37.34 | 28.90 | 10.65 | 20.94 | 39.29 |
| Qwen3-VL-4B | 75.52 | 68.20 | 56.20 | 60.17 | 47.08 | 42.53 | 11.71 | 29.55 | 48.87 |
| Qwen3-VL-8B | 79.32 | 69.80 | 59.34 | 65.59 | 53.90 | 48.05 | 14.16 | 37.17 | 53.42 |
| Qwen3-VL-30B-A3B | 83.81 | 78.00 | 67.56 | 73.68 | 60.71 | 50.65 | 18.90 | 45.05 | 59.79 |
| **Qwen3-VL-32B** | **86.42** | 73.20 | **69.28** | **76.90** | **63.96** | **55.19** | 17.96 | **48.53** | **61.43** |
| Qwen3.5-2B | 62.27 | 73.20 | 45.71 | 48.08 | 44.81 | 41.88 | 10.45 | 18.98 | 43.17 |
| Qwen3.5-4B | 77.34 | 76.60 | 57.78 | 60.17 | 53.90 | 44.16 | 13.71 | 30.24 | 51.74 |
| Qwen3.5-9B | 75.73 | 76.40 | 61.10 | **24.43**† | **25.97**† | **19.48**† | 18.86 | 34.30 | 42.04 |
| InternVL2.5-8B | 76.38 | **77.20** | 53.45 | 55.22 | 47.40 | 41.56 | 12.12 | 26.24 | 48.70 |
| InternVL3-8B | 77.98 | 75.40 | 59.48 | 63.47 | 51.62 | 45.13 | 12.94 | 35.06 | 52.64 |
| InternVL3.5-8B | 79.42 | 72.00 | 58.69 | 66.06 | 53.90 | 47.40 | 14.12 | 32.16 | 52.97 |

† Qwen3.5-9B 在选择题任务上严重退化，接近随机水平（详见发现 6）

### Open-ended VQA (LLM Judge)

| Model | VQA-RAD Open | SLAKE Open | PathVQA Open | VQA-Med | **Avg** |
|-------|-------------|------------|-------------|---------|---------|
| Qwen3.5-2B | 36.9 | 52.9 | 8.3 | 18.5 | 29.2 |
| Qwen3.5-4B | 46.9 | 57.6 | 11.4 | 20.9 | 34.2 |
| Qwen3.5-9B | 42.5 | 60.3 | 11.6 | 22.6 | 34.3 |

### Report Generation

| Model | IU-XRay BLEU4 | IU-XRay RaTEScore | MIMIC-CXR BLEU4 | MIMIC-CXR RaTEScore | **Avg-BLEU4** | **Avg-RaTEScore** |
|-------|-------------|------------------|----------------|---------------------|-------------|-------------------|
| Qwen2.5-VL-3B | 5.17 | 46.29 | 3.41 | 45.56 | 4.29 | 45.92 |
| Qwen2.5-VL-7B | 3.67 | 48.31 | 2.86 | 45.97 | 3.27 | 47.14 |
| Qwen3-VL-2B | 3.94 | 59.66 | 2.53 | 50.26 | 3.23 | 54.96 |
| Qwen3-VL-4B | 2.24 | 53.42 | 2.02 | 50.66 | 2.13 | 52.04 |
| Qwen3-VL-8B | 3.31 | 52.07 | 3.43 | 49.58 | 3.37 | 50.82 |
| Qwen3-VL-30B-A3B | 3.21 | 54.88 | 3.62 | 50.69 | 3.41 | 52.78 |
| Qwen3-VL-32B | 2.58 | 53.79 | 2.39 | 50.64 | 2.48 | 52.22 |
| **Qwen3.5-2B** | **11.84** | **64.50** | 8.39 | 57.34 | **10.11** | 60.92 |
| **Qwen3.5-4B** | 11.81 | **64.80** | **8.71** | 57.36 | **10.26** | **61.08** |
| **Qwen3.5-9B** | 9.21 | 64.15 | 8.13 | **58.11** | 8.67 | **61.13** |
| InternVL2.5-8B | 3.17 | 51.74 | 3.35 | 47.68 | 3.26 | 49.71 |
| InternVL3-8B | 2.79 | 50.91 | 2.91 | 49.01 | 2.85 | 49.96 |
| InternVL3.5-8B | 5.52 | 54.56 | 8.15 | 56.58 | 6.83 | 55.57 |

---

## 三、关键发现

### 发现 1：Qwen3.5 在报告生成任务上大幅领先所有模型

| 模型 | Avg-BLEU4 | Avg-RaTEScore |
|------|----------|---------------|
| Qwen3-VL-8B | 3.37 | 50.82 |
| InternVL3.5-8B | 6.83 | 55.57 |
| **Qwen3.5-4B** | **10.26** | **61.08** |
| **Qwen3.5-2B** | **10.11** | **60.92** |

Qwen3.5-4B 的 BLEU-4 是 Qwen3-VL-8B 的 **3 倍**，RaTEScore 高出 10 个绝对百分点。即便与同属原生多模态预训练的 InternVL3.5-8B 相比，Qwen3.5-4B 也有明显优势（BLEU4: 10.26 vs 6.83）。

**分析**：报告生成需要模型将视觉信息（胸片）转化为结构化文本描述，深度依赖视觉-语言对齐质量。Qwen3.5 的 early fusion 架构让视觉和语言表征从预训练开始就相互适配，生成的报告在词汇选择和句式上更接近参考标准。这也与 Qwen3.5 在 OCR 和文档理解上的官方 benchmark 优势一致（OmniDocBench v1.5: 90.8，超过 GPT-5.2 和 Gemini 3 Pro）。

### 发现 2：同参数量下，Qwen3.5 在选择题上并不一定优于 Qwen3-VL

| 模型 | Multimodal Avg | Text QA Avg |
|------|---------------|-------------|
| Qwen3-VL-2B | 41.26 | 39.29 |
| **Qwen3.5-2B** | **45.89** | **43.17** |
| Qwen3-VL-4B | 46.02 | 48.87 |
| **Qwen3.5-4B** | **49.75** | **51.74** |
| Qwen3-VL-8B | **47.96** | **53.42** |
| Qwen3.5-9B | 47.51 | 42.04 |

2B 和 4B 尺寸上，Qwen3.5 略优于同尺寸 Qwen3-VL（4B multimodal: 49.75 vs 46.02）。但 9B 由于 instruction following 缺陷，选择题表现严重退化（Text QA: 42.04 vs Qwen3-VL-8B 的 53.42）。

**分析**：Qwen3.5 的训练目标偏向 agentic 场景（工具调用、UI 交互、长程推理），而非短答选择题。较大的 9B 模型在 non-thinking 模式下仍倾向输出详细推理过程，导致答案提取困难和 overthinking。这提示 **native multimodal 架构在通用能力上有优势，但在特定评测格式（选择题直接作答）上不一定更强**。

### 发现 3：Qwen3-VL 的 scaling 行为最一致

在所有 Qwen 系列中，Qwen3-VL 展现了最稳定的 scaling 行为：

| Qwen3-VL 模型 | 激活参数 | Multimodal Avg | Text QA Avg |
|--------------|---------|---------------|-------------|
| 2B | 2B | 41.26 | 39.29 |
| 4B | 4B | 46.02 | 48.87 |
| 8B | 8B | 47.96 | 53.42 |
| 30B-A3B (MoE) | 3B | 51.73 | 59.79 |
| 32B (Dense) | 32B | 52.35 | 61.43 |

从 2B 到 32B，每一档都有稳定提升，无异常跳变。相比之下，Qwen3.5 在 9B 处出现异常，Qwen2.5-VL 的 3B→7B 提升也较有限。

**分析**：Qwen3-VL 基于成熟的 Qwen3 LLM 基座 + 经过验证的三模块 VLM 范式，各尺寸的训练配方和后处理策略相对统一。Qwen3.5 作为新范式，小尺寸模型的训练可能还未完全成熟，尤其在 instruction following 方面。

### 发现 4：MoE 模型在医学场景下性价比突出

| 模型 | 激活参数 | Multimodal Avg | Text QA Avg |
|------|---------|---------------|-------------|
| Qwen3-VL-8B (Dense) | 8B | 47.96 | 53.42 |
| **Qwen3-VL-30B-A3B (MoE)** | **3B** | **51.73** | **59.79** |
| Qwen3-VL-32B (Dense) | 32B | 52.35 | 61.43 |

30B-A3B 仅激活 3B 参数（推理成本与 4B dense 模型相当），但性能接近 32B dense 模型（Multimodal: 51.73 vs 52.35），远超 8B dense 模型。Text QA 上差距更小：59.79 vs 61.43，仅差 1.6 个百分点。

**分析**：MoE 架构的专家路由机制可能使不同专家隐式专门化于不同医学子领域（影像、病理、药理等），从而在低计算开销下覆盖更广的知识面。这对医学评估这类**知识密集型多领域任务**尤其有利。

### 发现 5：InternVL3.5 在 8B 级别最强，后训练策略功不可没

| 8B 级模型 | Multimodal Avg | Text QA Avg | Report Avg-RaTEScore |
|----------|---------------|-------------|---------------------|
| Qwen2.5-VL-7B | 44.92 | 48.52 | 47.14 |
| Qwen3-VL-8B | 47.96 | 53.42 | 50.82 |
| Qwen3.5-9B | 47.51 | 42.04* | 61.13 |
| **InternVL3.5-8B** | **53.62** | **52.97** | 55.57 |

InternVL3.5-8B 在多模态 VQA 上以 53.62 领先（比 Qwen3-VL-8B 高 5.7，比 Qwen3.5-9B 高 6.1），同时在 17 个医学任务中的 15 个优于前代 InternVL3。

**分析**：InternVL3.5 的核心技术贡献在于 **Cascade RL**（离线 RL 稳定收敛 + 在线 RL 精调对齐）和 **Visual Consistency Learning (ViCO)**。InternVL3 本身就采用了 Native Multimodal Pre-Training（区别于 InternVL2.5 的"先训语言再接视觉"），InternVL3.5 在此基础上通过 RL 进一步强化推理能力。这提示 **后训练策略（RL）对医学推理任务的提升可能与架构创新同等重要**。

### 发现 6：Qwen3.5-9B 的 instruction following 缺陷导致选择题严重退化

**现象**：Qwen3.5-9B 在 MedQA (24.43%)、MedBullets-4 (25.97%)、MedBullets-5 (19.48%)、OmniMedVQA (55.64%) 上分数异常，部分接近随机水平。

**根因**：9B 在 non-thinking 模式下**自发生成 2000-4000 字符的推理文本**，虽然没有进入 thinking mode（无 `<think>` 标签），但输出冗长的推理过程。这导致：
1. 答案提取困难（最终答案淹没在长文本中）
2. Overthinking 导致改变正确的初始判断
3. MedQA 24.43% ≈ 随机 25%，说明推理过程本质上是噪声

**对比**：4B 的典型响应长度为 1 字符（直接输出选项字母），9B 为 2000-4000 字符。

**参考**：社区已广泛报告此问题（Reddit、GitHub issues），Qwen3.5 较大模型在 non-thinking 模式下仍倾向详细推理，这是模型训练特性而非配置问题。

### 发现 7：Open-ended 任务与选择题的能力存在解耦

以 Qwen3.5-9B 为例：

| 维度 | 表现 | 说明 |
|------|------|------|
| 选择题（MedQA/MedBullets） | 严重退化（24%/26%/19%） | 冗长输出 + overthinking |
| Open-ended llm_judge | 系列最高（SLAKE 60.3, VQA-Med 22.6） | 详细回答反而提供更多语义匹配 |
| 报告生成 METEOR | 系列最高（41.58） | 生成内容丰富度高 |

9B 的"冗长推理"特性在选择题中是缺陷，但在 open-ended 和生成任务中反而成为优势。这提示**不同评测指标衡量的是不同能力维度**，单一指标不足以反映模型的医学能力全貌。

### 发现 8：代际提升 > 参数量提升

跨代际对比中，新一代小模型往往接近甚至超过上一代大模型：

| 对比 | Multimodal Avg | Text QA Avg |
|------|---------------|-------------|
| Qwen2.5-VL-7B | 44.92 | 48.52 |
| Qwen3-VL-4B | 46.02 | 48.87 |
| Qwen3.5-4B | 49.75 | 51.74 |

Qwen3-VL-4B（4B 参数）已超过 Qwen2.5-VL-7B（7B 参数）；Qwen3.5-4B 进一步拉大差距。类似地：

| 对比 | Multimodal Avg | Text QA Avg |
|------|---------------|-------------|
| InternVL2.5-8B | 48.58 | 48.70 |
| InternVL3-8B | 50.18 | 52.64 |
| InternVL3.5-8B | 53.62 | 52.97 |

InternVL 在同一参数量下，每代提升约 2-5 个百分点。

**分析**：预训练数据质量和规模的提升（Qwen3 使用 ~36T tokens，远超 Qwen2.5）、训练范式革新（native multimodal pre-training）、后训练策略（RL alignment）是跨代提升的主要驱动力，其边际收益在医学任务上甚至超过参数量扩展。

---

## 四、综合结论

1. **架构范式与任务匹配**：Native multimodal（Qwen3.5）在**生成类任务**上有结构性优势（报告生成 BLEU-4 达到传统架构的 3 倍）；传统 ViT+LLM 拼接架构（Qwen3-VL）在**选择题任务**上更稳定可控。选择评估模型时应考虑目标任务类型。

2. **后训练比架构更关键**：InternVL3.5 用"传统"ViT-MLP-LLM 拼接架构 + Cascade RL 后训练，在 8B 级别打赢了 Qwen3.5 的 native multimodal 架构（Multimodal: 53.62 vs 47.51），说明 **RL 对齐策略在医学推理场景中的价值可能被低估**。

3. **MoE 是医学场景的高性价比选择**：Qwen3-VL-30B-A3B 以 3B 激活参数达到接近 32B dense 的效果（Multimodal: 51.73 vs 52.35），特别适合资源受限的医学部署场景。

4. **Instruction Following 是小模型的瓶颈**：Qwen3.5-9B 的异常表现警示——模型的"真实能力"和"在特定评测格式下的表现"之间存在显著 gap，评估框架需要综合多种指标形式（选择题 + open-ended + 生成）才能给出可靠结论。

5. **代际提升效率高于参数量扩展**：Qwen3-VL-4B ≈ Qwen2.5-VL-7B，Qwen3.5-4B > Qwen2.5-VL-7B。在医学场景下，优先升级模型版本比盲目增大参数量更有效。

6. **Qwen 系列的医学最优选择**：
   - **选择题/QA 任务最强**: Qwen3-VL-32B（Text QA Avg: 61.43）
   - **报告生成最强**: Qwen3.5-4B（Avg-BLEU4: 10.26, Avg-RaTEScore: 61.08）
   - **综合性价比**: Qwen3-VL-30B-A3B（MoE，3B 激活，全面均衡）
   - **8B 级别全能**: InternVL3.5-8B（Multimodal Avg: 53.62，8B 级最强）
