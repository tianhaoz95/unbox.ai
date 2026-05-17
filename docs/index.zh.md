# unbox.ai

**从零构建大语言模型的全栈研究与教学平台。**

---

unbox.ai 是一个开源项目，完整实现了现代大语言模型的每一个训练阶段：数据集整理、分词器训练、预训练、监督微调、基于人类反馈的强化学习，以及生产级推理引擎。每一个组件都以可读性优先，而非仅仅追求运行效率。

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **数据集**

    ---

    FineWeb-Edu 10BT —— 100亿个高质量教育类网页文本 Token，经过过滤与预处理，专为语言模型训练而设计。

    [:octicons-arrow-right-24: 数据集选择](curriculum/01_dataset.md)

-   :material-alphabetical:{ .lg .middle } **分词器**

    ---

    从零训练的 32,768 词表 ByteLevel BPE 分词器，内置 ChatML 对话模板。

    [:octicons-arrow-right-24: 分词器训练](curriculum/03_tokenizer.md)

-   :material-brain:{ .lg .middle } **模型架构**

    ---

    7.6亿参数的仅解码器 Transformer，采用 GQA、SwiGLU 和 RoPE —— 与 Llama、Qwen 相同的基础构件。

    [:octicons-arrow-right-24: 模型架构](curriculum/04_architecture.md)

-   :material-chart-line:{ .lg .middle } **预训练**

    ---

    在约 100亿 Token 上训练，使用余弦学习率调度、bfloat16 混合精度、梯度累积和 W&B 可视化追踪。

    [:octicons-arrow-right-24: 预训练](curriculum/05_pretraining.md)

-   :material-tune:{ .lg .middle } **监督微调**

    ---

    基于 TRL 的 SFTTrainer，在 UltraChat 200k 数据集上进行监督微调，自动应用 ChatML 对话模板。

    [:octicons-arrow-right-24: 监督微调](curriculum/06_sft.md)

-   :material-robot:{ .lg .middle } **强化学习**

    ---

    离线 RL（DPO、GRPO）使用 TRL；在线 PPO 使用 OpenRLHF —— 完整还原生产环境的 RLHF 流程。

    [:octicons-arrow-right-24: 强化学习](curriculum/07_rl.md)

-   :material-server:{ .lg .middle } **推理引擎**

    ---

    完整的生产级推理引擎：分页 KV 缓存、连续批处理、预填充-解码分离、NCCL 张量并行、ZMQ 进程间通信以及 Triton 自定义算子。

    [:octicons-arrow-right-24: 推理引擎](curriculum/08_inference.md)

-   :material-flask:{ .lg .middle } **实验列表**

    ---

    核心平台验证完毕后，计划开展的研究实验清单。

    [:octicons-arrow-right-24: 计划实验](experiments.md)

</div>

---

## 设计理念

> **简洁优于极限性能。** 目标是达到业界吞吐量的约 50%，同时保持每个实现都足够简洁，研究者可以在一个下午内读懂并修改。

本项目并非要超越 vLLM 或 Megatron-LM。它的目标是让 LLM 技术栈的每一个环节都清晰可读 —— 研究者可以从头到尾读懂任何一个文件，理解其原理，替换其中某个组件，并立即开展新的实验。

每一个重要的架构决策都有文档记录。每次训练都会生成健全性检查报告。每个子系统都有测试覆盖。

## 当前状态

| 阶段 | 状态 |
|---|---|
| 分词器 | ✅ 已完成 |
| 预训练 | 🔄 进行中（约 20% Token）|
| 监督微调 | 🔄 运行中 |
| 强化学习 | 📋 计划中 |
| 推理引擎 | 📋 计划中 |
