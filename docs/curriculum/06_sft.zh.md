# 6 · 监督微调（SFT）

## SFT 的作用

监督微调（SFT）教会基础模型遵循指令。模型在（提示词，回复）对组成的数据集上训练，损失仅在回复 Token 上计算 —— 模型因能根据提示词生成正确回复而得到奖励。

经过 SFT 后，原本生成 FineWeb-Edu 说明性文本的模型将转变为能够回答问题、遵循格式并保持主题聚焦的模型。

## SFT 的局限

**SFT 教的是格式，而非事实。** 如果基础模型在预训练中从未吸收某条知识，SFT 无法将其注入。一个不知道北京是中国首都的模型，经过 SFT 后，会对"中国的首都是哪里"给出流畅、切题、却错误的回答。

这正是预训练至关重要的核心原因 —— SFT 只是叠加在基础模型已有知识之上的薄薄一层。

## 框架：TRL

我们使用 [TRL](https://github.com/huggingface/trl) 的 `SFTTrainer` + `SFTConfig`。TRL 是离线监督微调的行业标准，提供：

- 梯度检查点
- 混合精度
- 评估循环
- W&B 集成
- 对话模板应用（从 `"messages"` 列自动处理）

替代方案 —— 自定义训练循环 —— 会重复 TRL 的基础设施，毫无收益。

## 数据集：UltraChat 200k

[UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) 是一个包含约 20 万条多轮对话的数据集，专为指令微调整理。每个样本有一个 `"messages"` 列，包含 `{"role": ..., "content": ...}` 轮次列表。

TRL 检测到 `"messages"` 列时，会自动应用我们的 ChatML 对话模板：

```
<|im_start|>user
法国的首都是哪里？<|im_end|>
<|im_start|>assistant
法国的首都是巴黎。<|im_end|>
```

!!! note "数据集列过滤"
    UltraChat 200k 有三个列：`"prompt"`、`"prompt_id"` 和 `"messages"`。TRL 1.4 检测到 `"prompt"` 列时，会尝试提示词+补全路径而非对话模板路径 —— 导致 `KeyError: 'completion'`。修复方法：在传入 Trainer 之前，仅保留 `["messages"]` 列。

## 训练配置

| 参数 | 值 |
|---|---|
| 基础检查点 | `checkpoints/pretrain/760m/latest.pt` |
| 数据集 | UltraChat 200k（`train_sft` / `test_sft`） |
| 轮数 | 1 |
| 批大小 | 2 |
| 梯度累积 | 4（等效批大小 = 8） |
| 最大学习率 | 2e-5（比预训练低 10 倍） |
| 学习率调度 | 余弦衰减 |
| 预热步数 | 100 |
| 最大序列长度 | 2048 |
| 数据类型 | bfloat16 |

学习率远低于预训练（2e-5 对比 3e-4）—— 我们希望调整模型行为，而非覆盖其预训练表示。

## 运行 SFT

```bash
.venv/bin/python -m unbox_platform.sft.train --config configs/sft/basic.yaml
```

## 实验依据

我们在预训练第 15,000 步（约 20% Token 预算）时启动了 SFT —— 并非因为基础模型已充分训练，而是为了回答一个问题：*需要多少预训练才能让 SFT 产出有效结果？*

若在第 15,000 步的 SFT 已能产出可用的指令跟随能力，可提前停止预训练。否则，从第 15,000 步的检查点继续预训练，择机再试。

实验成本仅为数小时。收益是有可能节省数天的预训练算力。

## W&B 追踪

SFT 指标记录到 `unbox-ai-sft` W&B 项目，与预训练项目分开。关键指标：`train/loss`、`train/mean_token_accuracy`、`eval/loss`。
