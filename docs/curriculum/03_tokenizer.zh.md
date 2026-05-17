# 3 · 分词器训练

## 分词器的作用

分词器将原始文本转换为神经网络可以处理的整数 ID 序列。分词方案的选择会影响：

- **词表效率** —— 表示平均文本需要多少个 Token
- **压缩比** —— 一个 Token 对应多少个字符（越高 → Token 数越少 → 训练越快）
- **未知字符处理** —— 词表外的字符能否被表示
- **多语言覆盖** —— 词表对非英语文本的覆盖程度

## 为什么要从零训练

使用现有分词器（例如 GPT-4 的 cl100k_base）会将我们绑定到别人的词表设计决策上。自行训练带来以下好处：

1. 词表针对我们的具体语料（FineWeb-Edu，以英语为主）进行了优化
2. 对特殊 Token 布局拥有完全控制权
3. 对每个组件都有完整的理解 —— 没有黑盒

## 算法：ByteLevel BPE

我们使用 **字节级 BPE**（在 UTF-8 字节而非 Unicode 字符上运行的字节对编码）：

- **字节级** —— 所有可能的字节值（0–255）均在基础词表中。这意味着任何 UTF-8 字符串都可以无需 `<unk>` Token 表示，无论使用什么语言或字符集。
- **BPE** —— 迭代合并出现频率最高的相邻 Token 对，从单字节逐步构建出常见子词和完整词汇。

这与 GPT-2、GPT-4 和 Llama 所使用的方案相同。

## 词表大小：32,768

| 词表大小 | 压缩比 | 权衡 |
|---|---|---|
| 6,400（minimind）| ~3–4 字符/Token | 速度快但对英语效果差；序列更长 |
| **32,768（本项目）** | **~4 字符/Token** | 对英语为主的语料平衡较好 |
| 100,000+（GPT-4）| ~4.5 字符/Token | 多语言覆盖更好；嵌入表更大 |

32,768 = 2^15，在 hidden_size=1792 的情况下嵌入表大小可控（约 6000万参数），同时对英语文本实现良好的压缩比。

## 特殊 Token

| Token | ID | 用途 |
|---|---|---|
| `<unk>` | 0 | 未知（ByteLevel BPE 不会用到，但规范要求） |
| `<pad>` | 1 | 填充（用于批量推理） |
| `<s>` | 2 | 序列开始（BOS） |
| `</s>` | 3 | 序列结束（EOS） |
| `<|im_start|>` | 4 | ChatML 对话轮次开始 |
| `<|im_end|>` | 5 | ChatML 对话轮次结束 |

## 对话模板：ChatML

分词器内置了一个 Jinja2 对话模板，用于将多轮对话格式化为指令微调所需的格式：

```
<|im_start|>user
法国的首都是哪里？<|im_end|>
<|im_start|>assistant
法国的首都是巴黎。<|im_end|>
<|im_start|>assistant
```

这是 **ChatML** 格式，被 Qwen、InternLM 等模型广泛采用。`<|im_start|>` / `<|im_end|>` Token 分隔每个对话轮次，当 `add_generation_prompt=True` 时，会在末尾追加助手前缀以引导模型生成。

模板存储在 `tokenizer_config.json` 中，由 HuggingFace 分词器基础设施自动应用 —— SFTTrainer 在格式化训练样本时会从此处读取。

## 训练

```bash
.venv/bin/python -m unbox_platform.tokenizer.train \
    --data data/fineweb_edu_10bt.jsonl \
    --output checkpoints/tokenizer
```

训练好的分词器以 HuggingFace 格式保存（`tokenizer.json` + `tokenizer_config.json`），可直接通过 `PreTrainedTokenizerFast.from_pretrained()` 加载。

## HuggingFace 兼容性

分词器将训练好的 BPE 模型封装为 `PreTrainedTokenizerFast`，这意味着：

- TRL 的 `SFTTrainer` 可以自动应用对话模板
- 分词器可通过 `save_pretrained` / `from_pretrained` 保存和加载
- 与 HuggingFace 生态系统中的评测工具无缝集成
