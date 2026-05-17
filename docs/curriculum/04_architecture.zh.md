# 4 · 模型架构

## 概览

本模型是一个 **7.6亿参数的仅解码器 Transformer** —— 与 GPT、Llama 和 Qwen 同属一个家族。每个组件都是现代 LLM 文献中的标准构建块，选型标准是可读性和正确性，而非追求新颖性。

```
输入 Token ID
      ↓
Token 嵌入层  (vocab_size × hidden_size)
      ↓
24 × TransformerBlock
      ├── RMSNorm
      ├── 分组查询注意力（GQA）+ RoPE
      └── SwiGLU 前馈网络
      ↓
RMSNorm
      ↓
语言模型头  (hidden_size × vocab_size，与嵌入层共享权重)
      ↓
Logits → 损失（交叉熵）或下一个 Token
```

## 配置参数

| 参数 | 值 | 说明 |
|---|---|---|
| `hidden_size` | 1792 | 嵌入和残差流维度 |
| `num_layers` | 24 | Transformer 块数量 |
| `num_heads` | 16 | Query 头数 |
| `num_kv_heads` | 8 | Key/Value 头数（GQA 2:1 比例） |
| `ffn_intermediate_size` | 4864 | SwiGLU 隐藏层维度 |
| `vocab_size` | 32,768 | |
| `max_seq_len` | 2048 | |
| `rope_theta` | 500,000 | 长上下文 RoPE 基础频率 |
| **总参数量** | **~760M** | |

## 各组件详解

### RMSNorm

在每个注意力和前馈子层之前进行的前置归一化。比 LayerNorm 更简洁 —— 无均值减法，仅做 RMS 缩放：

```python
def forward(self, x):
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm * self.weight
```

前置归一化（在子层之前而非之后应用）在大规模训练中更稳定，是所有现代 LLM 的标准做法。

### 分组查询注意力（GQA）

标准多头注意力为每个 Query 头配备一个独立的 K/V 头 —— 在推理时内存开销较大。GQA 让多个 Query 头共享更少的 K/V 头：

- 16 个 Query 头，8 个 KV 头 → 2:1 比例
- KV 缓存比完整 MHA 小 2 倍
- 在当前规模下，注意力质量与 MHA 几乎无差异

K/V 头在注意力点积计算之前通过 `repeat_interleave` 扩展到与 Query 头数匹配。

### 旋转位置编码（RoPE）

RoPE 在注意力点积之前，通过在复数平面中旋转 Q 和 K 向量来编码位置信息。与可学习位置嵌入不同，RoPE：

- 可泛化到训练时未见过的序列长度
- 直接应用于 Q/K，而非加到嵌入向量上
- 保留相对位置信息

我们使用 `rope_theta=500,000`（原始值为 10,000）—— 更高的基础频率可扩展有效上下文长度，沿用 Llama 3 的缩放思路。

!!! warning "`freqs_cis` 缓冲区注意事项"
    RoPE 频率预先计算为 complex64 缓冲区（`freqs_cis`）。该缓冲区无法在强制转换为 bfloat16 时保持正确 —— PyTorch 会静默丢弃虚部，破坏所有位置信息。为此，`Transformer.to()` 被重写为：在类型转换之前先移除该缓冲区，转换完成后在正确设备上重新计算。

### SwiGLU 前馈网络

前馈层使用来自 Llama 家族的 SwiGLU 激活函数：

```python
def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

SwiGLU 包含两个线性投影（gate 和 up），通过 SiLU 激活函数进行门控，再经过 down 投影。其效果优于标准 ReLU 前馈网络，是 Llama、Qwen 和 Mistral 的默认选择。

### 权重绑定

输入嵌入矩阵与输出语言模型头的权重矩阵共享（`lm_head.weight = embed_tokens.weight`）。在几乎不影响质量的前提下节省约 6000万参数。

## HuggingFace 适配器

核心 `Transformer` 类对 HuggingFace 一无所知。一个独立的适配器（`unbox_platform/model/hf_adapter.py`）将其封装为 `PreTrainedModel` / `PretrainedConfig` 子类：

```python
class UnboxForCausalLM(PreTrainedModel):
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits, loss = self.model(input_ids, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
```

这使得 TRL、PEFT 和标准评测工具可以直接使用该模型，无需修改任何核心模型代码。适配器负责处理：

- `save_pretrained` / `from_pretrained` —— 标准 HF 检查点格式
- 梯度检查点 —— TRL 的 `SFTConfig` 默认启用
- 权重绑定 —— 通过 `_tied_weights_keys`
- 检查点互操作 —— `from_unbox_checkpoint()` 加载原始 `.pt` 训练检查点

## 梯度检查点

`TransformerBlock` 通过 HF 适配器设置的标志支持梯度检查点：

```python
def forward(self, x, freqs_cis, mask=None):
    if self.gradient_checkpointing and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward, x, freqs_cis, mask, use_reentrant=False
        )
    return self._forward(x, freqs_cis, mask)
```

这以重新计算换取内存：在反向传播期间重新计算激活值，而非存储。TRL 的 `SFTConfig` 默认启用此功能。
