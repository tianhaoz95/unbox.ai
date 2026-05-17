# 2 · 数据清洗与预处理

## 为什么清洗至关重要

原始网页文本噪声极大。典型的 Common Crawl 快照包含：

- 重复页面（完全重复和近似重复）
- 模板内容：导航菜单、Cookie 提示、页脚文字
- 低质量内容：SEO 垃圾、自动生成文本、产品列表爬取内容
- 编码错误和乱码
- 个人身份信息（PII）

在未清洗的数据上训练，会让模型学会重现这些模式。质量过滤是最高效的干预手段之一 —— FineWeb-Edu 论文表明，使用教育质量分类器过滤 Common Crawl 后，模型在知识和推理基准测试上的得分显著提高，且所需训练 Token 更少。

## FineWeb-Edu 已为我们做了什么

通过选择 FineWeb-Edu，我们继承了一套多阶段清洗流水线：

1. **URL 过滤** —— 移除已知的垃圾内容和成人内容域名
2. **语言识别** —— 保留英语页面（使用 fastText）
3. **质量启发式规则** —— 移除特殊字符过多、行长度过短或内容重复的页面
4. **去重** —— 在段落和文档级别进行 MinHash 去重
5. **教育质量评分** —— 使用 Llama-3-8B 分类器对每个页面进行 1–5 分的教育价值评分，仅保留 ≥3 分的页面

这意味着我们的数据流水线从已清洗的数据出发 —— 在本次训练中无需自行实现去重或质量过滤。

## 我们的流水线做了什么

我们的数据预处理层（`unbox_platform/data/`）负责：

### 格式规范化

原始数据集每条文档提供一个 `"text"` 字段。流水线会：

1. 去除首尾空白字符
2. 在 Token ID 序列前添加 `[BOS]`，末尾追加 `[EOS]`
3. 将 Token 打包为固定长度 `max_seq_len=2048` 的块 —— 无填充浪费

### 流式加载 vs. 预先加载

原始的 `PretrainDataset` 会预先加载所有文档 —— 在训练开始前对 960 万条文档进行分词。对于 30GB 的语料，这既耗时又占内存。

`StreamingPretrainDataset` 通过惰性迭代解决了这个问题：

```python
def __iter__(self):
    buffer = []
    for text in self._iter_texts():           # 每次只读取一条文档
        ids = [bos] + tokenizer.encode(text) + [eos]
        buffer.extend(ids)
        while len(buffer) >= max_seq_len:
            yield chunk(buffer[:max_seq_len]) # 输出打包好的块
            buffer = buffer[max_seq_len:]
```

无论语料库大小如何，内存占用保持恒定。

### 恢复正确性

当训练从步骤 `N` 的检查点恢复时，数据集会跳过前 `N × batch_size × grad_accumulation_steps` 个块 —— 即已消费的确切数量。这确保模型在同一个 epoch 内不会重复训练相同的数据。

```python
skip_chunks = start_step * grad_accumulation_steps * batch_size
train_loader.dataset.skip_chunks = skip_chunks
```

## 未来工作：自定义清洗

对于在原始 Common Crawl（而非 FineWeb-Edu）上训练的场景，`unbox_platform/data/` 将需要：

- **去重** —— 基于哈希的精确去重 + MinHash LSH 近似去重
- **质量过滤** —— 基于困惑度的过滤（在小型参考语言模型下困惑度高的文本通常质量较低），或训练专用分类器
- **语言过滤** —— 使用 fastText 进行语言识别
- **PII 去除** —— 基于正则表达式清除邮箱地址、电话号码等个人信息

这些功能尚未实现 —— 当流水线扩展到原始网页数据时将会添加。
