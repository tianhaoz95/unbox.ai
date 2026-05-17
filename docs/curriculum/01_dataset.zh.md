# 1 · 数据集选择

## 核心问题

语言模型本质上是其训练数据的压缩表示。数据集的选择是整个流水线中影响最深远的决策 —— 它决定了模型能够掌握哪些知识、如何推理，以及会携带怎样的偏差。

## 优质预训练语料的标准

预训练语料需要在四个相互竞争的属性之间取得平衡：

| 属性 | 重要性 | 常见失效模式 |
|---|---|---|
| **规模** | Token 数量越多 → 泛化能力越强（直到 Chinchilla 最优点） | 过小 → 欠拟合；过大 → 边际收益递减 |
| **质量** | 教育性强、写作规范的文本 → 更好的推理和语言结构 | 网络爬取内容包含大量垃圾信息、SEO 内容和重复数据 |
| **多样性** | 涵盖广泛主题 → 更强的下游迁移能力 | 特定领域语料产生特定领域模型 |
| **许可证** | 研究和部署的法律合规性 | 许多大型语料库的授权条款不明确或过于限制 |

## 我们的选择：FineWeb-Edu sample-10BT

我们使用来自 Hugging Face 的 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) `sample-10BT` —— 经过教育质量过滤的 100亿 Token 子集。

**选择 FineWeb-Edu 的理由：**

- **质量过滤。** FineWeb-Edu 使用一个分类器对网页的教育价值进行评分（1–5分），仅保留高分页面。与原始 Common Crawl 相比，垃圾信息、SEO 内容和低质量文本大幅减少。
- **符合我们的算力预算。** 在我们的硬件上约 33,000 token/秒，100亿 Token 需要约 84 小时 —— 与第一次训练可用的算力窗口完全匹配。
- **Chinchilla 对齐。** 对于 7.6亿参数的模型，Chinchilla 最优训练量约为 152亿 Token。100亿 Token 达到最优的约 67% —— 足以验证流水线并产出有效的基础模型。
- **开放许可。** FineWeb-Edu 采用 ODC-By 协议发布，研究用途条款明确。
- **以英语为主。** 语料以英语为主，与我们的分词器和评估设置相匹配。

## Token 预算计算

```
硬件：       NVIDIA GB10（DGX Spark）
吞吐量：     ~4,250 token/秒（实测）
训练时长：   100亿 Token 约需 84 小时

7.6亿参数 × 20 token/参数（Chinchilla）= 152亿最优 Token
100亿 Token = Chinchilla 最优的 66% → 足以验证流水线
```

## 未采用的数据集

| 数据集 | 未采用原因 |
|---|---|
| The Pile | 较旧；质量过滤较弱；部分子集许可证不明确 |
| RedPajama | 质量较好，但 1.2T Token 远超我们的预算 |
| C4 | 质量尚可，但仅限英语，且去重程度过高导致多样性下降 |
| 原始 Common Crawl | 需要大量清洗工作；300B+ Token，体量过大 |

## 下载数据

```bash
.venv/bin/python -m unbox_platform.data.prepare \
    --output data/fineweb_edu_10bt.jsonl
```

下载过程从 Hugging Face Hub 流式获取分片，并显示进度条。输出为 JSONL 文件，每行包含一个 `"text"` 字段 —— 这是预训练数据流水线所期望的格式。

## 数据流水线

下载完成后，数据经过以下处理流程：

```
JSONL 文件
    ↓ StreamingPretrainDataset（惰性迭代，无预先分词）
    ↓ 实时分词：[BOS] + Token 序列 + [EOS]
    ↓ 打包为 max_seq_len=2048 的固定长度块
    ↓ DataLoader → 训练循环
```

惰性流式处理避免了将 30GB 数据加载到内存中。`skip_chunks` 机制允许从检查点恢复训练时，跳过已处理的数据，避免重复训练。
