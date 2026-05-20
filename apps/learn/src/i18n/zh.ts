// Chinese (Simplified) translations
import type { Translations } from "./en";

export const zh: Translations = {
  nav: {
    overview: `概览`,
    chapters: `章节`,
    github: `GitHub`,
    signIn: `登录`,
    signOut: `退出`,
    soon: `即将推出`,
  },
  theme: {
    toggle: `切换主题`,
    dark: `深色`,
    light: `浅色`,
  },
  lang: {
    en: `EN`,
    zh: `中`,
  },
  auth: {
    title: `登录以发表评论`,
    subtitle: `选中章节中的任意文字即可留下反馈。您的评论帮助我们改进内容。`,
    google: `使用 Google 继续`,
    disclaimer: `评论对所有人可见，请保持建设性。`,
  },
  comments: {
    button: `评论`,
    count: `{n} 条评论`,
    countMany: `{n} 条评论`,
    panelTitle: `评论`,
    empty: `暂无评论。选中任意文字开始讨论。`,
    loading: `加载评论中…`,
    placeholder: `对选中的文字发表评论…`,
    send: `发送`,
    cancel: `取消`,
    signIn: `登录以发表评论`,
    selectText: `选中文字以评论`,
    justNow: `刚刚`,
    minsAgo: `{n} 分钟前`,
    hoursAgo: `{n} 小时前`,
    daysAgo: `{n} 天前`,
  },
  chapter: {
    prev: `上一章`,
    next: `下一章`,
    comments: `评论`,
  },
  callout: {
    why: `为什么重要`,
    tip: `专业提示`,
    warning: `注意`,
    insight: `关键洞察`,
  },
  comingSoon: {
    back: `返回概览`,
    badge: `即将推出 — 开发中`,
    learn: `你将学到`,
    meanwhile: `与此同时，探索已有章节：`,
    github: `在 GitHub 上关注进度`,
  },

  // ── Home ──────────────────────────────────────────────────────────────────
  home: {
    badge: `从零开始构建 LLM — 循序渐进`,
    title: `训练一个 LLM`,
    gradient: `从零开始`,
    subtitle: `构建工业级语言模型基础设施的实践指南：数据管道、分词器、预训练、微调与推理引擎。每个组件都配有代码和动画讲解。`,
    start: `开始学习`,
    source: `查看源代码`,
    stats: {
      lines: `行注释代码`,
      anims: `交互式动画`,
      chapters: `章节`,
      scratch: `从零构建`,
    },
    path: `学习路径`,
    ready: `已完成`,
    soon: `即将推出`,
    philosophy: {
      title: `设计理念`,
    },
    simplicity: {
      title: `简洁胜过速度`,
      desc: `目标达到业界吞吐量的约 50%，同时保持每个组件可读、可修改。先理解，再优化。`,
    },
    realCode: {
      title: `真实代码，真实系统`,
      desc: `每个代码片段都来自 unbox_platform —— 一个完整的、功能齐全的 LLM 栈，规模堪比 Megatron-LM，专为学习而构建。`,
    },
    build: {
      title: `通过构建来理解`,
      desc: `不要包装 API。构建分词器。编写训练循环。运行注意力内核。理解来自实现。`,
    },
  },

  // ── Chapter metadata ──────────────────────────────────────────────────────
  ch01: {
    title: `数据管道`,
    desc: `从原始网络爬取到打包的 token 序列。数据清洗、去重、分词与高效批处理。`,
    t1: `FineWeb-Edu 数据集`, t2: `质量过滤`, t3: `数据去重`, t4: `序列打包`,
  },
  ch02: {
    title: `分词器`,
    desc: `将文本转换为模型可处理的整数。学习 BPE、SentencePiece，以及如何训练自己的分词器。`,
    t1: `BPE 算法`, t2: `SentencePiece`, t3: `词表大小权衡`, t4: `特殊 token`,
  },
  ch03: {
    title: `预训练`,
    desc: `从零训练 GPT 级别的模型。Transformer 架构、训练循环、混合精度与检查点。`,
    t1: `Transformer 架构`, t2: `下一个 token 预测`, t3: `AdamW + 学习率调度`, t4: `混合精度`,
  },
  ch04: {
    title: `评估`,
    desc: `在进行微调之前衡量模型质量。困惑度、定性采样与 DPO 胜率。`,
    t1: `在保留数据上的困惑度`, t2: `采样策略（top-k，核采样）`, t3: `对话格式生成`, t4: `DPO 胜率指标`,
  },
  ch05: {
    title: `SFT`,
    desc: `将基础模型转变为指令跟随者。HF 适配器、对话模板、仅补全损失、TRL SFTTrainer。`,
    t1: `UnboxForCausalLM HF 适配器`, t2: `对话模板 (ChatML)`, t3: `completion_only_loss`, t4: `SFTTrainer + SFTConfig`,
  },
  ch06: {
    title: `DPO`,
    desc: `使用偏好对使模型与人类偏好对齐，无需奖励模型。`,
    t1: `DPO vs RLHF`, t2: `隐式奖励公式`, t3: `β 作为 KL 约束`, t4: `ultrafeedback_binarized 数据集`,
  },
  ch07: {
    title: `推理`,
    desc: `大规模部署模型。持续批处理、分页 KV 缓存、解耦预填充-解码、Triton 内核。`,
    t1: `PagedAttention`, t2: `持续批处理`, t3: `解耦式服务`, t4: `Triton 内核`,
  },
  ch08: {
    title: `蒸馏`,
    desc: `通过 logit 匹配和隐藏状态对齐，将知识从大型教师模型迁移到小型学生模型。`,
    t1: `Logit 蒸馏`, t2: `隐藏状态蒸馏`, t3: `GKDTrainer`, t4: `推理迁移`,
  },

  // ── Data Pipeline ─────────────────────────────────────────────────────────
  data: {
    subtitle: `从原始网络爬取到打包好的、可供训练使用的 token 序列。这是一切赖以运转的枯燥基础。`,
    s1: {
      title: `数据从哪里来？`,
      p1: `预训练数据是影响模型质量的最大杠杆。你需要<strong>数万亿 token</strong> 的多样化、高质量文本。行业标准是对 <strong>Common Crawl</strong> 的过滤子集——一个自 2008 年起每月拍摄的 PB 级网络快照。`,
      p2: `本项目使用 <strong>FineWeb-Edu</strong>：一个 1.3 万亿 token 的数据集，通过基于人类评分训练的分类器过滤至教育内容。它在 Hugging Face 上免费提供，与原始爬取数据相比，每 token 能产出质量更高的模型。`,
      why: `<strong>为什么不用维基百科或书籍？</strong>它们虽然干净，但数量太少——维基百科约 40 亿 token，古腾堡计划约 30 亿。一个 7.6 亿参数的模型需要 150 亿+ token 才能收敛（Chinchilla 缩放定律）。你必须使用网络数据。`,
      tip: `下载大型数据集时使用 <code>streaming=True</code>。它让你在无需将完整的约 200GB 数据下载到磁盘的情况下立即开始处理。`,
    },
    s2: {
      title: `数据质量：过滤与去重`,
      p1: `原始网络数据噪声很大。页面中包含垃圾信息、模板内容、重复内容和非预期语言的文本。两项操作具有最高的投入产出比：`,
      filter: {
        title: `质量过滤`,
        i1: `语言检测 (fasttext)`,
        i2: `使用 KenLM 进行困惑度过滤`,
        i3: `去除 HTML 残留`,
        i4: `过滤短文本和模板文本`,
      },
      dedup: {
        title: `去重`,
        i1: `使用 MD5/SHA-256 精确去重`,
        i2: `使用 MinHash LSH 近似去重`,
        i3: `URL 级别去重`,
        i4: `段落级精确匹配`,
      },
      insight: `去重的影响力远超预期。GPT-3 的训练数据中约有 3% 是重复的，但删除它们对验证损失的改善超过了添加等量独特 token。模型会记住重复内容而不是真正学习。`,
    },
    s3: {
      title: `分词与序列打包`,
      p1: `过滤后，我们将原始文本转换为 token ID，并将其打包成固定长度的序列。这是一次性的离线步骤，产出训练期间加载的精确 numpy 数组。`,
      why: `<strong>为什么要打包而不是填充？</strong>填充会浪费计算资源在对梯度贡献为零的 token 上。使用填充时，token 利用率可能只有 60-70%。带文档边界的序列打包可实现<strong>约 100% 的利用率</strong>。在规模化训练中，这是 100 万美元训练费用与 60 万美元之间的差异。`,
      vis: {
        title: `序列打包可视化`,
        note: `三个文档均打包到一个 1024 token 的序列中，用 <code>&lt;eos&gt;</code> 分隔符间隔。token 利用率 100%。`,
        packed: `已打包`,
      },
    },
    s4: {
      title: `构建 DataLoader`,
      p1: `最后一步是将打包好的 numpy 数组加载到 PyTorch <code>DataLoader</code> 中。我们使用<strong>内存映射</strong>，让操作系统只分页加载所需的块——无需将完整数据集放入内存。`,
      tip: `<strong>num_workers=4</strong> 会启动 4 个后台进程，在 GPU 处理上一步时预取批次数据。没有它，CPU 数据加载会成为瓶颈。设置 <code>pin_memory=True</code> 可加速 CPU→GPU 的 DMA 传输。`,
      stat1: { k: `批次大小`, v: `32–512`, n: `每 GPU，根据内存调整` },
      stat2: { k: `序列长度`, v: `1024–8192`, n: `越长计算量越大` },
      stat3: { k: `Token 利用率`, v: `~100%`, n: `相比填充的 60-70%` },
    },
    s5: {
      title: `运行`,
      p1: `完整管道已封装为 CLI 命令。在训练前运行一次：`,
      insight: `你只需运行数据管道<strong>一次</strong>。生成的 <code>.npy</code> 文件就是所有实验的训练数据。如果更改分词器词表，需要重新分词。如果只是更改模型架构或训练超参数，则不需要。`,
    },
  },

  // ── Tokenizer ─────────────────────────────────────────────────────────────
  tok: {
    subtitle: `文本是字符串，模型需要整数。分词器是两者之间的桥梁——它的设计会影响所有下游决策。`,
    s1: {
      title: `什么是分词器？`,
      p1: `分词器将文本分割成 <strong>token</strong>——映射到固定词表中整数的离散单元。模型从不直接处理字符或字节；它完全在 token ID 上操作。`,
      p2: `词表从训练数据中学习得到。常见的子词有自己的 ID；罕见的词被分割成多个 token。"tokenization"可能变成 <code>["token", "ization"]</code>，"ChatGPT"可能是 <code>["Chat", "G", "PT"]</code>。`,
      table: {
        title: `业界词表大小`,
        model: `模型`, vocab: `词表大小`, notes: `备注`,
      },
      why: `<strong>为什么不直接按单词或字符分割？</strong>词级词表体积会爆炸（数百万个词，OOV 问题）。字符级词表很小，但序列会变得很长——比 BPE 长 4 倍，意味着 4 倍的注意力操作。BPE 找到了最佳平衡点。`,
    },
    s2: {
      title: `BPE 算法`,
      p1: `字节对编码（BPE）通过从单个字符开始，迭代地合并<strong>最频繁的相邻对</strong>来构建词表。经过 N 次合并后，你得到一个包含 N +（初始字母表大小）个 token 的词表。`,
      insight: `合并顺序很重要，给定语料库是确定性的。保存分词器时，你保存的是有序的合并列表。推理时，你贪婪地应用这些合并来编码新文本。这就是为什么分词器是<strong>特定于语料库的</strong>——在英语代码上训练的分词器对日语文本会很低效。`,
      bpe: {
        title: `BPE 优势`,
        i1: `字节级别语言无关`,
        i2: `优雅处理任何 Unicode`,
        i3: `词表可解释`,
        i4: `编码速度快（O(n log n)）`,
      },
      sp: {
        title: `SentencePiece 的不同之处`,
        i1: `直接处理原始文本（无需预分词）`,
        i2: `将空格作为显式 token 处理`,
        i3: `支持 Unigram 语言模型变体`,
        i4: `被 LLaMA、T5、ALBERT 使用`,
      },
    },
    s3: {
      title: `训练你的分词器`,
      p1: `我们使用 HuggingFace 的 <code>tokenizers</code> 库——它用 Rust 编写，可在 2 分钟内对 10GB 文本训练出 32k 词表的分词器。`,
      warning: `在分词数据（第 01 章）<strong>之前</strong>先训练分词器。分词器是数据管道的前提条件。实际上，你先在一个代表性样本（例如 10 亿 token）上训练分词器，然后用得到的词表对完整数据集进行分词。`,
    },
    s4: {
      title: `推理时使用分词器`,
      p1: `推理期间，分词器有额外的职责：增量流式解码。模型每次生成一个 token，但我们不能总是将单个 token 解码为字符——有些字符在多字节 UTF-8 编码中跨越多个 token。`,
      insight: `这就是为什么流式 LLM 输出有时在第一个字符出现前会有短暂延迟——服务器在缓冲 token，直到确认一个完整的 UTF-8 序列。对于中文、阿拉伯语、emoji 等特殊字符也会出现同样的问题。`,
    },
    s5: {
      title: `关键设计决策`,
      q1: { q: `词表大小：32k 还是 128k？`, a: `32k 是 LLaMA 2 的最佳点。128k（LLaMA 3）以更大的嵌入表换取多语言场景中更短的序列。对于纯英语研究，32k 更具参数效率。` },
      q2: { q: `添加领域特定 token？`, a: `如果你有分割效果差的领域特定字符串，则添加——例如"<|endoftext|>"、代码关键字或数学符号。在训练前将它们作为特殊 token 添加。` },
      q3: { q: `区分大小写？`, a: `保留大小写。小写化会丢失模型可以学习的信息。BPE 自然将大小写变体处理为不同的 token（"The"与"the"），模型会学习它们之间的关联。` },
      q4: { q: `复用预训练分词器？`, a: `研究用途完全没问题。使用 LLaMA 的分词器省去了训练步骤，并确保与预训练权重的兼容性，便于微调实验。` },
    },
  },

  // ── Pre-training ──────────────────────────────────────────────────────────
  pre: {
    subtitle: `将打包好的 token 序列输入 Transformer 并预测下一个 token。这是模型学习语言、知识和推理的地方。`,
    s1: {
      title: `目标：下一个 token 预测`,
      p1: `预训练在概念上很简单：给定一个 token 序列，预测下一个。对来自多样化文本的数万亿 token 执行此操作，模型就会学习语法、事实、推理模式等等。`,
      tf: {
        title: `教师强制`,
        input: `输入`, target: `目标`,
        note: `输入偏移 1 位。训练期间每个位置并行预测下一个 token。`,
      },
      insight: `一个训练样本同时产生 T 个预测任务。一个 1024 token 的序列在单次前向传播中给出 1023 个（输入, 目标）对。这就是 Transformer 训练比逐步处理的循环网络效率高得多的原因。`,
    },
    s2: {
      title: `模型架构：LLaMA 风格 Transformer`,
      p1: `模型使用<strong>仅解码器 Transformer</strong>，并在原始 2017 年架构基础上进行了现代化改进：RMSNorm 替代 LayerNorm、SwiGLU 激活函数、旋转位置嵌入（RoPE）和分组查询注意力（GQA）。`,
      r1: `更便宜，同等质量——去除均值减法步骤`,
      r2: `通过旋转实现相对位置——泛化到更长的上下文`,
      r3: `SiLU × 门控 = 更平滑的梯度，经验上损失更低`,
      r4: `更少的 KV 头 → 更小的 KV 缓存 → 内存中更多 token`,
      why: `<strong>为什么不用偏置项？</strong>现代 LLM 中的线性层通常没有偏置。偏置省约 0.1% 的参数，但会给梯度增加噪声且没有帮助。LLaMA、Mistral、Qwen 都使用 <code>bias=False</code>。`,
    },
    s3: {
      title: `训练循环`,
      p1: `训练循环是最内层的热路径。每一行对正确性和性能都很重要。以下是按顺序排列的六个关键操作：前向传播、损失计算、反向传播、梯度裁剪、优化器步进、学习率步进。`,
      bf16: `<strong>BF16 vs FP16：</strong>如果你的 GPU 支持（Ampere+），请使用 BF16（bfloat16）。BF16 与 FP32 具有相同的指数范围，不会溢出；FP16 需要损失缩放。BF16 严格来说是 LLM 训练的更佳选择。`,
      gradacc: `<strong>梯度累积：</strong>如果批次不适合 GPU 内存，在调用 <code>optimizer.step()</code> 之前对 N 个微批次累积梯度。每次反向传播前将损失除以 N。`,
    },
    s4: {
      title: `学习率调度`,
      p1: `学习率调度是影响最大的超参数之一。太高会导致训练发散。太低会浪费计算资源。LLM 的标准配方是：<strong>线性预热 → 余弦衰减 → 最小学习率</strong>。`,
      lr: {
        title: `学习率曲线`,
        maxN: `峰值学习率`, warmupN: `约训练的 0.1%`, minN: `max_lr / 10`,
      },
    },
    s5: {
      title: `检查点与可恢复性`,
      p1: `预训练需要几天或几周。你<em>一定会</em>遇到硬件故障、抢占和需要恢复的实验。保存所有需要的内容以便从中断处精确恢复。`,
      warning: `保存<strong>优化器状态</strong>，而不仅仅是模型权重。AdamW 的矩估计（m, v）积累了关于梯度历史的知识。仅恢复权重意味着优化器从冷启动——你会看到临时的损失峰值和 500-1000 步的浪费计算。`,
      stat1: { k: `保存频率`, v: `每 1k 步`, n: `平衡开销与恢复成本` },
      stat2: { k: `保留 N 个检查点`, v: `最近 3–5 个`, n: `删除旧的以节省磁盘空间` },
      stat3: { k: `上传至`, v: `ModelScope / S3`, n: `用于跨机器恢复` },
    },
    s6: {
      title: `运行预训练`,
      p1: `准备好数据（第 01 章）并训练好分词器（第 02 章）后，就可以启动预训练了：`,
      insight: `<strong>7.6 亿参数模型</strong>在 150 亿 token 上训练（该规模的 Chinchilla 最优点）在 8× A100-80GB GPU 上大约需要 3 天。损失应该从约 10（随机）降至约 2.3–2.5（有能力的英语文本生成）。`,
      chinchilla: {
        title: `Chinchilla 缩放定律`,
        desc: `Chinchilla 论文（Hoffmann 等，2022）表明最优的计算分配是在更多数据上训练更小的模型：大约每个参数<strong> 20 个 token</strong>。7.6 亿参数 → 150 亿 token 是 Chinchilla 最优。`,
        params: `模型参数`, tokens: `最优 token 数`, ratio: `比例`,
      },
    },
  },

  // ── Evaluation ────────────────────────────────────────────────────────────
  eval: {
    subtitle: `预训练后你有了一个模型——但它好用吗？评估告诉你在进行昂贵的微调之前的当前状态。`,
    s1: {
      title: `为什么在微调前评估？`,
      p1: `预训练很昂贵。在花费更多计算资源进行 SFT 或 DPO 之前，你需要了解两件事：模型是否具有连贯的语言理解能力（困惑度），以及它是否能产生合理的文本（定性采样）？这些检查只需几分钟，但可以避免你浪费数天时间去微调一个有缺陷的基础模型。`,
      tool1: { tool: `eval/perplexity.py`, when: `预训练后`, desc: `衡量模型对保留文本的"惊讶"程度。越低 = 语言模型越好。` },
      tool2: { tool: `eval/sample.py`, when: `预训练后`, desc: `从基础模型生成文本。在进行任何指令微调之前检查连贯性。` },
      tool3: { tool: `eval/chat_sample.py`, when: `SFT 后`, desc: `从通过 HF 适配器加载的 SFT 检查点进行对话格式生成。` },
      why: `评估不只是最后一步——它是一个<strong>反馈循环</strong>。每 N 千个训练步骤运行困惑度评估，以确认保留数据上的损失在下降而不是过拟合。训练损失持续下降而评估损失发散是最清晰的过拟合信号。`,
      runLabel: `运行：`,
    },
    s2: {
      title: `困惑度：标准语言模型指标`,
      p1: `困惑度是保留文本上的<strong>平均交叉熵损失的指数（e^loss）</strong>。直觉上：如果模型给出 20 的困惑度，它在每个 token 上的不确定性就像在 20 个选项中均匀选择一样。越低越好。在 32k 词表上的随机模型困惑度约为 32,000。训练良好的 7.6 亿模型在 FineWeb-Edu 评估文本上应达到约 15–20。`,
      formula: {
        title: `公式`,
        note: `N = 评估的 token 总数；指数是平均负对数似然（= 交叉熵损失）`,
      },
      warning: `<strong>不要直接对每批损失取平均值。</strong>每个批次有不同数量的非填充 token。你必须恢复 token 级别的总和（<code>loss × n_tokens</code>），全局累积，然后除以总 token 数。对均值取平均会产生有偏估计。`,
    },
    s3: {
      title: `采样策略：温度、top-k、top-p`,
      p1: `在每个生成步骤，模型输出整个词表上的 logit 向量。采样是如何将这些 logit 转换为单个 token 的过程。三个旋钮——温度、top-k 和 top-p——控制创造力与连贯性的权衡。`,
      sweet: `最佳值：`,
      p1m1: { name: `温度 τ`, formula: `softmax(logits / τ)`, low: `τ→0：贪婪，重复`, high: `τ→∞：均匀噪声`, sweet: `τ = 0.7–0.9` },
      p1m2: { name: `Top-k`, formula: `保留 k 个最高 logit`, low: `k=1：贪婪`, high: `k=词表大小：无限制`, sweet: `k = 40–100` },
      p1m3: { name: `Top-p（核采样）`, formula: `保留最小集合 ≥ p`, low: `p→0：贪婪`, high: `p=1.0：无限制`, sweet: `p = 0.9–0.95` },
    },
    s4: {
      title: `从 SFT 检查点进行对话采样`,
      p1: `SFT（第 05 章）后，检查点通过 <code>UnboxForCausalLM</code> 以 HuggingFace 格式保存。评估方式有所不同：使用 <code>from_pretrained</code> 加载，应用对话模板，然后调用 HF 的 <code>model.generate()</code>——而不是原始 Transformer 的。`,
      p2: `两个不明显的陷阱会在第一次让大多数人踩坑：`,
      pitfall1: `对对话模板输出进行分词时，始终传入 <code>add_special_tokens=False</code>。模板已经包含所有特殊 token。不传入这个参数，会附加一个虚假的 <code>&lt;|eos|&gt;</code>，导致模型生成假的下一个用户轮次而不是你的回答。`,
      pitfall2: `<code>UnboxForCausalLM</code> 尚未实现 KV 缓存。如果调用 <code>generate(use_cache=True)</code>（HF 默认值），HF 在第 2 步以后只传入最后一个 token，破坏所有上下文并产生乱码。始终传入 <code>use_cache=False</code>。`,
    },
    s5: {
      title: `DPO 评估：胜率和对数概率差`,
      p1: `在运行 DPO（第 06 章）之前，先通过衡量 SFT 模型已经多频繁地偏好"chosen"回复而非"rejected"回复来建立基线。一个校准良好的 SFT 模型应该已经赢得约 55–65% 的对；DPO 应该将其推至约 70–80%。`,
      metricsTitle: `两个指标`,
      m1: { m: `胜率`, formula: `log P(chosen) > log P(rejected) 的比例`, baseline: `约 50% = 随机，约 58–65% = SFT，约 73%+ = 良好的 DPO` },
      m2: { m: `差值`, formula: `平均值( log P(chosen) − log P(rejected) ) 每 token`, baseline: `正值 = 模型偏好 chosen；越大 = 偏好越强` },
      insight: `比较回复时使用<strong>每 token</strong> 的对数概率（均值损失，而非总和）。如果使用总对数概率，一个长而正确的回答总是会击败一个短的——长度偏差会主导质量信号。`,
    },
    s6: {
      title: `运行`,
      tip: `在困惑度评估中添加 <code>--max-batches 200</code> 进行快速冒烟测试（CPU 上约 2 分钟）。移除它以进行完整评估（GPU 上约 20 分钟）。实际上，200 批次的估计与完整评估的误差在 0.3 PPL 以内。`,
    },
  },

  // ── SFT ───────────────────────────────────────────────────────────────────
  sft: {
    subtitle: `预训练的基础模型能预测文本。SFT 将其转变为能够遵循指令的助手。关键在于数据集格式和损失计算方式。`,
    s1: {
      title: `HuggingFace 适配器：为什么需要它`,
      p1: `我们预训练的 <code>Transformer</code> 是一个干净的 PyTorch 模块——但 TRL 的 <code>SFTTrainer</code> 需要 HuggingFace 的 <code>PreTrainedModel</code>。<code>UnboxForCausalLM</code> 适配器弥合了这一差距：它封装了现有模型，不做任何架构改动，只添加 HF 接口层。`,
      p2: `这意味着我们免费获得了 TRL、PEFT/LoRA 和 HF 生态系统，而无需触及核心模型代码——这正是架构设计所追求的关注点分离。`,
      insight: `<code>UnboxForCausalLM</code> 同时继承自 <code>PreTrainedModel</code> 和 <code>GenerationMixin</code>。<code>GenerationMixin</code> 免费提供完整的 <code>model.generate()</code> 循环——束搜索、采样、停止准则。我们只需要一个正确的 <code>forward()</code>。`,
      warning: `适配器接受 <code>attention_mask</code> 和 <code>use_cache</code> 以兼容 HF API，但两者均未实现。调用 <code>generate()</code> 时始终传入 <code>use_cache=False</code>——具体陷阱见评估章节（第 04 章）。`,
    },
    s2: {
      title: `对话模板：消息到 token 的映射`,
      p1: `SFT 训练数据是 <code>{"role": ..., "content": ...}</code> 消息字典的列表。<strong>对话模板</strong>将此结构转换为分词器可以处理的平面字符串。目前最常见的格式是 ChatML，被 Qwen、Mistral-Instruct 等众多模型使用。`,
      chatml: {
        title: `ChatML 格式`,
        masked: `已遮罩（labels = -100）`,
        loss: `计算损失`,
      },
      why: `<strong>为什么遮罩用户轮次？</strong>不使用 <code>completion_only_loss=True</code>，模型也会学习预测用户的消息。推理时它会开始生成<em>"User: ..."</em>的延续，而不是保持助手角色。遮罩迫使所有梯度信号仅通过助手的回复。`,
    },
    s3: {
      title: `SFTTrainConfig：超参数`,
      p1: `SFT 使用比预训练低得多的学习率——通常为 <strong>1e-5 到 5e-5</strong>，而预训练为 3e-4。模型已经初始化良好；我们是在引导它走向指令跟随，而不是从头学习语言。`,
      p1m1: { k: `max_lr`, n: `比预训练低 10 倍——保留基础知识` },
      p1m2: { k: `num_epochs`, n: `更多 epoch → 在小数据集上过拟合` },
      p1m3: { k: `packing`, n: `True = 更高效但会丢失对话边界` },
      p1m4: { k: `max_seq_len`, n: `限制以避免多轮对话中的 OOM` },
    },
    s4: {
      title: `使用 SFTTrainer 训练`,
      p1: `TRL 的 <code>SFTTrainer</code> 处理完整的训练循环：对话模板应用、分词、遮罩、梯度累积、评估和检查点保存。数据集必须有一个 <code>"messages"</code> 列——如果还有 <code>"prompt"</code> 列，TRL 会走另一条（更差的）代码路径，所以我们删除除 <code>"messages"</code> 以外的所有内容。`,
      tip: `SFT 训练通常只需要<strong>1 个 epoch</strong>。更多 epoch 在指令跟随上收益递减，并增加过拟合风险（模型开始记忆特定回复）。如果你的样本少于 5 万，考虑 2–3 个 epoch 并配合早停。`,
    },
    s5: {
      title: `运行`,
      stat1: { l: `数据集大小`, v: `20 万轮`, n: `ultrachat_200k` },
      stat2: { l: `训练时间`, v: `约 4 小时`, n: `在 2× A100-80GB 上` },
      stat3: { l: `预期 SFT 损失`, v: `约 1.4–1.6`, n: `1 个 epoch 后` },
      insight: `SFT 检查点以 HuggingFace 格式保存（而不是 <code>.pt</code>）。这意味着 <code>AutoTokenizer.from_pretrained()</code> 和 <code>UnboxForCausalLM.from_pretrained()</code> 都可以直接在输出目录上使用。这是你接下来要传递给 DPO 的检查点。`,
    },
  },

  // ── DPO ───────────────────────────────────────────────────────────────────
  dpo: {
    subtitle: `直接偏好优化使用偏好对使模型与人类偏好对齐——无需奖励模型。`,
    s1: {
      title: `为什么不用 RLHF？`,
      p1: `经典 RLHF（来自人类反馈的强化学习）需要三个独立阶段：(1) 从偏好对训练奖励模型，(2) 运行 PPO 以针对奖励模型优化策略，(3) 维护 KL 惩罚以防止模型操控奖励。每个阶段都有自己的超参数和失败模式。`,
      p2: `<strong>DPO</strong>（Rafailov 等，2023）完全消除了奖励模型。它表明最优 RLHF 策略可以直接在偏好对上表达为闭式更新——将问题变成一个监督二分类任务。`,
      rlhf: {
        title: `RLHF (PPO)`,
        i1: `→ 训练独立的奖励模型`,
        i2: `→ 运行 PPO（复杂，不稳定）`,
        i3: `→ 仔细调整 KL 系数`,
        i4: `→ 监控奖励黑客行为`,
        i5: `→ 需要在线回滚（昂贵）`,
      },
      dpo: {
        title: `DPO`,
        i1: `✓ 无需奖励模型`,
        i2: `✓ 简单的二元交叉熵损失`,
        i3: `✓ 稳定训练（类似 SFT）`,
        i4: `✓ 一个 β 超参数`,
        i5: `✓ 离线（使用存储的偏好）`,
      },
    },
    s2: {
      title: `DPO 损失：直觉理解`,
      p1: `DPO 源于 RLHF 目标的重参数化。关键洞察是任何最优 RLHF 策略都可以用参考模型的对数概率来表达。这消除了奖励模型作为独立对象的需求。`,
      reward: {
        title: `隐式奖励`,
        note: `β 控制策略偏离参考模型的程度（KL 惩罚强度）`,
      },
      loss: {
        title: `DPO 损失`,
        chosen: `y_w = chosen（偏好的）回复`,
        rejected: `y_l = rejected（不偏好的）回复`,
      },
      insight: `参考模型是一个 <strong>KL 约束</strong>，而不是性能基线。它防止模型找到退化解，比如将所有概率质量分配给单词回复（"是的"）——这技术上赢得了每次比较，但实际上毫无用处。更高的 β = 更贴近 SFT 模型；更低的 β = 更激进的偏好学习。`,
    },
    s3: {
      title: `数据集格式：偏好对`,
      p1: `DPO 需要一个包含 <strong>（提示，chosen，rejected）</strong> 三元组的数据集。我们使用 <code>HuggingFaceH4/ultrafeedback_binarized</code>——6 万个样本，GPT-4 评估了四个不同模型的回复，最好的被标记为"chosen"，最差的被标记为"rejected"。`,
      pair: {
        title: `偏好对结构`,
        prompt: `提示`, chosen: `chosen`, rejected: `rejected`,
      },
      warning: `<code>ultrafeedback_binarized</code> 既有字符串 <code>"prompt"</code> 列，也有列表格式的 <code>"chosen"</code>/<code>"rejected"</code> 列。TRL 的 <code>DPOTrainer</code> 根据是否存在字符串 <code>"prompt"</code> 走不同的代码路径。保留它会导致形状不匹配或悄无声息的错误训练目标。始终删除它，让 TRL 的 <code>extract_prompt()</code> 从共享前缀中推导提示。`,
    },
    s4: {
      title: `DPOTrainConfig 和关键超参数`,
      p1m1: { k: `β (beta)`, n: `越高 → 越贴近 SFT。太低 → 奖励黑客。太高 → 无变化。` },
      p1m2: { k: `learning_rate`, n: `约比 SFT 低 40 倍。DPO 是微小调整；大学习率会破坏 SFT 对齐。` },
      p1m3: { k: `loss_type`, n: `原始 DPO 论文中的方法。'ipo' 和 'hinge' 是具有不同理论特性的替代方案。` },
      p1m4: { k: `num_epochs`, n: `过拟合风险是真实的——更多 epoch 可能降低帮助性，即使胜率在上升。` },
    },
    s5: {
      title: `使用 DPOTrainer 训练`,
      p1: `一个关键细节：我们从相同的 SFT 检查点<strong>显式</strong>加载参考模型，而不是让 TRL 自动创建它。TRL 的自动创建内部调用 <code>AutoModelForCausalLM.from_pretrained()</code>——对标准 HF 模型效果很好，但可能会遗漏我们的自定义类注册。手动加载更可靠。`,
      tip: `DPO 训练期间，TRL 每步记录 <code>rewards/chosen</code> 和 <code>rewards/rejected</code>。健康的运行会显示两者之间的差距随时间扩大——chosen 奖励相对于参考增加，rejected 奖励降低。如果两者同向移动，则 β 太低或学习率太高。`,
    },
    s6: {
      title: `训练前后评估`,
      p1: `在训练前后运行 <code>eval/dpo_eval.py</code>（第 04 章）。报告既捕捉了定量胜率改进，也包括在相同固定提示上的定性回复比较，使改变清晰可见。`,
      results: {
        title: `预期结果`,
        checkpoint: `检查点`, winrate: `胜率`, margin: `差值`,
        note: `在 <code>ultrafeedback_binarized/test_prefs</code> 的 500 个保留对上评估。实际数字因训练数据和检查点质量而异。`,
      },
      insight: `即使模型在帮助性上变差了，胜率也可能看起来不错。始终将定量胜率与自己提示上的定性采样结合使用。学会写非常长、冗长回复的模型可能仅凭长度偏差就能实现高胜率。`,
    },
  },
};
