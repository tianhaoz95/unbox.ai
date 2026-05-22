// English translations
export const en = {
  nav: {
    overview: `Overview`,
    chapters: `Chapters`,
    github: `GitHub`,
    signIn: `Sign in`,
    signOut: `Sign out`,
    soon: `Soon`,
  },
  theme: {
    toggle: `Toggle theme`,
    dark: `Dark`,
    light: `Light`,
  },
  lang: {
    en: `EN`,
    zh: `中`,
  },
  auth: {
    title: `Sign in to comment`,
    subtitle: `Select any text in a chapter and leave feedback. Your comments help us improve the content.`,
    google: `Continue with Google`,
    disclaimer: `Comments are visible to everyone. Be constructive.`,
  },
  comments: {
    button: `Comments`,
    count: `{n} comment`,
    countMany: `{n} comments`,
    panelTitle: `Comments`,
    empty: `No comments yet. Select any text to start a discussion.`,
    loading: `Loading comments…`,
    placeholder: `Leave a comment on the selected text…`,
    send: `Send`,
    cancel: `Cancel`,
    signIn: `Sign in to comment`,
    selectText: `Select text to comment`,
    justNow: `just now`,
    minsAgo: `{n}m ago`,
    hoursAgo: `{n}h ago`,
    daysAgo: `{n}d ago`,
  },
  chapter: {
    prev: `Previous`,
    next: `Next Chapter`,
    comments: `Comments`,
  },
  callout: {
    why: `Why this matters`,
    tip: `Pro tip`,
    warning: `Watch out`,
    insight: `Key insight`,
  },
  comingSoon: {
    back: `Back to overview`,
    badge: `Coming soon — in development`,
    learn: `What you'll learn`,
    meanwhile: `In the meantime, explore the available chapters:`,
    github: `Follow progress on GitHub`,
  },

  // ── Home ──────────────────────────────────────────────────────────────────
  home: {
    badge: `Build LLMs from scratch — step by step`,
    title: `Train an LLM`,
    gradient: `from scratch`,
    subtitle: `A hands-on guide to building industrial-grade language model infrastructure: data pipelines, tokenizers, pre-training, fine-tuning, and inference engines. Every component explained with code and animations.`,
    start: `Start Learning`,
    source: `View Source Code`,
    stats: {
      lines: `Lines of explained code`,
      anims: `Interactive animations`,
      chapters: `Chapters`,
      scratch: `From scratch`,
    },
    path: `Learning Path`,
    ready: `Ready`,
    soon: `Coming soon`,
    philosophy: {
      title: `The Philosophy`,
    },
    simplicity: {
      title: `Simplicity over speed`,
      desc: `Target ~50% of industry throughput while keeping every component readable and hackable. Understand before you optimize.`,
    },
    realCode: {
      title: `Real code, real system`,
      desc: `Every snippet is from unbox_platform — a complete, functional LLM stack comparable to Megatron-LM in scope, built for learning.`,
    },
    build: {
      title: `Build to understand`,
      desc: `Don't wrap APIs. Build the tokenizer. Write the training loop. Run the attention kernel. Understanding comes from implementation.`,
    },
  },

  // ── Chapter metadata ──────────────────────────────────────────────────────
  ch01: {
    title: `Data Pipeline`,
    desc: `From raw web crawl to packed token sequences. Data curation, deduplication, tokenization, and efficient batching.`,
    t1: `FineWeb-Edu dataset`, t2: `Quality filtering`, t3: `Data deduplication`, t4: `Sequence packing`,
  },
  ch02: {
    title: `Tokenizer`,
    desc: `Turn text into integers the model can process. Learn BPE, SentencePiece, and how to train your own tokenizer.`,
    t1: `BPE algorithm`, t2: `SentencePiece`, t3: `Vocabulary size tradeoffs`, t4: `Special tokens`,
  },
  ch03: {
    title: `Pre-training`,
    desc: `Train a GPT-class model from scratch. Transformer architecture, the training loop, mixed precision, and checkpointing.`,
    t1: `Transformer architecture`, t2: `Next-token prediction`, t3: `AdamW + LR schedule`, t4: `Mixed precision`,
  },
  ch04: {
    title: `Evaluation`,
    desc: `Measure model quality before committing to fine-tuning. Perplexity, qualitative sampling, and DPO win-rate.`,
    t1: `Perplexity on held-out data`, t2: `Sampling strategies (top-k, nucleus)`, t3: `Chat-format generation`, t4: `DPO win-rate metric`,
  },
  ch05: {
    title: `SFT`,
    desc: `Turn a base model into an instruction follower. HF adapter, chat templates, completion-only loss, TRL SFTTrainer.`,
    t1: `UnboxForCausalLM HF adapter`, t2: `Chat templates (ChatML)`, t3: `completion_only_loss`, t4: `SFTTrainer + SFTConfig`,
  },
  ch06: {
    title: `DPO`,
    desc: `Align the model with human preferences using chosen/rejected pairs. No reward model needed.`,
    t1: `DPO vs RLHF`, t2: `Implicit reward formulation`, t3: `β as KL constraint`, t4: `ultrafeedback_binarized dataset`,
  },
  ch07: {
    title: `Inference`,
    desc: `Serve the model at scale. Continuous batching, paged KV cache, disaggregated prefill-decode, Triton kernels.`,
    t1: `PagedAttention`, t2: `Continuous batching`, t3: `Disaggregated serving`, t4: `Triton kernels`,
  },
  ch08: {
    title: `Distillation`,
    desc: `Transfer knowledge from a large teacher to a small student via logit matching and hidden state alignment.`,
    t1: `Logit distillation`, t2: `Hidden state distillation`, t3: `GKDTrainer`, t4: `Reasoning transfer`,
  },

  // ── Data Pipeline ─────────────────────────────────────────────────────────
  data: {
    subtitle: `From raw web crawl to packed token sequences ready for training. This is the unsexy foundation everything else depends on.`,
    s1: {
      title: `Where does the data come from?`,
      p1: `Pre-training data is the single biggest lever on model quality. You need <strong>trillions of tokens</strong> of diverse, high-quality text. The industry standard is a filtered subset of <strong>Common Crawl</strong> — a petabyte-scale snapshot of the web taken monthly since 2008.`,
      p2: `For this project we use <strong>FineWeb-Edu</strong>: a 1.3 trillion token dataset filtered to educational content using a classifier trained on human ratings. It's freely available on Hugging Face and produces much better models per token than raw crawl data.`,
      why: `<strong>Why not just use Wikipedia or books?</strong> They're clean but tiny — Wikipedia is ~4B tokens, Project Gutenberg ~3B. A 760M parameter model needs ~15B+ tokens to converge (Chinchilla scaling). You need web data.`,
      tip: `Use <code>streaming=True</code> when downloading large datasets. It lets you start processing immediately without downloading the full ~200GB to disk first.`,
    },
    s2: {
      title: `Data quality: filter and deduplicate`,
      p1: `Raw web data is noisy. Pages contain spam, boilerplate, duplicated content, and text in unexpected languages. Two operations have the highest ROI:`,
      filter: {
        title: `Quality filtering`,
        i1: `Language detection (fasttext)`,
        i2: `Perplexity filtering with KenLM`,
        i3: `Remove HTML artifacts`,
        i4: `Filter short / boilerplate text`,
      },
      dedup: {
        title: `Deduplication`,
        i1: `Exact dedup with MD5/SHA-256`,
        i2: `Near-dedup with MinHash LSH`,
        i3: `URL-level dedup`,
        i4: `Paragraph-level exact match`,
      },
      insight: `Deduplication is disproportionately impactful. GPT-3's training data was ~3% duplicates, but removing them improved validation loss more than adding equivalent unique tokens. The model memorizes duplicates instead of learning.`,
    },
    s3: {
      title: `Tokenize and pack sequences`,
      p1: `After filtering, we convert raw text to token IDs and pack them into fixed-length sequences. This is a one-time offline step that produces the exact numpy array we load during training.`,
      why: `<strong>Why pack instead of pad?</strong> Padding wastes computation on tokens that contribute zero gradient. With padding you might hit 60-70% token utilization. Sequence packing with document boundaries gives you <strong>~100% utilization</strong>. At scale this is the difference between a $1M training run and a $600K one.`,
      vis: {
        title: `Sequence packing visualized`,
        note: `All 3 documents packed into one 1024-token sequence with <code>&lt;eos&gt;</code> separators. 100% token utilization.`,
        packed: `Packed`,
      },
    },
    s4: {
      title: `Build the DataLoader`,
      p1: `The final step is loading the packed numpy arrays into a PyTorch <code>DataLoader</code>. We use <strong>memory mapping</strong> so the OS pages in only the chunks needed — no full dataset in RAM.`,
      tip: `<strong>num_workers=4</strong> launches 4 background processes that prefetch batches while the GPU is busy on the previous step. Without this, CPU data loading becomes the bottleneck. Set <code>pin_memory=True</code> to enable faster CPU→GPU DMA transfers.`,
      stat1: { k: `Batch size`, v: `32–512`, n: `per GPU, tune to memory` },
      stat2: { k: `Seq length`, v: `1024–8192`, n: `longer = more compute` },
      stat3: { k: `Token utilization`, v: `~100%`, n: `vs 60-70% with padding` },
    },
    s5: {
      title: `Run it`,
      p1: `The full pipeline is wired up as a CLI command. Run it once before training:`,
      insight: `You only run the data pipeline <strong>once</strong>. The resulting <code>.npy</code> file is your training data for all experiments. If you change the tokenizer vocabulary, you need to re-tokenize. If you just change model architecture or training hyperparameters, you don't.`,
    },
  },

  // ── Tokenizer ─────────────────────────────────────────────────────────────
  tok: {
    subtitle: `Text is a string. Models need integers. The tokenizer is the bridge — and its design cascades through every downstream decision.`,
    s1: {
      title: `What is a tokenizer?`,
      p1: `A tokenizer splits text into <strong>tokens</strong> — discrete units that map to integers from a fixed vocabulary. The model never sees characters or bytes directly; it operates entirely on token IDs.`,
      p2: `The vocabulary is learned from training data. Common subwords get their own ID; rare words are split into multiple tokens. "tokenization" might become <code>["token", "ization"]</code>. "ChatGPT" might be <code>["Chat", "G", "PT"]</code>.`,
      table: {
        title: `Industry vocabulary sizes`,
        model: `Model`, vocab: `Vocab size`, notes: `Notes`,
      },
      why: `<strong>Why not just split on words or characters?</strong> Word-level vocabularies explode in size (millions of words, OOV problem). Character-level vocabularies are tiny but sequences become very long — 4× longer than BPE, meaning 4× more attention operations. BPE finds the sweet spot.`,
    },
    s2: {
      title: `The BPE algorithm`,
      p1: `Byte Pair Encoding (BPE) builds a vocabulary by starting with individual characters and iteratively merging the <strong>most frequent adjacent pair</strong>. After N merges, you have a vocabulary of N + (initial alphabet size) tokens.`,
      insight: `The merge order matters and is deterministic given the corpus. When you save a tokenizer, you're saving the ordered list of merges. At inference time, you apply those same merges greedily to encode new text. This is why tokenizers are <strong>corpus-specific</strong> — a tokenizer trained on English code will be inefficient for Japanese text.`,
      bpe: {
        title: `BPE strengths`,
        i1: `Language-agnostic at byte level`,
        i2: `Handles any Unicode gracefully`,
        i3: `Vocabulary is interpretable`,
        i4: `Fast encoding (O(n log n))`,
      },
      sp: {
        title: `SentencePiece differences`,
        i1: `Works on raw text (no pre-tokenization)`,
        i2: `Handles spaces as explicit tokens`,
        i3: `Unigram Language Model variant`,
        i4: `Used by LLaMA, T5, ALBERT`,
      },
    },
    s3: {
      title: `Train your tokenizer`,
      p1: `We use the HuggingFace <code>tokenizers</code> library — it's written in Rust and trains a 32k vocab tokenizer in under 2 minutes on 10GB of text.`,
      warning: `Train the tokenizer <strong>before</strong> tokenizing your data (Chapter 01). The tokenizer is a prerequisite for the data pipeline. In practice, you train the tokenizer on a representative sample (e.g. 1B tokens), then tokenize the full dataset with the resulting vocabulary.`,
    },
    s4: {
      title: `Using the tokenizer at inference`,
      p1: `During inference, the tokenizer has an extra responsibility: incremental streaming decode. The model generates one token at a time, but we can't always decode a single token to a character — some characters span multiple tokens in multi-byte UTF-8 encoding.`,
      insight: `This is why streaming LLM outputs sometimes have a short delay before the first characters appear — the server is buffering tokens until it can confirm a complete UTF-8 sequence. The same issue occurs with special characters in Chinese, Arabic, emoji, etc.`,
    },
    s5: {
      title: `Key design decisions`,
      q1: { q: `Vocab size: 32k vs 128k?`, a: `32k is the LLaMA 2 sweet spot. 128k (LLaMA 3) trades a larger embedding table for shorter sequences in multilingual settings. For English-only research, 32k is more parameter-efficient.` },
      q2: { q: `Add domain tokens?`, a: `Yes if you have domain-specific strings that fragment badly — e.g., "<|endoftext|>", code keywords, or math notation. Add them as special tokens before training.` },
      q3: { q: `Case sensitivity?`, a: `Keep it. Lowercasing loses information the model can learn from. BPE naturally handles case variants as separate tokens ('The' vs 'the'), and the model learns they're related.` },
      q4: { q: `Re-use a pretrained tokenizer?`, a: `Absolutely fine for research. Using LLaMA's tokenizer saves you the training step and ensures compatibility with pretrained weights for fine-tuning experiments.` },
    },
  },

  // ── Pre-training ──────────────────────────────────────────────────────────
  pre: {
    subtitle: `Feed the packed token sequences into a Transformer and predict the next token. This is where the model learns language, facts, and reasoning.`,
    s1: {
      title: `The objective: next-token prediction`,
      p1: `Pre-training is conceptually simple: given a sequence of tokens, predict the next one. Do this for trillions of tokens from diverse text and the model learns grammar, facts, reasoning patterns, and much more.`,
      tf: {
        title: `Teacher forcing`,
        input: `Input`, target: `Target`,
        note: `Input is shifted by 1. Every position predicts the next token in parallel during training.`,
      },
      insight: `One training example generates T prediction tasks simultaneously. A 1024-token sequence gives you 1023 (input, target) pairs in a single forward pass. This is why transformer training is so much more efficient than recurrent networks, which process one step at a time.`,
    },
    s2: {
      title: `Model architecture: LLaMA-style Transformer`,
      p1: `The model uses a <strong>decoder-only Transformer</strong> with modern improvements over the original 2017 architecture: RMSNorm instead of LayerNorm, SwiGLU activation, Rotary Position Embeddings (RoPE), and Grouped Query Attention (GQA).`,
      r1: `Cheaper, same quality — removes mean subtraction step`,
      r2: `Relative positions via rotation — generalizes to longer contexts`,
      r3: `SiLU × gate = smoother gradient, empirically better loss`,
      r4: `Fewer KV heads → smaller KV cache → more tokens in memory`,
      why: `<strong>Why no bias terms?</strong> Linear layers in modern LLMs are typically bias-free. The bias saves ~0.1% parameters but adds noise to the gradient and doesn't help. LLaMA, Mistral, Qwen all use <code>bias=False</code>.`,
    },
    s3: {
      title: `The training loop`,
      p1: `The training loop is the innermost hot path. Every line matters for both correctness and performance. Here are the six key operations in order: forward, loss, backward, grad clip, optimizer step, LR step.`,
      bf16: `<strong>BF16 vs FP16:</strong> Use BF16 (bfloat16) if your GPU supports it (Ampere+). BF16 has the same exponent range as FP32 so it doesn't overflow; FP16 requires loss scaling. BF16 is strictly better for LLM training.`,
      gradacc: `<strong>Gradient accumulation:</strong> If your batch doesn't fit in GPU memory, accumulate gradients over N micro-batches before calling <code>optimizer.step()</code>. Divide the loss by N before each backward.`,
    },
    s4: {
      title: `Learning rate schedule`,
      p1: `The learning rate schedule is one of the most impactful hyperparameters. Too high and training diverges. Too low and you waste compute. The standard recipe for LLMs is: <strong>linear warmup → cosine decay → minimum LR</strong>.`,
      lr: {
        title: `Learning rate curve`,
        maxN: `Peak LR`, warmupN: `~0.1% of training`, minN: `max_lr / 10`,
      },
    },
    s5: {
      title: `Checkpointing and resumability`,
      p1: `Pre-training runs for days or weeks. You <em>will</em> have hardware failures, preemptions, and experiments that need to be resumed. Save everything you need to resume exactly where you left off.`,
      warning: `Save <strong>optimizer state</strong>, not just model weights. AdamW's moment estimates (m, v) accumulate knowledge about the gradient history. Restoring only weights means the optimizer restarts cold — you'll see a temporary loss spike and 500-1000 steps of wasted compute.`,
      stat1: { k: `Save frequency`, v: `Every 1k steps`, n: `balance overhead vs recovery cost` },
      stat2: { k: `Keep N ckpts`, v: `Last 3–5`, n: `delete old ones to save disk` },
      stat3: { k: `Upload to`, v: `ModelScope / S3`, n: `for cross-machine resume` },
    },
    s6: {
      title: `Run pre-training`,
      p1: `With data prepared (Chapter 01) and tokenizer trained (Chapter 02), you're ready to launch pre-training:`,
      insight: `A <strong>760M parameter model</strong> trained on 15B tokens (the Chinchilla optimal point for this size) takes roughly 3 days on 8× A100-80GB GPUs. The loss should drop from ~10 (random) to ~2.3–2.5 (competent English text generation) over this run.`,
      chinchilla: {
        title: `Chinchilla scaling law`,
        desc: `The Chinchilla paper (Hoffmann et al., 2022) showed that the optimal compute allocation trains a smaller model on more data: roughly <strong>20 tokens per parameter</strong>. 760M params → 15B tokens is Chinchilla-optimal.`,
        params: `Model params`, tokens: `Optimal tokens`, ratio: `Ratio`,
      },
    },
  },

  // ── Evaluation ────────────────────────────────────────────────────────────
  eval: {
    subtitle: `After pretraining you have a model — but is it any good? Evaluation tells you where you are before committing to expensive fine-tuning runs.`,
    s1: {
      title: `Why evaluate before fine-tuning?`,
      p1: `Pretraining is expensive. Before you spend more compute on SFT or DPO, you need to know two things: does the model have coherent language understanding (perplexity), and does it produce reasonable text (qualitative sampling)? These checks take minutes and can save you from wasting days fine-tuning a broken base model.`,
      tool1: { tool: `eval/perplexity.py`, when: `After pretraining`, desc: `Measures how surprised the model is by held-out text. Lower = better language model.` },
      tool2: { tool: `eval/sample.py`, when: `After pretraining`, desc: `Generates text from the base model. Checks coherence before any instruction tuning.` },
      tool3: { tool: `eval/chat_sample.py`, when: `After SFT`, desc: `Chat-format generation from an SFT checkpoint loaded via the HF adapter.` },
      why: `Evaluation is not just a final step — it's a <strong>feedback loop</strong>. Run perplexity every N thousand training steps to confirm loss is decreasing on held-out data and you haven't overfit. A diverging eval loss while training loss keeps dropping is the clearest overfitting signal.`,
      runLabel: `Run:`,
    },
    s2: {
      title: `Perplexity: the standard language model metric`,
      p1: `Perplexity is <strong>e^(average cross-entropy loss)</strong> on held-out text. Intuitively: if the model assigns a perplexity of 20, it's as uncertain as choosing uniformly among 20 options at every token. Lower is better. A random model over a 32k vocabulary has perplexity ≈ 32,000. A well-trained 760M model should reach ~15–20 on FineWeb-Edu eval text.`,
      formula: {
        title: `The formula`,
        note: `N = total tokens evaluated; the exponent is the average negative log-likelihood (= cross-entropy loss)`,
      },
      warning: `<strong>Don't average the per-batch losses directly.</strong> Each batch has a different number of non-padding tokens. You must recover the token-level sum (<code>loss × n_tokens</code>), accumulate globally, then divide by total tokens. Averaging averages gives a biased estimate.`,
    },
    s3: {
      title: `Sampling strategies: temperature, top-k, top-p`,
      p1: `At each generation step, the model outputs a logit vector over the full vocabulary. Sampling is how you convert those logits into a single token. The three knobs — temperature, top-k, and top-p — control the creativity vs. coherence tradeoff.`,
      sweet: `Sweet spot:`,
      p1m1: { name: `Temperature τ`, formula: `softmax(logits / τ)`, low: `τ→0: greedy, repetitive`, high: `τ→∞: uniform noise`, sweet: `τ = 0.7–0.9` },
      p1m2: { name: `Top-k`, formula: `keep k highest logits`, low: `k=1: greedy`, high: `k=vocab_size: none`, sweet: `k = 40–100` },
      p1m3: { name: `Top-p (nucleus)`, formula: `keep smallest set ≥ p`, low: `p→0: greedy`, high: `p=1.0: none`, sweet: `p = 0.9–0.95` },
    },
    s4: {
      title: `Chat sampling from an SFT checkpoint`,
      p1: `After SFT (Chapter 05), the checkpoint is saved in HuggingFace format via <code>UnboxForCausalLM</code>. Evaluation looks different: you load with <code>from_pretrained</code>, apply a chat template, and call HF's <code>model.generate()</code> — not the raw Transformer's.`,
      p2: `Two non-obvious pitfalls trip up most people the first time:`,
      pitfall1: `When tokenizing chat-template output, always pass <code>add_special_tokens=False</code>. The template already includes all special tokens. Without this, a spurious <code>&lt;|eos|&gt;</code> is appended and the model generates a fake next-user-turn instead of your answer.`,
      pitfall2: `<code>UnboxForCausalLM</code> doesn't implement the KV cache yet. If you call <code>generate(use_cache=True)</code> (the HF default), HF passes only the last token on step 2+, destroying all context and producing garbage. Always pass <code>use_cache=False</code>.`,
    },
    s5: {
      title: `DPO evaluation: win rate and log-prob margin`,
      p1: `Before running DPO (Chapter 06), establish a baseline by measuring how often the SFT model already prefers the "chosen" response over the "rejected" one. A well-calibrated SFT model should already win ~55–65% of pairs; DPO should push this to ~70–80%.`,
      metricsTitle: `Two metrics`,
      m1: { m: `Win rate`, formula: `fraction where log P(chosen) > log P(rejected)`, baseline: `~50% = random, ~58–65% = SFT, ~73%+ = good DPO` },
      m2: { m: `Margin`, formula: `mean( log P(chosen) − log P(rejected) ) per token`, baseline: `positive = model prefers chosen; larger = stronger preference` },
      insight: `Use <strong>per-token</strong> log-probability (mean loss, not sum) when comparing responses. A long correct answer would always win over a short one if you used the total log-prob — the length bias would dominate the quality signal.`,
    },
    s6: {
      title: `Run it`,
      tip: `Add <code>--max-batches 200</code> to perplexity for a quick smoke test (~2 min on CPU). Remove it for the full eval (~20 min on GPU). The estimate from 200 batches is within 0.3 PPL of the full eval in practice.`,
    },
  },

  // ── SFT ───────────────────────────────────────────────────────────────────
  sft: {
    subtitle: `A pretrained base model predicts text. SFT turns it into an assistant that follows instructions. The key is the dataset format and how you compute the loss.`,
    s1: {
      title: `The HuggingFace adapter: why it exists`,
      p1: `Our pretrained <code>Transformer</code> is a clean PyTorch module — but TRL's <code>SFTTrainer</code> expects a HuggingFace <code>PreTrainedModel</code>. The <code>UnboxForCausalLM</code> adapter bridges this gap: it wraps the existing model with no architectural changes, just the HF interface layer.`,
      p2: `This means we get TRL, PEFT/LoRA, and the HF ecosystem for free without ever touching the core model code — exactly the separation of concerns the architecture is designed for.`,
      insight: `<code>UnboxForCausalLM</code> inherits from both <code>PreTrainedModel</code> and <code>GenerationMixin</code>. <code>GenerationMixin</code> provides the full <code>model.generate()</code> loop — beam search, sampling, stopping criteria — for free. We just need a correct <code>forward()</code>.`,
      warning: `The adapter accepts <code>attention_mask</code> and <code>use_cache</code> for HF API compatibility but neither is implemented. Always pass <code>use_cache=False</code> to <code>generate()</code> — see Eval chapter (Chapter 04) for the exact pitfall.`,
    },
    s2: {
      title: `Chat templates: the message → token mapping`,
      p1: `SFT training data is a list of <code>{"role": ..., "content": ...}</code> message dicts. A <strong>chat template</strong> converts this structure into a flat string that the tokenizer can process. The most common format today is ChatML, used by Qwen, Mistral-Instruct, and many others.`,
      chatml: {
        title: `ChatML format`,
        masked: `masked (labels = -100)`,
        loss: `loss computed`,
      },
      why: `<strong>Why mask user turns?</strong> Without <code>completion_only_loss=True</code>, the model learns to predict the user's message too. During inference it starts generating <em>"User: ..."</em> continuation instead of staying in the assistant role. Masking forces all gradient signal through the assistant response only.`,
    },
    s3: {
      title: `SFTTrainConfig: the hyperparameters`,
      p1: `SFT uses a much lower learning rate than pretraining — typically <strong>1e-5 to 5e-5</strong> vs. 3e-4 for pretraining. The model is already well-initialized; we're nudging it toward instruction following, not learning language from scratch.`,
      p1m1: { k: `max_lr`, n: `10× lower than pretrain — preserves base knowledge` },
      p1m2: { k: `num_epochs`, n: `more epochs → overfitting on small datasets` },
      p1m3: { k: `packing`, n: `True = more efficient but loses conversation boundaries` },
      p1m4: { k: `max_seq_len`, n: `cap to avoid OOM on multi-turn conversations` },
    },
    s4: {
      title: `Training with SFTTrainer`,
      p1: `TRL's <code>SFTTrainer</code> handles the full training loop: chat template application, tokenization, masking, gradient accumulation, evaluation, and checkpoint saving. The dataset must have a <code>"messages"</code> column — if it also has a <code>"prompt"</code> column TRL takes a different (and worse) code path, so we drop everything except <code>"messages"</code>.`,
      tip: `SFT training is usually only <strong>1 epoch</strong> over the dataset. More epochs produce diminishing returns on instruction following and increase the risk of overfit (the model starts memorizing specific responses). If you have &lt;50K examples, consider 2–3 epochs with early stopping.`,
    },
    s5: {
      title: `Run it`,
      stat1: { l: `Dataset size`, v: `200K turns`, n: `ultrachat_200k` },
      stat2: { l: `Training time`, v: `~4 hours`, n: `on 2× A100-80GB` },
      stat3: { l: `Expected SFT loss`, v: `~1.4–1.6`, n: `after 1 epoch` },
      insight: `The SFT checkpoint is saved in HuggingFace format (not <code>.pt</code>). That means <code>AutoTokenizer.from_pretrained()</code> and <code>UnboxForCausalLM.from_pretrained()</code> both work directly on the output directory. This is the checkpoint you hand to DPO next.`,
    },
  },

  // ── DPO ───────────────────────────────────────────────────────────────────
  dpo: {
    subtitle: `Direct Preference Optimization aligns the model with human preferences using pairs of chosen and rejected responses — no reward model required.`,
    s1: {
      title: `Why not RLHF?`,
      p1: `Classic RLHF (Reinforcement Learning from Human Feedback) requires three separate stages: (1) train a reward model from preference pairs, (2) run PPO to optimize the policy against the reward model, (3) maintain a KL penalty to prevent the model from gaming the reward. Each stage has its own hyperparameters and failure modes.`,
      p2: `<strong>DPO</strong> (Rafailov et al., 2023) eliminates the reward model entirely. It shows that the optimal RLHF policy can be expressed as a closed-form update directly on the preference pairs — making the problem a supervised binary classification task.`,
      rlhf: {
        title: `RLHF (PPO)`,
        i1: `→ Train a separate reward model`,
        i2: `→ Run PPO (complex, unstable)`,
        i3: `→ Tune KL coefficient carefully`,
        i4: `→ Monitor reward hacking`,
        i5: `→ Needs online rollouts (expensive)`,
      },
      dpo: {
        title: `DPO`,
        i1: `✓ No reward model needed`,
        i2: `✓ Simple binary cross-entropy loss`,
        i3: `✓ Stable training (like SFT)`,
        i4: `✓ One β hyperparameter`,
        i5: `✓ Offline (uses stored preferences)`,
      },
    },
    s2: {
      title: `The DPO loss: intuition`,
      p1: `DPO derives from a reparameterization of the RLHF objective. The key insight is that any optimal RLHF policy can be written in terms of the reference model's log-probabilities. This eliminates the reward model as a separate object.`,
      reward: {
        title: `Implicit reward`,
        note: `β controls how far the policy can diverge from the reference (KL penalty strength)`,
      },
      loss: {
        title: `DPO loss`,
        chosen: `y_w = chosen (preferred) response`,
        rejected: `y_l = rejected (dispreferred) response`,
      },
      insight: `The reference model is a <strong>KL constraint</strong>, not a performance baseline. It prevents the model from degenerate solutions like assigning all probability mass to one-word responses ("Yes.") that technically win every comparison but are useless in practice. Higher β = stay closer to the SFT model; lower β = more aggressive preference learning.`,
    },
    s3: {
      title: `Dataset format: preference pairs`,
      p1: `DPO requires a dataset of <strong>(prompt, chosen, rejected)</strong> triples. We use <code>HuggingFaceH4/ultrafeedback_binarized</code> — 60K examples where GPT-4 rated responses from four different models and the best was labeled "chosen", worst "rejected".`,
      pair: {
        title: `Preference pair structure`,
        prompt: `prompt`, chosen: `chosen`, rejected: `rejected`,
      },
      warning: `<code>ultrafeedback_binarized</code> has both a string <code>"prompt"</code> column and list-format <code>"chosen"</code>/<code>"rejected"</code> columns. TRL's <code>DPOTrainer</code> has two code paths depending on whether a string <code>"prompt"</code> exists. Leaving it in causes shape mismatches or silently wrong training targets. Always drop it and let TRL's <code>extract_prompt()</code> derive the prompt from the shared prefix.`,
    },
    s4: {
      title: `DPOTrainConfig and key hyperparameters`,
      p1m1: { k: `β (beta)`, n: `Higher → stay closer to SFT. Too low → reward hack. Too high → no change.` },
      p1m2: { k: `learning_rate`, n: `~40× lower than SFT. DPO is a small adjustment; large LR destroys the SFT alignment.` },
      p1m3: { k: `loss_type`, n: `The original DPO paper. 'ipo' and 'hinge' are alternatives with different theoretical properties.` },
      p1m4: { k: `num_epochs`, n: `Overfit risk is real — more epochs can degrade helpfulness even as win rate climbs.` },
    },
    s5: {
      title: `Training with DPOTrainer`,
      p1: `One critical detail: we load the reference model <strong>explicitly</strong> from the same SFT checkpoint rather than letting TRL auto-create it. TRL's auto-creation calls <code>AutoModelForCausalLM.from_pretrained()</code> internally — which works fine for standard HF models but can miss our custom class registration. Loading manually is more reliable.`,
      tip: `During DPO training, TRL logs <code>rewards/chosen</code> and <code>rewards/rejected</code> per step. A healthy run shows the gap between them widening over time — chosen rewards increasing and rejected rewards decreasing relative to the reference. If both move together in the same direction, β is too low or the LR is too high.`,
    },
    s6: {
      title: `Evaluate before and after`,
      p1: `Run <code>eval/dpo_eval.py</code> (Chapter 04) before and after training. The report captures both the quantitative win-rate improvement and qualitative response comparisons on the same fixed prompts, making the delta legible.`,
      results: {
        title: `Expected results`,
        checkpoint: `Checkpoint`, winrate: `Win rate`, margin: `Margin`,
        note: `Evaluated on 500 held-out pairs from <code>ultrafeedback_binarized/test_prefs</code>. Actual numbers vary with training data and checkpoint quality.`,
      },
      insight: `Win rate can look good even when the model has gotten worse at helpfulness. Always pair quantitative win-rate with qualitative sampling on your own prompts. A model that learned to write very long, verbose responses can achieve high win-rate by length bias alone.`,
    },
  },
};

export type Translations = typeof en;
