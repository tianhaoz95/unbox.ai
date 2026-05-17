# unbox.ai

**A full-stack LLM research and education platform — built from scratch.**

---

unbox.ai is an open-source project that implements every stage of a modern large language model pipeline: dataset curation, tokenizer training, pre-training, supervised fine-tuning, reinforcement learning from human feedback, and a production-grade inference engine. Every component is written to be read and understood, not just run.

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **Dataset**

    ---

    FineWeb-Edu 10BT — 10 billion tokens of high-quality educational web text, filtered and prepared for language model training.

    [:octicons-arrow-right-24: Dataset selection](curriculum/01_dataset.md)

-   :material-alphabetical:{ .lg .middle } **Tokenizer**

    ---

    A 32,768-token ByteLevel BPE tokenizer trained from scratch on the pretraining corpus, with a ChatML chat template.

    [:octicons-arrow-right-24: Tokenizer training](curriculum/03_tokenizer.md)

-   :material-brain:{ .lg .middle } **Architecture**

    ---

    A 760M-parameter decoder-only Transformer with GQA, SwiGLU, and RoPE — the same building blocks as Llama and Qwen.

    [:octicons-arrow-right-24: Model architecture](curriculum/04_architecture.md)

-   :material-chart-line:{ .lg .middle } **Pre-training**

    ---

    Training on ~10B tokens with cosine LR scheduling, bfloat16 mixed precision, gradient accumulation, and W&B tracking.

    [:octicons-arrow-right-24: Pre-training](curriculum/05_pretraining.md)

-   :material-tune:{ .lg .middle } **SFT**

    ---

    Supervised fine-tuning via TRL's SFTTrainer on UltraChat 200k, using the ChatML chat template.

    [:octicons-arrow-right-24: Supervised fine-tuning](curriculum/06_sft.md)

-   :material-robot:{ .lg .middle } **RL**

    ---

    Offline RL (DPO, GRPO) via TRL; online PPO via OpenRLHF — mirroring the full RLHF pipeline used in production.

    [:octicons-arrow-right-24: Reinforcement learning](curriculum/07_rl.md)

-   :material-server:{ .lg .middle } **Inference Engine**

    ---

    A fully-fledged serving engine: paged KV cache, continuous batching, disaggregated prefill-decode, NCCL tensor parallelism, ZMQ IPC, and Triton kernels.

    [:octicons-arrow-right-24: Inference engine](curriculum/08_inference.md)

-   :material-flask:{ .lg .middle } **Experiments**

    ---

    A living list of research experiments to run once the core platform is validated.

    [:octicons-arrow-right-24: Planned experiments](experiments.md)

</div>

---

## Design philosophy

> **Simplicity over peak performance.** Target ~50% of industry throughput while keeping every implementation readable enough to modify in an afternoon.

This project is not trying to beat vLLM or Megatron-LM at their own game. It is trying to make every piece of the LLM stack legible — so a researcher can read any file top-to-bottom, understand what it does, swap a component out, and run a new experiment.

Every major architectural decision has a documented rationale. Every training run produces a sanity-check report. Every subsystem has tests.

## Status

| Stage | Status |
|---|---|
| Tokenizer | ✅ Complete |
| Pre-training | 🔄 In progress (~20% of tokens) |
| SFT | 🔄 Running |
| RL | 📋 Planned |
| Inference engine | 📋 Planned |
