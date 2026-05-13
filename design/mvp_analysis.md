# MVP Analysis: minimind Reference & unbox.ai POC Design

## Objective

Build a verifiable MVP covering `platform/model`, `platform/tokenizer`, and `platform/train` — a small model that can generate coherent text after pretraining, completing within 72 hours on a DGX Spark.

---

## What We Can Copy Directly from minimind

### Model Architecture

The core transformer stack in `model/model_minimind.py` is clean and well-chosen:

- **Pre-norm + RMSNorm**: modern standard, no reason to deviate
- **RoPE positional encoding**: copy directly; the YaRN extension is a bonus for long-context experiments later
- **SwiGLU FFN**: standard across Llama/Qwen/Mistral family; keep it
- **GQA (Grouped Query Attention)**: good default even at small scale — keeps KV cache small and the code is already written
- **Tied input/output embeddings**: saves parameters, standard practice
- **Flash Attention via `scaled_dot_product_attention`**: free speedup, no custom kernel needed

The config system (`MiniMindConfig`) is minimal and clean. We can adapt it directly with minor renaming.

**What to skip for MVP**: MoE variant. It adds routing complexity and aux loss tuning that is not needed to verify the baseline works.

### Tokenizer

The BPE tokenizer trained via `tokenizers` library on domain data is the right approach. Key decisions to carry over:

- ByteLevel BPE (handles any UTF-8 without unknown tokens)
- Vocabulary size 6,400 is surprisingly effective for a research platform — keep it for now, revisit for production
- Special token layout (BOS/EOS/PAD + reserved slots for future features) is well thought out
- Chat template via Jinja2 in `tokenizer_config.json` — copy this pattern; it keeps the tokenizer self-contained

### Training Loop Patterns

- **Cosine LR schedule with floor** (`get_lr` in `trainer_utils.py`): simple, effective, copy
- **Gradient accumulation + gradient clipping**: standard, copy
- **bfloat16 mixed precision via `torch.amp.autocast`**: correct choice for modern GPUs, copy
- **Checkpoint save/resume with optimizer state**: the pattern in `lm_checkpoint` is solid; adapt it
- **DDP via `DistributedDataParallel`**: minimind uses DDP which is fine for the MVP; we will replace with FSDP for larger models but DDP is correct for the POC

### Data Pipeline

- JSONL with a "text" field for pretraining, tokenize-on-the-fly in `PretrainDataset`
- BOS + tokens + EOS + padding, labels masked at padding positions
- `DistributedSampler` for multi-GPU data sharding

---

## What Does Not Align With Our Mission

### No Separation Between Platform and Experiment Code

minimind is a monolithic script collection — `train_pretrain.py` hardcodes the model class, optimizer, and data path. There is no injection point for swapping components. For our platform, training loops must accept the model, optimizer, and dataloader as arguments so `/unbox` experiments can plug in variants without forking the loop.

**Resolution**: extract a `Trainer` class (or plain function) in `platform/train/` that takes `model`, `optimizer`, `dataloader`, `config` as arguments.

### DDP Only — No 3D Parallelism

minimind uses `DistributedDataParallel` which requires the full model to fit in each GPU's memory and provides no tensor or pipeline parallelism. Our platform targets industry-scale training (30B+ models) which requires all three parallelism axes composed together. Starting with DDP for the MVP is acceptable, but the abstraction must be designed so TP and PP are additive rather than architectural rewrites later.

**Resolution**: wrap model parallelism behind a `setup_model(model, parallel_config)` utility from day one, even if `parallel_config` only supports DDP in the MVP. The interface should anticipate TP rank groups and PP stages.

### Hardcoded Hyperparameters in Scripts

minimind puts hyperparameters (`batch_size=32`, `lr=5e-4`, `max_seq_len=340`) as argparse defaults directly in `train_pretrain.py`. This makes systematic hyperparameter sweeps awkward.

**Resolution**: all hyperparameters live in a typed config dataclass (or TOML/YAML file), passed to the training entry point. Scripts are thin wrappers around `platform/train/pretrain.py`.

### No Evaluation Infrastructure

minimind has `eval_llm.py` for interactive chat but no systematic benchmark harness (perplexity curves, downstream task accuracy). For our MVP, even a basic perplexity-on-held-out-data eval run is sufficient, but it needs to be a first-class step in the workflow, not an afterthought.

**Resolution**: `platform/eval/` gets a `perplexity.py` that runs on a held-out split as part of the POC acceptance criteria.

### Tokenizer Vocabulary Is Too Small for General Use

6,400 tokens produces compression ratios that are poor for English (~4-5 chars/token vs. ~4 chars/token for a 32K vocab). For a research platform targeting English-heavy pretraining data, a 32K vocabulary is the right default.

**Resolution**: train a 32K BPE tokenizer for our platform. The minimind tokenizer training script is a good template; adjust vocab size and train on a more representative corpus.

### No Type Annotations or Config Validation

minimind uses duck typing throughout. For a platform meant to be modified frequently by researchers, silent type errors are a productivity killer.

**Resolution**: use `dataclasses` with type hints for all configs. No need for Pydantic at this stage.

---

## MVP Scope

The POC acceptance criterion is: **a pretrained model that generates grammatically coherent English text**, verified by perplexity on a held-out split and qualitative sampling.

### Components to Build (in order)

| Component | Based on minimind? | Key delta |
|---|---|---|
| `platform/tokenizer/` — BPE training | Yes, template from `train_tokenizer.py` | 32K vocab, English corpus |
| `platform/model/` — transformer | Yes, port from `model_minimind.py` | Type hints, injection-friendly config, no MoE |
| `platform/data/` — pretrain dataset | Yes, port from `lm_dataset.py` | Typed config, decouple from model class |
| `platform/train/pretrain.py` | Yes, port from `train_pretrain.py` | Config-driven, FSDP-ready abstraction |
| `platform/eval/perplexity.py` | New (not in minimind) | Minimal: perplexity on held-out split |

### Target Model Size for 72hr DGX Spark Budget

DGX Spark (GB10 Grace Blackwell) provides ~1000 TOPS with 128GB unified memory. For a rough estimate:

- minimind 26M param model trains in ~8hrs on a single RTX 3090 (35 TFLOPS) on the full 10GB dataset
- DGX Spark is approximately 10-15x faster for this workload → same model trains in ~30-45 minutes
- 72hr budget → can scale to a **~360M–1B parameter model** on a comparable dataset

**Recommended POC size**: **~360M parameters** (e.g., hidden=1024, layers=24, heads=16, GQA kv_heads=8). This is large enough to produce clearly coherent text and stress-test the training infrastructure, but trains comfortably within the 72hr window with room to spare for debugging and iteration.

### Config Target

```
hidden_size:     1024
num_layers:      24
num_heads:       16
num_kv_heads:    8      # GQA
ffn_intermediate: ~2730  # ceil(1024 * π / 64) * 64
vocab_size:      32768
max_seq_len:     2048
params:          ~360M
```

---

## What to Leave Out of MVP

- MoE routing
- YaRN long-context scaling (keep RoPE base, add YaRN later in `/unbox`)
- LoRA / SFT / RL (those come after pretraining is verified)
- Inference server (out of MVP scope)
- Anything requiring a reward model

---

## Open Questions Before Implementation

1. **Pretraining corpus**: minimind uses a Chinese-English mix dataset. What is our target data source and language distribution?
2. **Framework decision**: DDP for MVP is fine, but do we want to commit to FSDP as the production strategy now so the abstraction is designed correctly from day one?
3. **Evaluation bar**: is perplexity on a held-out split sufficient to call the POC verified, or do we want a small downstream benchmark (e.g., BLiMP, HellaSwag subset)?
