import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { TransformerAnim } from "../components/animations/TransformerAnim";
import { TrainingLoopAnim } from "../components/animations/TrainingLoopAnim";

const modelConfigCode = `# unbox_platform/model/config.py
from dataclasses import dataclass, field
from transformers import PretrainedConfig

@dataclass
class UnboxConfig(PretrainedConfig):
    """
    Configuration for the Unbox language model.
    Inherits from HuggingFace PretrainedConfig for full ecosystem compatibility
    — this lets TRL, PEFT, and eval harnesses consume the model without modification.
    """
    model_type: str = "unbox"

    # Architecture
    vocab_size: int = 32_000
    hidden_dim: int = 1024          # d_model
    num_layers: int = 24            # transformer depth
    num_heads: int = 16             # attention heads
    num_kv_heads: int = 8           # GQA: fewer KV heads → smaller KV cache
    ffn_multiplier: float = 8/3     # SwiGLU: 2/3 * 4 = 8/3 of hidden_dim
    max_seq_len: int = 4096
    rope_theta: float = 10_000.0    # RoPE base frequency

    # Training
    dropout: float = 0.0            # 0.0 at scale — dropout hurts large models
    norm_eps: float = 1e-5

    # Derived
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return int(self.hidden_dim * self.ffn_multiplier)
`;

const modelArchCode = `# unbox_platform/model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import UnboxConfig

class RMSNorm(nn.Module):
    """Root Mean Square normalization — simpler than LayerNorm, same quality."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize by RMS instead of mean+variance
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) and RoPE.

    GQA: num_kv_heads < num_heads — share K/V projections across head groups.
    This reduces the KV cache size proportionally without significant quality loss.
    LLaMA 2 70B uses GQA; LLaMA 2 7B uses MHA (num_kv_heads == num_heads).
    """
    def __init__(self, config: UnboxConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.groups = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Repeat K/V to match Q heads (GQA)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # Flash Attention via PyTorch — dispatches to FA2 if available
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
`;

const trainingLoopCode = `# unbox_platform/train/pretrain.py
import torch
from torch.amp import autocast, GradScaler
from .config import TrainConfig

def train(config: TrainConfig) -> None:
    model = build_model(config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(0.9, 0.95),   # β2=0.95 is standard for LLM pre-training
        weight_decay=0.1,    # L2 regularization on weights (not biases/norms)
        fused=True,          # fused AdamW: ~2x faster on CUDA
    )
    scaler = GradScaler()    # for mixed precision (BF16/FP16)
    scheduler = build_cosine_schedule(optimizer, config)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(config.device)
        labels    = batch["labels"].to(config.device)

        # --- Forward pass in BF16 ---
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            # Cross-entropy over next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        # --- Backward pass ---
        scaler.scale(loss).backward()

        # Gradient clipping: prevent exploding gradients
        # clip to max_norm=1.0 — standard for all LLM training
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)   # set_to_none is faster than zeros

        if step % config.log_every == 0:
            print(f"step={step} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
`;

const lrScheduleCode = `# unbox_platform/train/schedule.py
import math
from torch.optim.lr_scheduler import LambdaLR

def build_cosine_schedule(
    optimizer,
    config,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay to min_lr.

    Why warmup?
    - At step 0, weights are random and gradients are huge.
    - A large LR at the start causes divergence.
    - Warmup ramps LR from ~0 to max_lr over the first few hundred steps,
      letting the optimizer stabilize before committing to a direction.

    Why cosine decay?
    - Empirically outperforms linear decay on language models.
    - The slow tail end of training (low LR) lets the model 'polish'
      without overshooting the loss minimum.
    """
    warmup_steps = config.warmup_steps
    total_steps  = config.max_steps
    max_lr       = config.max_lr
    min_lr       = config.min_lr   # typically max_lr / 10

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps   # linear ramp

        # Cosine decay from max_lr to min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr / max_lr + (1 - min_lr / max_lr) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
`;

const checkpointCode = `# unbox_platform/train/checkpoint.py
import torch
from pathlib import Path

def save_checkpoint(
    model, optimizer, scheduler, scaler,
    step: int, loss: float, config, output_dir: str,
) -> None:
    """Save a full training checkpoint for resumability."""
    path = Path(output_dir) / f"step_{step:07d}.pt"
    torch.save({
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config,
    }, path)
    # Also keep a 'latest' symlink for easy resume
    latest = Path(output_dir) / "latest.pt"
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)


def load_checkpoint(path: str, model, optimizer, scheduler, scaler) -> int:
    """Resume training from a checkpoint, return the step number."""
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["step"]
`;

export function PretrainingPage() {
  return (
    <ChapterLayout
      num="03"
      title="Pre-training"
      subtitle="Feed the packed token sequences into a Transformer and predict the next token. This is where the model learns language, facts, and reasoning."
      color="text-brand-400"
      prev={{ path: "/tokenizer", label: "Tokenizer" }}
      next={{ path: "/sft", label: "SFT (coming soon)" }}
    >
      {/* Live animations */}
      <div className="grid sm:grid-cols-2 gap-4">
        <TransformerAnim />
        <TrainingLoopAnim />
      </div>

      <Section stepNum={1} title="The objective: next-token prediction">
        <p className="prose-custom text-base">
          Pre-training is conceptually simple: given a sequence of tokens, predict
          the next one. Do this for trillions of tokens from diverse text and the
          model learns grammar, facts, reasoning patterns, and much more.
        </p>

        <div className="card-glass p-5">
          <div className="text-sm font-semibold text-white mb-3">Teacher forcing</div>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-xs font-semibold text-gray-600 w-14 flex-shrink-0 mt-0.5">
                Input
              </span>
              <div className="flex flex-wrap gap-1.5">
                {["The", "cat", "sat", "on", "the"].map((tok, i) => (
                  <span key={i} className="px-2 py-1 rounded-lg bg-surface-700 text-gray-300 text-sm font-mono">
                    {tok}
                  </span>
                ))}
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-xs font-semibold text-gray-600 w-14 flex-shrink-0 mt-0.5">
                Target
              </span>
              <div className="flex flex-wrap gap-1.5">
                {["cat", "sat", "on", "the", "mat"].map((tok, i) => (
                  <span key={i} className="px-2 py-1 rounded-lg bg-brand-500/15 text-brand-300 border border-brand-500/25 text-sm font-mono">
                    {tok}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Input is shifted by 1. Every position predicts the next token in parallel during training.
          </p>
        </div>

        <Callout type="insight">
          One training example generates T prediction tasks simultaneously. A 1024-token
          sequence gives you 1023 (input, target) pairs in a single forward pass.
          This is why transformer training is so much more efficient than
          recurrent networks, which process one step at a time.
        </Callout>
      </Section>

      <Section stepNum={2} title="Model architecture: LLaMA-style Transformer">
        <p className="prose-custom text-base">
          The model uses a <strong>decoder-only Transformer</strong> with modern
          improvements over the original 2017 architecture: RMSNorm instead of LayerNorm,
          SwiGLU activation, Rotary Position Embeddings (RoPE), and Grouped Query
          Attention (GQA).
        </p>

        <div className="grid sm:grid-cols-2 gap-4 mb-4">
          {[
            {
              old: "LayerNorm",
              new_: "RMSNorm",
              reason: "Cheaper, same quality — removes mean subtraction step",
              color: "#0ea5e9",
            },
            {
              old: "Sinusoidal PE",
              new_: "RoPE",
              reason: "Relative positions via rotation — generalizes to longer contexts",
              color: "#10b981",
            },
            {
              old: "GELU",
              new_: "SwiGLU",
              reason: "SiLU × gate = smoother gradient, empirically better loss",
              color: "#a855f7",
            },
            {
              old: "MHA",
              new_: "GQA",
              reason: "Fewer KV heads → smaller KV cache → more tokens in memory",
              color: "#f59e0b",
            },
          ].map((item) => (
            <div key={item.old} className="card-glass p-4" style={{ borderColor: `${item.color}20` }}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-mono text-gray-600 line-through">{item.old}</span>
                <span className="text-xs text-gray-600">→</span>
                <span className="text-xs font-mono font-semibold" style={{ color: item.color }}>
                  {item.new_}
                </span>
              </div>
              <p className="text-xs text-gray-400">{item.reason}</p>
            </div>
          ))}
        </div>

        <CodeBlock
          code={modelConfigCode}
          filename="unbox_platform/model/config.py"
          highlights={[21, 22, 23, 24, 25]}
        />

        <CodeBlock
          code={modelArchCode}
          filename="unbox_platform/model/model.py"
          highlights={[48, 49, 50, 51, 55, 56, 57, 58]}
        />

        <Callout type="why">
          <strong>Why no bias terms?</strong> Linear layers in modern LLMs are
          typically bias-free. The bias saves ~0.1% parameters but adds noise to
          the gradient and doesn't help. LLaMA, Mistral, Qwen all use <code>bias=False</code>.
        </Callout>
      </Section>

      <Section stepNum={3} title="The training loop">
        <p className="prose-custom text-base">
          The training loop is the innermost hot path. Every line matters for
          both correctness and performance. Here are the six key operations
          in order: forward, loss, backward, grad clip, optimizer step, LR step.
        </p>

        <CodeBlock
          code={trainingLoopCode}
          filename="unbox_platform/train/pretrain.py"
          highlights={[17, 18, 22, 23, 24, 29, 30, 31, 33, 34, 35]}
        />

        <div className="grid sm:grid-cols-2 gap-4">
          <Callout type="tip">
            <strong>BF16 vs FP16:</strong> Use BF16 (bfloat16) if your GPU supports it
            (Ampere+). BF16 has the same exponent range as FP32 so it doesn't overflow;
            FP16 requires loss scaling. BF16 is strictly better for LLM training.
          </Callout>
          <Callout type="warning">
            <strong>Gradient accumulation:</strong> If your batch doesn't fit in GPU
            memory, accumulate gradients over N micro-batches before calling
            <code>optimizer.step()</code>. Divide the loss by N before each backward.
          </Callout>
        </div>
      </Section>

      <Section stepNum={4} title="Learning rate schedule">
        <p className="prose-custom text-base">
          The learning rate schedule is one of the most impactful hyperparameters.
          Too high and training diverges. Too low and you waste compute. The standard
          recipe for LLMs is: <strong>linear warmup → cosine decay → minimum LR</strong>.
        </p>

        <CodeBlock
          code={lrScheduleCode}
          filename="unbox_platform/train/schedule.py"
          highlights={[24, 25, 26, 27, 28, 29, 30, 31]}
        />

        {/* LR curve visualization */}
        <div className="card-glass p-5">
          <div className="text-sm font-semibold text-white mb-3">Learning rate curve</div>
          <svg viewBox="0 0 300 80" className="w-full h-24">
            <defs>
              <linearGradient id="lrGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.3" />
                <stop offset="15%" stopColor="#0ea5e9" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.1" />
              </linearGradient>
            </defs>
            {/* Warmup phase */}
            <line x1="0" y1="75" x2="45" y2="8" stroke="#0ea5e9" strokeWidth="2" strokeLinecap="round" />
            {/* Cosine decay */}
            <path
              d={`M 45 8 Q 100 8 150 30 Q 200 52 250 68 Q 270 73 300 74`}
              fill="none"
              stroke="#0ea5e9"
              strokeWidth="2"
            />
            {/* Area fill */}
            <path
              d={`M 0 75 L 45 8 Q 100 8 150 30 Q 200 52 250 68 Q 270 73 300 74 L 300 75 Z`}
              fill="url(#lrGrad)"
            />
            {/* Labels */}
            <text x="5" y="72" fill="#6b7280" fontSize="7">0</text>
            <text x="22" y="95" fill="#6b7280" fontSize="7">warmup</text>
            <text x="130" y="95" fill="#6b7280" fontSize="7">cosine decay</text>
            <text x="248" y="88" fill="#6b7280" fontSize="7">min_lr</text>
            <text x="1" y="12" fill="#0ea5e9" fontSize="7">max_lr</text>
          </svg>

          <div className="grid grid-cols-3 gap-3 mt-2">
            {[
              { key: "max_lr", val: "3e-4", note: "Peak LR" },
              { key: "warmup", val: "2000 steps", note: "~0.1% of training" },
              { key: "min_lr", val: "3e-5", note: "max_lr / 10" },
            ].map((item) => (
              <div key={item.key} className="text-center">
                <div className="font-mono text-xs text-brand-400">{item.val}</div>
                <div className="text-xs text-gray-600">{item.note}</div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      <Section stepNum={5} title="Checkpointing and resumability">
        <p className="prose-custom text-base">
          Pre-training runs for days or weeks. You <em>will</em> have hardware failures,
          preemptions, and experiments that need to be resumed. Save everything you need
          to resume exactly where you left off.
        </p>

        <CodeBlock
          code={checkpointCode}
          filename="unbox_platform/train/checkpoint.py"
          highlights={[10, 11, 12, 13, 14, 15]}
        />

        <Callout type="warning">
          Save <strong>optimizer state</strong>, not just model weights. AdamW's moment
          estimates (m, v) accumulate knowledge about the gradient history. Restoring
          only weights means the optimizer restarts cold — you'll see a temporary loss spike
          and 500-1000 steps of wasted compute.
        </Callout>

        <div className="grid sm:grid-cols-3 gap-4">
          {[
            { key: "Save frequency", val: "Every 1k steps", note: "balance overhead vs recovery cost" },
            { key: "Keep N ckpts", val: "Last 3–5", note: "delete old ones to save disk" },
            { key: "Upload to", val: "ModelScope / S3", note: "for cross-machine resume" },
          ].map((item) => (
            <div key={item.key} className="card-glass p-4">
              <div className="text-xs text-gray-500 mb-1">{item.key}</div>
              <div className="text-sm font-bold text-white">{item.val}</div>
              <div className="text-xs text-gray-600 mt-1">{item.note}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section stepNum={6} title="Run pre-training">
        <p className="prose-custom text-base">
          With data prepared (Chapter 01) and tokenizer trained (Chapter 02),
          you're ready to launch pre-training:
        </p>

        <CodeBlock
          language="bash"
          code={`# Single GPU
.venv/bin/python -m unbox_platform.train.pretrain \\
    --config configs/pretrain/760m.yaml

# Multi-GPU (8 GPUs, data parallel)
.venv/bin/torchrun --nproc_per_node=8 \\
    -m unbox_platform.train.pretrain \\
    --config configs/pretrain/760m.yaml

# Resume from checkpoint
.venv/bin/python -m unbox_platform.train.pretrain \\
    --config configs/pretrain/760m.yaml \\
    --resume checkpoints/pretrain/760m/latest.pt

# Expected output:
# step=0     loss=10.821 lr=0.00e+00
# step=100   loss=7.432  lr=1.50e-05
# step=500   loss=5.891  lr=7.50e-05
# step=1000  loss=4.234  lr=1.50e-04
# step=5000  loss=3.102  lr=1.41e-04
# step=10000 loss=2.876  lr=1.23e-04
# ...
`}
          filename="terminal"
        />

        <Callout type="insight">
          A <strong>760M parameter model</strong> trained on 15B tokens (the Chinchilla
          optimal point for this size) takes roughly 3 days on 8× A100-80GB GPUs.
          The loss should drop from ~10 (random) to ~2.3–2.5 (competent English text
          generation) over this run.
        </Callout>

        <div className="card-glass p-5 gradient-border">
          <div className="text-sm font-semibold text-white mb-3">Chinchilla scaling law</div>
          <p className="text-sm text-gray-400 leading-relaxed mb-3">
            The Chinchilla paper (Hoffmann et al., 2022) showed that the optimal
            compute allocation trains a smaller model on more data: roughly{" "}
            <strong className="text-white">20 tokens per parameter</strong>. 760M params
            → 15B tokens is Chinchilla-optimal.
          </p>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Model params", val: "760M" },
              { label: "Optimal tokens", val: "15B" },
              { label: "Ratio", val: "20×" },
            ].map((item) => (
              <div key={item.label} className="text-center">
                <div className="text-xl font-bold gradient-text">{item.val}</div>
                <div className="text-xs text-gray-500">{item.label}</div>
              </div>
            ))}
          </div>
        </div>
      </Section>
    </ChapterLayout>
  );
}
