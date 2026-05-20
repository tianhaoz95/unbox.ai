import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { ChatTemplateAnim } from "../components/animations/ChatTemplateAnim";

const hfAdapterCode = `# unbox_platform/model/hf_adapter.py
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.generation import GenerationMixin

class UnboxConfig(PretrainedConfig):
    """
    PretrainedConfig wrapper around ModelConfig.
    Inheriting from PretrainedConfig means TRL, PEFT, and the HF Trainer
    can all consume this model without modification.
    """
    model_type = "unbox"

    def __init__(
        self,
        hidden_size: int = 1792,
        num_layers: int = 24,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        ffn_intermediate_size: int = 4864,
        vocab_size: int = 32768,
        max_seq_len: int = 2048,
        **kwargs,
    ) -> None:
        # ... store attrs ...
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.num_hidden_layers = self.num_layers  # HF internals expect this


class UnboxForCausalLM(PreTrainedModel, GenerationMixin):
    """HF-compatible causal LM. Wraps bare Transformer for TRL/PEFT/eval."""

    config_class = UnboxConfig
    _supports_cache_class = False      # KV cache not yet implemented
    _tied_weights_keys = {"model.lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: UnboxConfig) -> None:
        super().__init__(config)
        self.model = Transformer(config.to_model_config())
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,   # accepted for compatibility, but ignored
        labels=None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # attention_mask: ignored — causal masking is built into the model via
        #   scaled_dot_product_attention(is_causal=True)
        # use_cache: ignored — pass use_cache=False to generate() explicitly
        logits, loss = self.model(input_ids, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @classmethod
    def from_unbox_checkpoint(cls, checkpoint_path, config, device="cpu"):
        """Load a .pt pretrain checkpoint, adding the 'model.' prefix needed by HF."""
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        prefixed = {"model." + k: v for k, v in state["model"].items()}
        model = cls(config)
        model.load_state_dict(prefixed)
        return model

# Register so AutoModelForCausalLM.from_pretrained() can load our checkpoints
AutoConfig.register("unbox", UnboxConfig)
AutoModelForCausalLM.register(UnboxConfig, UnboxForCausalLM)
`;

const sftConfigCode = `# unbox_platform/sft/config.py
@dataclass
class SFTTrainConfig:
    # Paths
    checkpoint_path: str = ""           # pretrain .pt checkpoint to start from
    tokenizer_path:  str = "checkpoints/tokenizer"
    output_dir:      str = "checkpoints/sft"

    # Dataset — must have a "messages" column
    dataset_name:       str = "HuggingFaceH4/ultrachat_200k"
    dataset_split:      str = "train_sft"
    eval_dataset_split: str = "test_sft"
    max_samples:        int = -1        # -1 = use all

    # Training
    num_epochs:              int   = 1
    batch_size:              int   = 2
    grad_accumulation_steps: int   = 4
    max_lr:                  float = 2e-5   # much lower than pretrain (3e-4)
    warmup_steps:            int   = 100
    weight_decay:            float = 0.01
    grad_clip:               float = 1.0
    dtype:                   str   = "bfloat16"
    max_seq_len:             int   = 2048

    # SFTTrainer-specific
    packing: bool = False   # can set True for efficiency, but lose document boundaries
`;

const sftTrainCode = `# unbox_platform/sft/train.py
def main() -> None:
    cfg = SFTTrainConfig(...)

    # Step 1: Load pretrained base model via HF adapter
    model = UnboxForCausalLM.from_unbox_checkpoint(
        cfg.checkpoint_path,
        UnboxConfig(),
        device="cpu",
    )

    # Step 2: Load HF-compatible tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # SFTTrainer needs pad_token

    # Step 3: Load dataset — keep only the "messages" column.
    # The "messages" column triggers TRL's chat-template path.
    # If a "prompt" column is also present, TRL uses a different (worse) path.
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    dataset = dataset.select_columns(["messages"])

    # Step 4: Configure SFTTrainer
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.max_lr,
        bf16=True,
        lr_scheduler_type="cosine",
        max_length=cfg.max_seq_len,
        packing=cfg.packing,

        # completion_only_loss=True: mask user turns so gradient only flows
        # through the assistant response tokens.
        # Without this the model learns to predict user messages too, and
        # starts generating fake "User: ..." turns during inference.
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
`;

const datasetFormatCode = `# The ultrachat_200k "messages" column format
# Each example has a list of dicts with "role" and "content"
example = {
    "messages": [
        {"role": "user",      "content": "What is backpropagation?"},
        {"role": "assistant", "content": "Backpropagation is the algorithm..."},
        {"role": "user",      "content": "Can you give a code example?"},
        {"role": "assistant", "content": "Sure! Here's a simple example in PyTorch..."},
    ]
}

# SFTTrainer calls tokenizer.apply_chat_template() internally,
# producing (for ChatML format):
# <|im_start|>user
# What is backpropagation?<|im_end|>
# <|im_start|>assistant
# Backpropagation is the algorithm...<|im_end|>
# ...

# With completion_only_loss=True, labels look like:
# [-100, -100, ..., -100,   token_ids_of_assistant_response...,   -100, -100, ...]
#  ^^^^ user turn masked ^^^^                                       ^^^ user ^^^
`;

const runSFTCode = `# Run SFT from a pretrain checkpoint
.venv/bin/python -m unbox_platform.sft.train --config configs/sft/basic.yaml

# configs/sft/basic.yaml:
# checkpoint_path: checkpoints/pretrain/760m/latest.pt
# tokenizer_path:  checkpoints/tokenizer
# output_dir:      checkpoints/sft/basic
# dataset_name:    HuggingFaceH4/ultrachat_200k
# num_epochs:      1
# batch_size:      2
# grad_accumulation_steps: 4
# max_lr:          2e-5
# max_seq_len:     2048
# completion_only_loss: true

# Expected output:
# {'loss': 1.821, 'grad_norm': 0.52, 'learning_rate': 2e-05, 'epoch': 0.01}
# {'loss': 1.543, 'grad_norm': 0.41, 'learning_rate': 1.9e-05, 'epoch': 0.05}
# ...
# SFT complete. Model saved to checkpoints/sft/basic

# Verify — chat sample from the SFT checkpoint
.venv/bin/python -m unbox_platform.eval.chat_sample \\
    --checkpoint checkpoints/sft/basic/checkpoint-3000 \\
    --prompt "Explain the transformer attention mechanism"
`;

export function SFTPage() {
  return (
    <ChapterLayout
      num="05"
      title="Supervised Fine-Tuning"
      subtitle="A pretrained base model predicts text. SFT turns it into an assistant that follows instructions. The key is the dataset format and how you compute the loss."
      color="text-amber-400"
      prev={{ path: "/eval", label: "Evaluation" }}
      next={{ path: "/dpo", label: "DPO" }}
    >
      <ChatTemplateAnim />

      <Section stepNum={1} title="The HuggingFace adapter: why it exists">
        <p className="prose-custom text-base">
          Our pretrained <code>Transformer</code> is a clean PyTorch module — but TRL's{" "}
          <code>SFTTrainer</code> expects a HuggingFace <code>PreTrainedModel</code>.
          The <code>UnboxForCausalLM</code> adapter bridges this gap: it wraps the
          existing model with no architectural changes, just the HF interface layer.
        </p>
        <p className="prose-custom text-base">
          This means we get TRL, PEFT/LoRA, and the HF ecosystem for free without
          ever touching the core model code — exactly the separation of concerns
          the architecture is designed for.
        </p>

        <Callout type="insight">
          <code>UnboxForCausalLM</code> inherits from both <code>PreTrainedModel</code>{" "}
          and <code>GenerationMixin</code>. <code>GenerationMixin</code> provides
          the full <code>model.generate()</code> loop — beam search, sampling, stopping
          criteria — for free. We just need a correct <code>forward()</code>.
        </Callout>

        <CodeBlock
          code={hfAdapterCode}
          filename="unbox_platform/model/hf_adapter.py"
          highlights={[51, 52, 53, 54, 56, 57, 58]}
        />

        <Callout type="warning">
          The adapter accepts <code>attention_mask</code> and <code>use_cache</code>
          for HF API compatibility but neither is implemented. Always pass{" "}
          <code>use_cache=False</code> to <code>generate()</code> — see Eval chapter
          (Chapter 04) for the exact pitfall.
        </Callout>
      </Section>

      <Section stepNum={2} title="Chat templates: the message → token mapping">
        <p className="prose-custom text-base">
          SFT training data is a list of <code>{`{"role": ..., "content": ...}`}</code>
          message dicts. A <strong>chat template</strong> converts this structure into
          a flat string that the tokenizer can process. The most common format today is
          ChatML, used by Qwen, Mistral-Instruct, and many others.
        </p>

        <CodeBlock
          code={datasetFormatCode}
          filename="dataset format → template → labels"
          highlights={[23, 24, 25, 26]}
        />

        <div className="card-glass p-5">
          <div className="text-sm font-semibold text-white mb-3">ChatML format</div>
          <div className="font-mono text-xs leading-loose text-gray-300 bg-surface-700/50 rounded-lg p-3">
            <span className="text-indigo-400">{"<|im_start|>system\n"}</span>
            <span className="text-gray-500">{"You are a helpful assistant.<|im_end|>\n"}</span>
            <span className="text-brand-400">{"<|im_start|>user\n"}</span>
            <span className="text-gray-500">{"What is backpropagation?<|im_end|>\n"}</span>
            <span className="text-emerald-400">{"<|im_start|>assistant\n"}</span>
            <span className="text-white">{"Backpropagation is the algorithm..."}</span>
            <span className="text-gray-500">{"<|im_end|>"}</span>
          </div>
          <div className="flex gap-4 mt-3 text-xs text-gray-500">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500/70 inline-block" /> masked (labels = -100)</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500/70 inline-block" /> loss computed</span>
          </div>
        </div>

        <Callout type="why">
          <strong>Why mask user turns?</strong> Without <code>completion_only_loss=True</code>,
          the model learns to predict the user's message too. During inference it starts
          generating <em>"User: ..."</em> continuation instead of staying in the assistant
          role. Masking forces all gradient signal through the assistant response only.
        </Callout>
      </Section>

      <Section stepNum={3} title="SFTTrainConfig: the hyperparameters">
        <p className="prose-custom text-base">
          SFT uses a much lower learning rate than pretraining — typically{" "}
          <strong>1e-5 to 5e-5</strong> vs. 3e-4 for pretraining. The model is
          already well-initialized; we're nudging it toward instruction following,
          not learning language from scratch.
        </p>

        <CodeBlock
          code={sftConfigCode}
          filename="unbox_platform/sft/config.py"
          highlights={[15, 16, 17, 18, 19, 30]}
        />

        <div className="grid sm:grid-cols-2 gap-4">
          {[
            { key: "max_lr", val: "2e-5", note: "10× lower than pretrain — preserves base knowledge", color: "#0ea5e9" },
            { key: "num_epochs", val: "1–3", note: "more epochs → overfitting on small datasets", color: "#10b981" },
            { key: "packing", val: "False", note: "True = more efficient but loses conversation boundaries", color: "#f59e0b" },
            { key: "max_seq_len", val: "2048", note: "cap to avoid OOM on multi-turn conversations", color: "#a855f7" },
          ].map((item) => (
            <div key={item.key} className="card-glass p-4" style={{ borderColor: `${item.color}20` }}>
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-xs" style={{ color: item.color }}>{item.key}</span>
                <span className="font-mono text-sm font-bold text-white">{item.val}</span>
              </div>
              <p className="text-xs text-gray-400">{item.note}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section stepNum={4} title="Training with SFTTrainer">
        <p className="prose-custom text-base">
          TRL's <code>SFTTrainer</code> handles the full training loop: chat template
          application, tokenization, masking, gradient accumulation, evaluation,
          and checkpoint saving. The dataset must have a <code>"messages"</code> column
          — if it also has a <code>"prompt"</code> column TRL takes a different (and
          worse) code path, so we drop everything except <code>"messages"</code>.
        </p>

        <CodeBlock
          code={sftTrainCode}
          filename="unbox_platform/sft/train.py"
          highlights={[28, 29, 30, 36, 37, 38, 39, 40, 41]}
        />

        <Callout type="tip">
          SFT training is usually only <strong>1 epoch</strong> over the dataset.
          More epochs produce diminishing returns on instruction following and
          increase the risk of overfit (the model starts memorizing specific responses).
          If you have &lt;50K examples, consider 2–3 epochs with early stopping.
        </Callout>
      </Section>

      <Section stepNum={5} title="Run it">
        <CodeBlock
          language="bash"
          code={runSFTCode}
          filename="terminal"
        />

        <div className="grid sm:grid-cols-3 gap-4">
          {[
            { label: "Dataset size", val: "200K turns", note: "ultrachat_200k" },
            { label: "Training time", val: "~4 hours", note: "on 2× A100-80GB" },
            { label: "Expected SFT loss", val: "~1.4–1.6", note: "after 1 epoch" },
          ].map((item) => (
            <div key={item.label} className="card-glass p-4 text-center">
              <div className="text-xl font-bold gradient-text mb-0.5">{item.val}</div>
              <div className="text-xs text-gray-400">{item.label}</div>
              <div className="text-xs text-gray-600 mt-0.5">{item.note}</div>
            </div>
          ))}
        </div>

        <Callout type="insight">
          The SFT checkpoint is saved in HuggingFace format (not <code>.pt</code>).
          That means <code>AutoTokenizer.from_pretrained()</code> and{" "}
          <code>UnboxForCausalLM.from_pretrained()</code> both work directly on
          the output directory. This is the checkpoint you hand to DPO next.
        </Callout>
      </Section>
    </ChapterLayout>
  );
}
