import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { DPOAnim } from "../components/animations/DPOAnim";

const dpoIntuitionCode = `# The DPO objective — intuition in code
# For each (prompt, chosen, rejected) triple:

log_prob_chosen   = model_log_prob(prompt + chosen)
log_prob_rejected = model_log_prob(prompt + rejected)

log_prob_chosen_ref   = ref_model_log_prob(prompt + chosen)
log_prob_rejected_ref = ref_model_log_prob(prompt + rejected)

# The "implicit reward" for a response r given prompt x:
# r(x, r) = β * (log π(r|x) - log π_ref(r|x))
# where π is the policy (model being trained) and π_ref is the frozen reference.

implicit_reward_chosen   = beta * (log_prob_chosen   - log_prob_chosen_ref)
implicit_reward_rejected = beta * (log_prob_rejected - log_prob_rejected_ref)

# Loss: push reward(chosen) > reward(rejected) via Bradley-Terry model
loss = -F.logsigmoid(implicit_reward_chosen - implicit_reward_rejected)

# What this does:
# - Increases log P(chosen) relative to reference
# - Decreases log P(rejected) relative to reference
# - The reference model acts as a KL constraint (prevents reward hacking)
`;

const dpoConfigCode = `# unbox_platform/rl/dpo/train.py — DPOTrainConfig
@dataclass
class DPOTrainConfig:
    # Start from the SFT checkpoint (not the pretrain .pt)
    model_path: str = "checkpoints/sft/basic/checkpoint-3000"
    output_dir: str = "checkpoints/rl/dpo"

    # Dataset — chosen/rejected conversation pairs
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split:  str = "train_prefs"
    eval_split:   str = "test_prefs"

    # Key DPO hyperparameters
    beta: float      = 0.1    # KL penalty coefficient — higher = stay closer to SFT
    loss_type: str   = "sigmoid"  # original DPO; "hinge" or "ipo" are alternatives
    max_length: int  = 2048

    # Training — much lower LR than SFT
    learning_rate: float = 5e-7   # 40× lower than SFT's 2e-5
    num_epochs:    int   = 1
    batch_size:    int   = 2
    grad_accumulation_steps: int = 4
`;

const datasetPrepCode = `# unbox_platform/rl/dpo/train.py — dataset preparation
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

# CRITICAL: drop the string "prompt" column.
#
# ultrafeedback_binarized has both:
#   "prompt"   — a plain string (the user question)
#   "chosen"   — list of message dicts [{"role":"user",...}, {"role":"assistant",...}]
#   "rejected" — list of message dicts
#
# TRL's DPOTrainer has two code paths:
#   1. If "prompt" is a string → uses it directly as the prompt text
#   2. If no "prompt" → calls extract_prompt() to split chosen/rejected on shared prefix
#
# Path 1 conflicts with the conversational format in chosen/rejected. The result
# is that TRL tries to tokenize both the string prompt and the list conversation
# independently, causing shape mismatches or silently wrong training targets.
#
# Solution: drop the string prompt and let TRL use extract_prompt.
drop_cols = [c for c in ["prompt", "prompt_id", "messages", "score_chosen", "score_rejected"]
             if c in dataset.column_names]
dataset = dataset.remove_columns(drop_cols)

# After dropping, the dataset has only: "chosen" and "rejected"
# Both are lists of message dicts. TRL finds the shared prefix (the user turn)
# and uses it as the prompt, with the differing assistant turn as the response.
`;

const dpoTrainCode = `# unbox_platform/rl/dpo/train.py — main training
def main() -> None:
    cfg = DPOTrainConfig(...)

    model = UnboxForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16)

    # Load the reference model explicitly from the SAME SFT checkpoint.
    # The reference model is frozen throughout training — it provides the KL baseline.
    # TRL can auto-create it, but only for models registered in HF's model hub.
    # For our custom UnboxForCausalLM, we load it manually.
    ref_model = UnboxForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
        learning_rate=cfg.learning_rate,
        bf16=True,
        lr_scheduler_type="cosine",
        # ... standard training args ...
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,   # frozen reference — NOT updated during training
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
`;

const runDPOCode = `# Step 1: Establish SFT baseline metrics (Chapter 04)
.venv/bin/python -m unbox_platform.eval.dpo_eval \\
    --checkpoint checkpoints/sft/basic/checkpoint-3000 \\
    --label "SFT baseline" \\
    --output reports/dpo_experiment.md
# Win rate: 58.4%  Margin: +0.0423

# Step 2: Run DPO training
.venv/bin/python -m unbox_platform.rl.dpo.train \\
    --config configs/rl/dpo.yaml
# {'loss': 0.693, 'rewards/chosen': 0.021, 'rewards/rejected': -0.018, ...}
# {'loss': 0.681, 'rewards/chosen': 0.087, 'rewards/rejected': -0.072, ...}
# ...
# DPO complete. Model saved to checkpoints/rl/dpo

# Step 3: Evaluate improvement
.venv/bin/python -m unbox_platform.eval.dpo_eval \\
    --checkpoint checkpoints/rl/dpo \\
    --label "After DPO" \\
    --output reports/dpo_experiment.md
# Win rate: 73.1%  Margin: +0.2847

# Step 4: View report
cat reports/dpo_experiment.md
`;

export function DPOPage() {
  return (
    <ChapterLayout
      num="06"
      title="DPO"
      subtitle="Direct Preference Optimization aligns the model with human preferences using pairs of chosen and rejected responses — no reward model required."
      color="text-pink-400"
      prev={{ path: "/sft", label: "SFT" }}
      next={{ path: "/inference", label: "Inference (coming soon)" }}
    >
      <DPOAnim />

      <Section stepNum={1} title="Why not RLHF?">
        <p className="prose-custom text-base">
          Classic RLHF (Reinforcement Learning from Human Feedback) requires three
          separate stages: (1) train a reward model from preference pairs, (2) run
          PPO to optimize the policy against the reward model, (3) maintain a KL
          penalty to prevent the model from gaming the reward. Each stage has its
          own hyperparameters and failure modes.
        </p>
        <p className="prose-custom text-base">
          <strong>DPO</strong> (Rafailov et al., 2023) eliminates the reward model
          entirely. It shows that the optimal RLHF policy can be expressed as a
          closed-form update directly on the preference pairs — making the problem
          a supervised binary classification task.
        </p>

        <div className="grid sm:grid-cols-2 gap-4">
          <div className="card-glass p-4 border border-red-500/15">
            <div className="text-sm font-semibold text-red-400 mb-2">RLHF (PPO)</div>
            <ul className="space-y-1 text-xs text-gray-400">
              <li>→ Train a separate reward model</li>
              <li>→ Run PPO (complex, unstable)</li>
              <li>→ Tune KL coefficient carefully</li>
              <li>→ Monitor reward hacking</li>
              <li>→ Needs online rollouts (expensive)</li>
            </ul>
          </div>
          <div className="card-glass p-4 border border-emerald-500/15">
            <div className="text-sm font-semibold text-emerald-400 mb-2">DPO</div>
            <ul className="space-y-1 text-xs text-gray-400">
              <li>✓ No reward model needed</li>
              <li>✓ Simple binary cross-entropy loss</li>
              <li>✓ Stable training (like SFT)</li>
              <li>✓ One β hyperparameter</li>
              <li>✓ Offline (uses stored preferences)</li>
            </ul>
          </div>
        </div>
      </Section>

      <Section stepNum={2} title="The DPO loss: intuition">
        <p className="prose-custom text-base">
          DPO derives from a reparameterization of the RLHF objective. The key insight
          is that any optimal RLHF policy can be written in terms of the reference
          model's log-probabilities. This eliminates the reward model as a separate object.
        </p>

        <div className="card-glass p-5 mb-4">
          <div className="text-sm font-semibold text-white mb-3">Implicit reward</div>
          <div className="font-mono text-sm text-center py-2 leading-loose">
            <span className="text-amber-400">r(x, y)</span>
            <span className="text-gray-400"> = β · (</span>
            <span className="text-brand-300">log π(y|x)</span>
            <span className="text-gray-400"> − </span>
            <span className="text-gray-500">log π_ref(y|x)</span>
            <span className="text-gray-400">)</span>
          </div>
          <p className="text-xs text-gray-500 text-center mt-2">
            β controls how far the policy can diverge from the reference (KL penalty strength)
          </p>
        </div>

        <div className="card-glass p-5 mb-4">
          <div className="text-sm font-semibold text-white mb-3">DPO loss</div>
          <div className="font-mono text-sm text-center py-2 leading-loose">
            <span className="text-amber-400">L_DPO</span>
            <span className="text-gray-400"> = −log σ(</span>
            <span className="text-emerald-400">r(x, y_w)</span>
            <span className="text-gray-400"> − </span>
            <span className="text-red-400">r(x, y_l)</span>
            <span className="text-gray-400">)</span>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-3 text-xs text-gray-500">
            <div><span className="text-emerald-400">y_w</span> = chosen (preferred) response</div>
            <div><span className="text-red-400">y_l</span> = rejected (dispreferred) response</div>
          </div>
        </div>

        <CodeBlock
          code={dpoIntuitionCode}
          filename="DPO objective — annotated"
          highlights={[19, 20, 21, 22, 23, 24]}
        />

        <Callout type="insight">
          The reference model is a <strong>KL constraint</strong>, not a performance
          baseline. It prevents the model from degenerate solutions like assigning all
          probability mass to one-word responses ("Yes.") that technically win every
          comparison but are useless in practice. Higher β = stay closer to the SFT
          model; lower β = more aggressive preference learning.
        </Callout>
      </Section>

      <Section stepNum={3} title="Dataset format: preference pairs">
        <p className="prose-custom text-base">
          DPO requires a dataset of <strong>(prompt, chosen, rejected)</strong> triples.
          We use <code>HuggingFaceH4/ultrafeedback_binarized</code> — 60K examples
          where GPT-4 rated responses from four different models and the best was
          labeled "chosen", worst "rejected".
        </p>

        <div className="card-glass p-5 mb-4">
          <div className="text-sm font-semibold text-white mb-3">Preference pair structure</div>
          <div className="space-y-3">
            {[
              { label: "prompt", content: '[{"role": "user", "content": "Explain gradient descent."}]', color: "#6366f1" },
              { label: "chosen", content: '[{"role": "user", ...}, {"role": "assistant", "content": "Gradient descent iteratively minimizes loss by..."}]', color: "#10b981" },
              { label: "rejected", content: '[{"role": "user", ...}, {"role": "assistant", "content": "It makes the model learn better."}]', color: "#ef4444" },
            ].map((item) => (
              <div key={item.label} className="rounded-lg p-3 border" style={{ background: `${item.color}08`, borderColor: `${item.color}25` }}>
                <span className="text-xs font-mono font-semibold" style={{ color: item.color }}>{item.label}</span>
                <p className="text-xs text-gray-400 font-mono mt-1 break-all">{item.content}</p>
              </div>
            ))}
          </div>
        </div>

        <Callout type="warning" title="Critical: drop the string 'prompt' column">
          <code>ultrafeedback_binarized</code> has both a string <code>"prompt"</code>
          column and list-format <code>"chosen"</code>/<code>"rejected"</code> columns.
          TRL's <code>DPOTrainer</code> has two code paths depending on whether
          a string <code>"prompt"</code> exists. Leaving it in causes shape mismatches
          or silently wrong training targets. Always drop it and let TRL's{" "}
          <code>extract_prompt()</code> derive the prompt from the shared prefix.
        </Callout>

        <CodeBlock
          code={datasetPrepCode}
          filename="unbox_platform/rl/dpo/train.py — dataset prep"
          highlights={[19, 20, 21, 22, 23]}
        />
      </Section>

      <Section stepNum={4} title="DPOTrainConfig and key hyperparameters">
        <CodeBlock
          code={dpoConfigCode}
          filename="unbox_platform/rl/dpo/train.py — DPOTrainConfig"
          highlights={[13, 14, 15, 18, 19]}
        />

        <div className="grid sm:grid-cols-2 gap-4">
          {[
            { key: "β (beta)", val: "0.1", note: "Higher → stay closer to SFT. Too low → reward hack. Too high → no change.", color: "#f59e0b" },
            { key: "learning_rate", val: "5e-7", note: "~40× lower than SFT. DPO is a small adjustment; large LR destroys the SFT alignment.", color: "#0ea5e9" },
            { key: "loss_type", val: "sigmoid", note: "The original DPO paper. 'ipo' and 'hinge' are alternatives with different theoretical properties.", color: "#a855f7" },
            { key: "num_epochs", val: "1–2", note: "Overfit risk is real — more epochs can degrade helpfulness even as win rate climbs.", color: "#10b981" },
          ].map((item) => (
            <div key={item.key} className="card-glass p-4" style={{ borderColor: `${item.color}20` }}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-xs" style={{ color: item.color }}>{item.key}</span>
                <span className="font-mono text-sm font-bold text-white">{item.val}</span>
              </div>
              <p className="text-xs text-gray-400 leading-relaxed">{item.note}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section stepNum={5} title="Training with DPOTrainer">
        <p className="prose-custom text-base">
          One critical detail: we load the reference model <strong>explicitly</strong>
          from the same SFT checkpoint rather than letting TRL auto-create it.
          TRL's auto-creation calls <code>AutoModelForCausalLM.from_pretrained()</code>
          internally — which works fine for standard HF models but can miss our
          custom class registration. Loading manually is more reliable.
        </p>

        <CodeBlock
          code={dpoTrainCode}
          filename="unbox_platform/rl/dpo/train.py"
          highlights={[10, 11, 12, 13, 30, 31, 32]}
        />

        <Callout type="tip">
          During DPO training, TRL logs <code>rewards/chosen</code> and{" "}
          <code>rewards/rejected</code> per step. A healthy run shows the gap between
          them widening over time — chosen rewards increasing and rejected rewards
          decreasing relative to the reference. If both move together in the same
          direction, β is too low or the LR is too high.
        </Callout>
      </Section>

      <Section stepNum={6} title="Evaluate before and after">
        <p className="prose-custom text-base">
          Run <code>eval/dpo_eval.py</code> (Chapter 04) before and after training.
          The report captures both the quantitative win-rate improvement and qualitative
          response comparisons on the same fixed prompts, making the delta legible.
        </p>

        <CodeBlock
          language="bash"
          code={runDPOCode}
          filename="terminal"
        />

        <div className="card-glass p-5 gradient-border">
          <div className="text-sm font-semibold text-white mb-3">Expected results</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left py-2 text-gray-500 font-medium">Checkpoint</th>
                  <th className="text-left py-2 text-gray-500 font-medium">Win rate</th>
                  <th className="text-left py-2 text-gray-500 font-medium">Margin</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/3">
                  <td className="py-2 text-gray-300 font-mono text-xs">SFT baseline</td>
                  <td className="py-2 text-brand-300 font-mono text-xs">~58%</td>
                  <td className="py-2 text-gray-400 font-mono text-xs">+0.04</td>
                </tr>
                <tr>
                  <td className="py-2 text-gray-300 font-mono text-xs">After DPO (β=0.1)</td>
                  <td className="py-2 text-emerald-400 font-mono text-xs">~73%</td>
                  <td className="py-2 text-emerald-400 font-mono text-xs">+0.28</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Evaluated on 500 held-out pairs from <code>ultrafeedback_binarized/test_prefs</code>.
            Actual numbers vary with training data and checkpoint quality.
          </p>
        </div>

        <Callout type="insight">
          Win rate can look good even when the model has gotten worse at helpfulness.
          Always pair quantitative win-rate with qualitative sampling on your own
          prompts. A model that learned to write very long, verbose responses can
          achieve high win-rate by length bias alone.
        </Callout>
      </Section>
    </ChapterLayout>
  );
}
