import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { PerplexityAnim } from "../components/animations/PerplexityAnim";
import { useLanguage } from "../contexts/LanguageContext";

const perplexityCode = `# unbox_platform/eval/perplexity.py
@torch.no_grad()
def compute_perplexity(
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    data_config: DataConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 4,
    max_batches: int | None = None,
) -> float:
    tokenizer = Tokenizer(tokenizer_path)

    model = Transformer(model_config)
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model = model.to(device)
    model.eval()

    eval_loader = build_dataloader(
        tokenizer, data_config, split="eval", batch_size=batch_size
    )

    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(eval_loader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(input_ids, labels)

        # loss is mean over non-ignored tokens; recover sum for accurate aggregation
        n_tokens = (labels != -100).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))   # cap at 20 to avoid overflow on early ckpts
`;

const sampleCode = `# unbox_platform/eval/sample.py
def sample(
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    tokenizer = Tokenizer(tokenizer_path)

    model = Transformer(model_config)
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model = model.to(device).to(dtype)
    model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,   # 0.0 = greedy; 1.0 = full distribution
        top_k=top_k,               # keep only top-k logits before sampling
        top_p=top_p,               # nucleus sampling: smallest set summing to p
        eos_id=tokenizer.eos_id,
    )

    # Slice off the prompt tokens — return only the generated continuation
    generated = output_ids[0, len(input_ids):].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)
`;

const chatSampleCode = `# unbox_platform/eval/chat_sample.py
def chat_sample(
    checkpoint: str | Path,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.0,   # default: greedy (deterministic)
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = UnboxForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # PITFALL: add_special_tokens=False is required here.
    # The chat template already contains all special tokens (<|im_start|> etc.).
    # If you let the tokenizer add its own, a spurious <|eos|> is appended at
    # the end, causing the model to generate the next "user" turn header instead
    # of an actual response.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            use_cache=False,   # KV cache not yet implemented in UnboxForCausalLM
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input tokens from output; decode only the generated response
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
`;

const dpoEvalCode = `# unbox_platform/eval/dpo_eval.py
def _completion_log_prob(
    model: UnboxForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: list[dict],
    device: str,
) -> float:
    """Mean per-token log P(assistant_response | prompt).

    Builds two sequences:
      prompt_ids  — user turns + generation prompt header
      full_ids    — prompt_ids + assistant response + end-of-turn

    Then masks all prompt tokens with -100 so cross-entropy only fires
    on the assistant response tokens. Returns negative of mean loss.
    """
    user_turns  = [m for m in conversation if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(user_turns, tokenize=False,
                                                add_generation_prompt=True)
    full_text   = tokenizer.apply_chat_template(conversation, tokenize=False,
                                                add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                           return_tensors="pt")["input_ids"][0]
    full_ids   = tokenizer(full_text,   add_special_tokens=False,
                           return_tensors="pt")["input_ids"][0]

    labels = full_ids.clone()
    labels[:len(prompt_ids)] = -100   # mask prompt; only score the response

    with torch.no_grad():
        out = model(
            input_ids=full_ids.unsqueeze(0).to(device),
            labels=labels.unsqueeze(0).to(device),
        )
    return -out.loss.item()   # negate NLL → mean log prob


def run_eval(checkpoint: str, n_pairs: int = 500, device: str = "cuda") -> dict:
    model    = UnboxForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = model.to(device).eval()

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    dataset = dataset.select(range(min(n_pairs, len(dataset))))

    wins, margins = 0, []
    for example in dataset:
        lp_chosen   = _completion_log_prob(model, tokenizer, example["chosen"],   device)
        lp_rejected = _completion_log_prob(model, tokenizer, example["rejected"], device)
        if lp_chosen > lp_rejected:
            wins += 1
        margins.append(lp_chosen - lp_rejected)

    return {
        "win_rate": wins / len(margins),
        "avg_margin": sum(margins) / len(margins),
    }
`;

const runEvalCode = `# Run perplexity on the pretrained checkpoint
.venv/bin/python -m unbox_platform.eval.perplexity \\
    --checkpoint checkpoints/pretrain/760m/latest.pt \\
    --tokenizer  checkpoints/tokenizer \\
    --config     configs/pretrain/760m.yaml \\
    --max-batches 200

# Output:
# Perplexity: 17.84

# Qualitative sampling from pretrain checkpoint
.venv/bin/python -m unbox_platform.eval.sample \\
    --checkpoint checkpoints/pretrain/760m/latest.pt \\
    --tokenizer  checkpoints/tokenizer \\
    --config     configs/pretrain/760m.yaml \\
    --prompt "The theory of relativity states that" \\
    --temperature 0.8 --top-k 50 --top-p 0.9

# Output:
# The theory of relativity states that the laws of physics are the same
# for all observers in uniform motion, and that the speed of light...

# Chat sampling from SFT checkpoint
.venv/bin/python -m unbox_platform.eval.chat_sample \\
    --checkpoint checkpoints/sft/basic/checkpoint-3000 \\
    --prompt "What is backpropagation?"

# DPO preference evaluation — run before AND after DPO to see the delta
.venv/bin/python -m unbox_platform.eval.dpo_eval \\
    --checkpoint checkpoints/sft/basic/checkpoint-3000 \\
    --label "SFT baseline" \\
    --output reports/dpo_experiment.md

.venv/bin/python -m unbox_platform.eval.dpo_eval \\
    --checkpoint checkpoints/rl/dpo \\
    --label "After DPO" \\
    --output reports/dpo_experiment.md
`;

const samplingCode = `# Three sampling strategies — all built into model.generate()

# 1. Greedy (temperature=0): always pick the most probable next token.
#    Fast, deterministic, but repetitive for long outputs.
output = model.generate(input_ids, temperature=0.0, max_new_tokens=100)

# 2. Top-k sampling: sample from the k highest probability tokens only.
#    Prevents the model from sampling very low-probability garbage.
output = model.generate(input_ids, temperature=0.8, top_k=50, max_new_tokens=100)

# 3. Nucleus (top-p) sampling: sample from the smallest set of tokens
#    whose cumulative probability >= p. Adapts vocabulary size dynamically.
output = model.generate(input_ids, temperature=0.8, top_p=0.9, max_new_tokens=100)

# In practice: combine top-k and top-p together (as in the eval scripts).
# top-k removes extreme long-tail tokens; top-p then adapts to local context.
output = model.generate(
    input_ids,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    max_new_tokens=200,
    eos_id=tokenizer.eos_id,
)
`;

export function EvalPage() {
  const { t } = useLanguage();
  return (
    <ChapterLayout
      num="04"
      title={t("ch04.title")}
      subtitle={t("eval.subtitle")}
      color="text-violet-400"
      prev={{ path: "/pretraining", label: t("ch03.title") }}
      next={{ path: "/sft", label: t("ch05.title") }}
    >
      <PerplexityAnim />

      <Section stepNum={1} title={t("eval.s1.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("eval.s1.p1") }} />

        <div className="grid sm:grid-cols-3 gap-4">
          {[
            { tool: t("eval.s1.tool1.tool"), when: t("eval.s1.tool1.when"), desc: t("eval.s1.tool1.desc"), color: "#0ea5e9" },
            { tool: t("eval.s1.tool2.tool"), when: t("eval.s1.tool2.when"), desc: t("eval.s1.tool2.desc"), color: "#a855f7" },
            { tool: t("eval.s1.tool3.tool"), when: t("eval.s1.tool3.when"), desc: t("eval.s1.tool3.desc"), color: "#10b981" },
          ].map((item) => (
            <div key={item.tool} className="card-glass p-4" style={{ borderColor: `${item.color}20` }}>
              <div className="font-mono text-xs mb-1" style={{ color: item.color }}>{item.tool}</div>
              <div className="text-xs text-gray-500 mb-2">{t("eval.s1.runLabel")} {item.when}</div>
              <p className="text-xs text-gray-400">{item.desc}</p>
            </div>
          ))}
        </div>

        <Callout type="why">
          <span dangerouslySetInnerHTML={{ __html: t("eval.s1.why") }} />
        </Callout>
      </Section>

      <Section stepNum={2} title={t("eval.s2.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("eval.s2.p1") }} />

        <div className="card-glass p-5 mb-2">
          <div className="text-sm font-semibold text-white mb-3">{t("eval.s2.formula.title")}</div>
          <div className="font-mono text-sm text-center py-3">
            <span className="text-brand-300">PPL</span>
            <span className="text-gray-400"> = </span>
            <span className="text-amber-400">e</span>
            <span className="text-gray-400">^(</span>
            <span className="text-emerald-400">−(1/N)</span>
            <span className="text-gray-400"> Σ </span>
            <span className="text-emerald-400">log P(xᵢ | x₁…xᵢ₋₁)</span>
            <span className="text-gray-400">)</span>
          </div>
          <p className="text-xs text-gray-500 text-center mt-2">{t("eval.s2.formula.note")}</p>
        </div>

        <Callout type="warning">
          <span dangerouslySetInnerHTML={{ __html: t("eval.s2.warning") }} />
        </Callout>

        <CodeBlock
          code={perplexityCode}
          filename="unbox_platform/eval/perplexity.py"
          highlights={[38, 39, 40, 41, 42]}
        />
      </Section>

      <Section stepNum={3} title={t("eval.s3.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("eval.s3.p1") }} />

        <div className="grid sm:grid-cols-3 gap-4 mb-4">
          {[
            { name: t("eval.s3.p1m1.name"), formula: t("eval.s3.p1m1.formula"), low: t("eval.s3.p1m1.low"), high: t("eval.s3.p1m1.high"), sweet: t("eval.s3.p1m1.sweet"), color: "#f59e0b" },
            { name: t("eval.s3.p1m2.name"), formula: t("eval.s3.p1m2.formula"), low: t("eval.s3.p1m2.low"), high: t("eval.s3.p1m2.high"), sweet: t("eval.s3.p1m2.sweet"), color: "#0ea5e9" },
            { name: t("eval.s3.p1m3.name"), formula: t("eval.s3.p1m3.formula"), low: t("eval.s3.p1m3.low"), high: t("eval.s3.p1m3.high"), sweet: t("eval.s3.p1m3.sweet"), color: "#10b981" },
          ].map((item) => (
            <div key={item.name} className="card-glass p-4" style={{ borderColor: `${item.color}20` }}>
              <div className="font-semibold text-sm mb-1" style={{ color: item.color }}>{item.name}</div>
              <div className="font-mono text-xs text-gray-500 mb-2">{item.formula}</div>
              <div className="text-xs text-gray-400 space-y-0.5">
                <div>{item.low}</div>
                <div>{item.high}</div>
                <div className="text-white font-medium mt-1">{t("eval.s3.sweet")} {item.sweet}</div>
              </div>
            </div>
          ))}
        </div>

        <CodeBlock
          code={samplingCode}
          filename="sampling strategies"
          highlights={[20, 21, 22, 23]}
        />

        <CodeBlock
          code={sampleCode}
          filename="unbox_platform/eval/sample.py"
          highlights={[27, 28, 29, 30, 34]}
        />
      </Section>

      <Section stepNum={4} title={t("eval.s4.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("eval.s4.p1") }} />
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("eval.s4.p2") }} />

        <div className="grid sm:grid-cols-2 gap-4 mb-4">
          <Callout type="warning" title="Pitfall 1: add_special_tokens=False">
            <span dangerouslySetInnerHTML={{ __html: t("eval.s4.pitfall1") }} />
          </Callout>
          <Callout type="warning" title="Pitfall 2: use_cache=False">
            <span dangerouslySetInnerHTML={{ __html: t("eval.s4.pitfall2") }} />
          </Callout>
        </div>

        <CodeBlock
          code={chatSampleCode}
          filename="unbox_platform/eval/chat_sample.py"
          highlights={[20, 21, 22, 23, 24, 30, 31]}
        />
      </Section>

      <Section stepNum={5} title={t("eval.s5.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("eval.s5.p1") }} />

        <div className="card-glass p-5 mb-4">
          <div className="text-sm font-semibold text-white mb-3">{t("eval.s5.metricsTitle")}</div>
          <div className="space-y-3">
            {[
              { metric: t("eval.s5.m1.m"), formula: t("eval.s5.m1.formula"), baseline: t("eval.s5.m1.baseline"), color: "#10b981" },
              { metric: t("eval.s5.m2.m"), formula: t("eval.s5.m2.formula"), baseline: t("eval.s5.m2.baseline"), color: "#0ea5e9" },
            ].map((item) => (
              <div key={item.metric} className="flex gap-3">
                <div className="w-2 rounded-full flex-shrink-0 mt-1" style={{ background: item.color, minHeight: 40 }} />
                <div>
                  <div className="font-semibold text-sm mb-0.5" style={{ color: item.color }}>{item.metric}</div>
                  <div className="font-mono text-xs text-gray-400 mb-1">{item.formula}</div>
                  <div className="text-xs text-gray-500">{item.baseline}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <CodeBlock
          code={dpoEvalCode}
          filename="unbox_platform/eval/dpo_eval.py"
          highlights={[16, 17, 18, 19, 20, 21, 22]}
        />

        <Callout type="insight">
          <span dangerouslySetInnerHTML={{ __html: t("eval.s5.insight") }} />
        </Callout>
      </Section>

      <Section stepNum={6} title={t("eval.s6.title")}>
        <CodeBlock
          language="bash"
          code={runEvalCode}
          filename="terminal"
        />

        <Callout type="tip">
          <span dangerouslySetInnerHTML={{ __html: t("eval.s6.tip") }} />
        </Callout>
      </Section>
    </ChapterLayout>
  );
}
