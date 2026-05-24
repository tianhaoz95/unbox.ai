# Postmortem: SFT User Token Masking Bug

**Date discovered:** 2026-05-25
**Date introduced:** Initial SFT training run (~2026-05-23)
**Severity:** Critical — all SFT and DPO checkpoints produced degenerate outputs
**Training wasted:** ~2 hours SFT + ~2 hours DPO (3 epochs) + multiple debugging runs

---

## 1. How We Observed the Issue

After completing the full training pipeline (Pretrain → SFT → DPO), we attempted
qualitative evaluation by generating responses to three prompts across all DPO
checkpoints:

**Prompts:**
1. `Explain the difference between a list and a tuple in Python in one sentence.`
2. `I have 10 minutes to study for an exam. What should I do?`
3. `Is it better to use tabs or spaces for Python indentation? Give me a direct answer.`

**Observed output (every checkpoint, every prompt):**
```
"A" + "user" × 149
"1" + "user" × 149
"It" + "user" × 149
```

The model produced a single meaningful first token then immediately looped on the
word `user` for the entire remaining generation budget. The same pattern appeared
in ALL checkpoints — SFT baseline and DPO steps 500 through 2868.

**Wider test — plain text continuation:**
```
Input:  "The capital of France is"
Output: " Paris" + "user" × 29
```

Correct first token (`Paris`), then immediate degeneration. This ruled out the
chat template format as the cause and pointed to a deeper training issue.

---

## 2. Initial Hypothesis

**User's hypothesis:** *"The model is trying to generate the user's next message
rather than an assistant response — it's putting words in the user's mouth.
This suggests user tokens were not masked from the loss during SFT or DPO."*

This was the correct hypothesis. In standard SFT, only assistant response tokens
should contribute to the training loss. If user turn tokens are included in the loss,
the model learns to predict them, making `user` a high-probability output token.

---

## 3. Investigation: Tracing the Masking Path

### Step 1: Inspect the SFT training config

```python
# unbox_platform/sft/train.py
training_args = SFTConfig(
    ...
    completion_only_loss=True,   # ← this was supposed to mask user tokens
    ...
)
```

Appeared correct at first glance.

### Step 2: Read TRL 1.4 source code

Inspecting `trl/trainer/sft_trainer.py` revealed two entirely separate masking systems:

```python
# For messages/conversational format (line 686):
processed = self._tokenize(
    processing_class,
    example["messages"],
    return_assistant_tokens_mask=assistant_only_loss,  # ← checks assistant_only_loss
)
output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}

# For prompt+completion format (line 676):
completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
output["completion_mask"] = completion_mask
```

The collator then applies whichever mask is present:
```python
if completion_mask is not None:
    output["labels"][completion_mask == 0] = -100   # prompt+completion path
elif assistant_masks is not None:
    output["labels"][assistant_masks == 0] = -100   # messages path
# If NEITHER is present: no masking at all
```

**Key finding:** `completion_only_loss` controls the `prompt+completion` path.
For `messages` format, the correct parameter is `assistant_only_loss`. We used
the wrong one. Since `assistant_only_loss` defaulted to `False`, no masking
was ever applied.

### Step 3: Identify the second blocker

Even if `assistant_only_loss=True` had been set, TRL requires `{% generation %}`
markers in the chat template (line 388):

```python
if args.assistant_only_loss and "{% generation %}" not in processing_class.chat_template:
    raise ValueError("Chat template missing {% generation %} keyword...")
```

Our original chat template:
```jinja
{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

No `{% generation %}` → even the correct parameter would have failed silently
(or raised an error depending on TRL version).

---

## 4. Validation

### Direct mask verification

After applying the fix (see Section 5), confirmed correct masking:

```python
tok.apply_chat_template(sample, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
# Output:
# input_ids:       [5, 15724, 195, 29503, 6, 195, 5, 532, 8489, 195, 30762, 6, 195]
# assistant_masks: [0,     0,   0,     0, 0,   0, 0,   0,    0,   0,     1, 0,   0]
```

Token-by-token:
```
<|im_start|>  → masked (0)
user          → masked (0)
\n            → masked (0)
Hi            → masked (0)  ← user content, correctly masked
<|im_end|>    → masked (0)
\n            → masked (0)
<|im_start|>  → masked (0)
ass           → masked (0)  ← assistant header, correctly masked
istant        → masked (0)  ← assistant header, correctly masked
\n            → masked (0)
Hello         → TRAINED (1) ← only assistant content token gets loss
<|im_end|>    → masked (0)
\n            → masked (0)
```

Only the assistant's actual content (`Hello`) is trainable. All user turn tokens
and assistant header tokens (`<|im_start|>assistant\n`) are masked.

### Why the loop occurs on the buggy model

With user tokens included in the loss, the model learned the full conversation
structure including turn transitions. Every training example ended with:
```
{assistant response}<|im_end|>\n<|im_start|>user\n{next question}...
```

After 207,865 training examples × 1 epoch, the model learned:
*"After generating content, the next highly probable tokens are `<|im_end|>`,
`\n`, `<|im_start|>`, `user`."*

On a model with only ~5B pretraining tokens, this structural attractor overwhelmed
the language content priors. Greedy decoding locks onto the highest-probability
continuation, which was `user` at almost any position.

---

## 5. Solution

Three coordinated changes:

### Fix 1: Add `{% generation %}` to the chat template

```jinja
{% for message in messages %}{% if message['role'] == 'assistant' %}{{ '<|im_start|>' + message['role'] + '\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}{% else %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

Stored in `configs/tokenizer/chat_template.jinja` (tracked in git).

### Fix 2: Change `completion_only_loss` → `assistant_only_loss`

```python
# sft/train.py — SFTConfig
assistant_only_loss=True,   # was: completion_only_loss=True
```

### Fix 3: Load chat template in tokenizer builder

```python
# sft/train.py — build_tokenizer()
if cfg.chat_template_path:
    tokenizer.chat_template = Path(cfg.chat_template_path).read_text()
```

Config update (`configs/sft/basic.yaml`):
```yaml
chat_template_path: configs/tokenizer/chat_template.jinja
```

---

## 6. Impact Assessment

| Training run | Status | Action required |
|---|---|---|
| Pretrain (34k steps) | ✅ Unaffected (no chat format) | None |
| SFT v1 (1 epoch, buggy) | ❌ All tokens in loss | **Retrain** |
| DPO run 1 (LR=5e-7) | ❌ Based on buggy SFT | **Retrain after SFT** |
| DPO run 2 (LR=1e-6, 1 epoch) | ❌ Based on buggy SFT | **Retrain after SFT** |
| DPO run 3 (LR=1e-6, 3 epochs) | ❌ Based on buggy SFT | **Retrain after SFT** |

DPO metric improvements (margins, accuracy) in runs 2–3 are still valid as evidence
that the DPO training procedure works correctly. Those improvements were real relative
improvements over the reference model — just a reference model that was already broken.

---

## 7. Lessons Learned

### Always verify masking before starting a long training run

```python
# Add this check to any SFT training script
result = tokenizer.apply_chat_template(
    [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}],
    tokenize=True, return_dict=True, return_assistant_tokens_mask=True
)
assert 1 in result["assistant_masks"], "No assistant tokens found — masking is broken!"
assert result["assistant_masks"].count(1) < len(result["assistant_masks"]), "All tokens trainable — masking is broken!"
```

### `completion_only_loss` ≠ `assistant_only_loss` in TRL

| | `completion_only_loss` | `assistant_only_loss` |
|---|---|---|
| Dataset format | `prompt`+`completion` | `messages` (conversational) |
| Mask field generated | `completion_mask` | `assistant_masks` |
| Requires in template | Nothing | `{% generation %}` markers |

When your dataset has a `messages` column with multi-turn conversations, always use
`assistant_only_loss=True`.

### `{% generation %}` is required for `assistant_only_loss` to work

Any custom chat template used with `assistant_only_loss=True` must include
`{% generation %}` and `{% endgeneration %}` markers around the assistant content.
Without them, TRL either raises an error or silently produces no masks depending on
the version.

### Test generation quality early

A 10-minute inference check after the first 500 SFT steps would have caught this
before wasting the full training run. Early qualitative evaluation is cheap insurance.

### A weak base model amplifies masking bugs

With a well-pretrained model (100B+ tokens), the language content priors are strong
enough that structural attractors like `user` are suppressed during generation. Our
model (~5B tokens) had weak priors, making it extremely sensitive to what the loss
was computed on. The bug would have been less visible — but still harmful — on a
stronger model.
