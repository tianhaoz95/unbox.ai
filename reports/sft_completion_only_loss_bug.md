# SFT Bug: `completion_only_loss` Has No Effect for Messages-Format Datasets

**Date:** 2026-05-25
**Symptom:** Model generates degenerate "user" loops — correct first token, then `user` repeated
**Root cause:** `completion_only_loss=True` has no effect for `messages`-format datasets in TRL 1.4

---

## Symptom

All checkpoints (SFT and DPO) produced degenerate generation:

```
Input:  "The capital of France is"
Output: " Paris" + "user" × 29

Input:  "<|im_start|>user\nExplain lists vs tuples<|im_end|>\n<|im_start|>assistant\n"
Output: "A" + "user" × 149
```

The model produces a correct first token (factual knowledge intact) then immediately loops
on the word `user`. This is the **"user attractor" effect**: `user` became a near-universal
high-probability next token at any position in the sequence.

---

## Root Cause

### TRL 1.4 has two separate masking systems

| Parameter | Dataset format | Mechanism |
|---|---|---|
| `completion_only_loss=True` | `prompt`+`completion` | Creates `completion_mask` |
| `assistant_only_loss=True` | `messages` (conversational) | Creates `assistant_masks` via `return_assistant_tokens_mask=True` |

Our SFT training used **`messages` format** (UltraChat has a `messages` column) but set
**`completion_only_loss=True`**. TRL's tokenization path for conversational data:

```python
# trl/trainer/sft_trainer.py:686
processed = self._tokenize(
    processing_class,
    example["messages"],
    return_assistant_tokens_mask=assistant_only_loss,  # ← uses assistant_only_loss, NOT completion_only_loss
)
output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
```

Since `assistant_only_loss` defaulted to `False`, `return_assistant_tokens_mask=False`,
no `assistant_masks` were generated, and **all tokens — including user turns — contributed
to the SFT loss**.

### Second blocker: missing `{% generation %}` in chat template

Even with `assistant_only_loss=True`, the tokenizer needs `{% generation %}` markers in
the chat template to know which token positions are assistant-generated. Our original
template had no such markers, causing TRL to raise an error.

### Consequence

The model trained on loss over ALL tokens in every example:
```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n
         ←────────────── loss computed here ────────────────→
```

After 1 epoch on 207k UltraChat examples, every training example ended with
`<|im_end|>\n<|im_start|>user` — the transition to a new user turn. The model learned
this structural pattern so strongly (on a weak base model with only ~5B pretrain tokens)
that `user` became a near-universal high-probability continuation token, overriding any
content generation priors.

---

## Fix

### 1. Update chat template with `{% generation %}` markers (`configs/tokenizer/chat_template.jinja`)

```jinja
{% for message in messages %}{% if message['role'] == 'assistant' %}{{ '<|im_start|>' + message['role'] + '\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}{% else %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

### 2. Change `completion_only_loss` → `assistant_only_loss` in `sft/train.py`

```python
# Before (broken — no effect for messages format)
completion_only_loss=True,

# After (correct for messages/conversational format)
assistant_only_loss=True,
```

### 3. Load chat template in `build_tokenizer()` via `chat_template_path` config field

### Verification

With the fix, `apply_chat_template(..., return_assistant_tokens_mask=True)` correctly
marks only the assistant content tokens (1), masking all user turn and formatting tokens (0):

```
<|im_start|>  → 0  (masked)
user          → 0  (masked)
\n            → 0  (masked)
Hi            → 0  (masked — user content)
<|im_end|>    → 0  (masked)
<|im_start|>  → 0  (masked)
assistant     → 0  (masked — role header)
\n            → 0  (masked — role header)
Hello         → 1  ← TRAINED (assistant content only)
<|im_end|>    → 0  (masked)
```

---

## Impact

This bug affected the SFT checkpoint and (through it) the DPO checkpoint. The DPO training
metrics (margins, accuracy) are still valid — the preference learning signal is real — but
generation quality requires a corrected SFT rerun.

---

## Lessons

1. **`completion_only_loss` ≠ `assistant_only_loss` in TRL.** They target different dataset
   formats. Using the wrong one silently does nothing.
2. **The `{% generation %}` keyword is required** for `assistant_only_loss` to work.
3. **Always verify masking** with `apply_chat_template(..., return_assistant_tokens_mask=True)`
   before starting a long training run.
