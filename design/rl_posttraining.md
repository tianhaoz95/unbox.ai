# RL Post-Training Design

## Overview

RL post-training teaches the model to produce outputs that score well on a reward signal, rather than simply imitating a training dataset. This is the stage that separates a capable base/SFT model from an aligned, instruction-following assistant (RLHF) or a reasoning model (GRPO on verifiable tasks).

This document covers framework choices, the three-method progression, implementation targets, and open design questions.

---

## Framework Choices

| Method | Framework | Rationale |
|---|---|---|
| DPO | TRL `DPOTrainer` + `DPOConfig` | Same trainer ecosystem as SFT; no rollout infrastructure needed |
| GRPO | TRL `GRPOTrainer` + `GRPOConfig` | Offline group-relative scoring; reward function replaces reward model |
| PPO (online) | OpenRLHF | TRL's PPO is not production-grade for actor-critic rollout loops; OpenRLHF provides the distributed rollout buffer, reward model serving, and reference model KL penalty needed for stable training |
| RLVR (custom) | mini-sglang rollout + Megatron-Core training | Disaggregated rollout and training for verifiable-reward tasks; see `design/rlvr.md` |

TRL is used for all offline methods (DPO, GRPO) because it shares the same trainer ecosystem as SFT and requires no running environment. OpenRLHF is used for online PPO. The custom RLVR path (see `design/rlvr.md`) targets verifiable-reward tasks (math, code) with a disaggregated rollout engine and Megatron-Core training, giving full-stack visibility into how production RLVR systems work.

---

## Method Progression

### Stage 1 · DPO (Direct Preference Optimization)

**What it does:** Trains directly on static preference pairs (chosen > rejected) without a reward model. Increases the log-probability of the chosen response relative to the rejected one.

**When to run:** Immediately after SFT is validated. DPO is the cheapest RL experiment — no rollout loop, no reward model training, just a dataset and a trainer.

**Dataset:** `HuggingFaceH4/ultrafeedback_binarized` or `Anthropic/hh-rlhf`. Must have `"prompt"`, `"chosen"`, `"rejected"` columns.

**Entry point:**
```bash
.venv/bin/python -m unbox_platform.rl.train --config configs/rl/dpo.yaml
```

**Key config parameters:**

| Parameter | Value | Notes |
|---|---|---|
| `beta` | 0.1 | KL penalty coefficient; lower = more aggressive preference learning |
| `loss_type` | `sigmoid` | Standard DPO loss |
| `max_length` | 2048 | |
| `max_prompt_length` | 512 | |
| `learning_rate` | 5e-7 | Much lower than SFT; adjusting preference, not overwriting representations |

**Limitation:** Bounded by what's in the preference dataset. Cannot improve capabilities the base model doesn't already have.

---

### Stage 2 · GRPO (Group Relative Policy Optimization)

**What it does:** For each prompt, generates a group of N responses, scores them with a reward function, and trains the model to prefer responses that score above the group mean. No reward model required — the reward function is rule-based.

**When to run:** After DPO is validated, or as an alternative first step if the goal is capability improvement (e.g. math) rather than alignment.

**Why GRPO over PPO for verifiable tasks:** GRPO's group-relative baseline avoids reward hacking and does not require a critic network. For tasks with a binary verifiable reward (correct/incorrect), it is simpler and more stable than PPO.

**Target task: GSM8K (grade-school math)**

The reward function checks exact answer match after extracting the final numeric answer:

```python
def reward_fn(responses: list[str], prompts: list[str]) -> list[float]:
    scores = []
    for response, prompt in zip(responses, prompts):
        extracted = extract_answer(response)   # parse "#### <number>" pattern
        correct = get_ground_truth(prompt)
        scores.append(1.0 if extracted == correct else 0.0)
    return scores
```

**Entry point:**
```bash
.venv/bin/python -m unbox_platform.rl.train --config configs/rl/grpo_math.yaml
```

**Key config parameters:**

| Parameter | Value | Notes |
|---|---|---|
| `num_generations` | 8 | Responses per prompt per step |
| `learning_rate` | 1e-6 | |
| `max_new_tokens` | 512 | Enough for chain-of-thought |
| `reward_funcs` | `["exact_match"]` | Pluggable; add format compliance, length penalty, etc. |

---

### Stage 3 · PPO (Proximal Policy Optimization, online)

**What it does:** Full online RLHF — generates responses during training, scores them with a trained reward model, applies KL penalty from a frozen reference model (the SFT model), and updates both actor and critic. This is how ChatGPT, Claude, and Llama-3-Instruct achieve their alignment properties.

**When to run:** After DPO and GRPO are validated and a reward model has been trained. This is the most complex stage and should not be attempted before the earlier stages are stable.

**Infrastructure (via OpenRLHF):**

```
Actor (SFT model, trainable)
    → generate responses
    → reward model scores them
    → KL penalty vs. reference model (frozen SFT)
    → PPO clipped objective
    → update actor + critic
```

OpenRLHF runs these as separate processes with vLLM handling fast generation in the rollout pool. The reward model is a separate server.

**Reward model training:** Train a scalar-head classifier on a preference dataset (same as DPO) using the SFT model as backbone. Freeze it before the PPO run.

**Entry point:**
```bash
# reward model training
.venv/bin/python -m unbox_platform.rl.train_reward_model --config configs/rl/reward_model.yaml

# PPO via OpenRLHF
.venv/bin/python -m unbox_platform.rl.train --config configs/rl/ppo.yaml
```

---

## Evaluation

RL improvements must be measured against concrete benchmarks — loss alone is not meaningful after SFT.

| Stage | Benchmark | Metric |
|---|---|---|
| DPO | MT-Bench | GPT-4 judge score (1–10) |
| DPO | AlpacaEval | Win rate vs. reference |
| GRPO | GSM8K | Exact match accuracy |
| PPO | MT-Bench + MMLU | GPT-4 judge + accuracy |

Run the benchmark on the SFT checkpoint before any RL training to establish a baseline, then again after each RL stage to measure delta.

---

## Implementation Status

| Component | Status |
|---|---|
| `unbox_platform/rl/` directory | Stub only |
| DPO trainer | Not implemented |
| GRPO trainer | Not implemented |
| Reward model training | Not implemented |
| PPO via OpenRLHF | Not implemented |
| `configs/rl/` configs | Not implemented |

Implementation will begin after pre-training and SFT are validated.

---

## Open Questions

- **DPO dataset choice:** `ultrafeedback_binarized` has broader topic coverage; `hh-rlhf` is more focused on helpfulness/harmlessness. Which aligns better with the evaluation target?
- **GRPO reward shaping:** Binary exact-match reward is sparse for longer chains of thought. A partial-credit reward (correct reasoning steps even if final answer is wrong) may train more stably.
- **Skip DPO?** Given the model is small (760M) and SFT is fresh, DPO gains may be marginal. GRPO on a verifiable task may be more informative as a research experiment.
- **Reward model size:** Does the reward model need to be the same size as the policy, or can a smaller model serve as an adequate judge?
