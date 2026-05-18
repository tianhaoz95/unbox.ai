# Knowledge Distillation: Logit, Hidden State & Reasoning Transfer

## Objective

Knowledge Distillation (KD) aims to compress the capabilities of a large "teacher" model into a smaller "student" model. Unlike simple Supervised Fine-Tuning (SFT) which only uses hard labels (tokens), KD transfers the teacher's probability distributions ("dark knowledge"), internal representations, and logical reasoning patterns.

The goal for `unbox_platform/distill/` is to provide a unified, config-driven interface for three primary distillation modes:
1. **Response-based (Logit Matching)**: Student mimics the teacher's output distribution.
2. **Feature-based (Hidden State Distillation)**: Student aligns its internal activations with the teacher's.
3. **Rationale-based (Reasoning Transfer)**: Student learns step-by-step reasoning traces (DeepSeek-R1 / o1 style).

---

## Architectural Principles

- **Simplicity & Framework Alignment**: Leverage TRL's `GKDTrainer` and `DistillationTrainer` where possible.
- **On-Policy vs. Off-Policy**: Support both static dataset distillation (off-policy) and student-generated feedback loops (on-policy GKD).
- **Heterogeneous Models**: Support cases where student and teacher have different architectures, hidden dimensions, or even different tokenizers.
- **Resource Efficiency**: Support offloading the teacher model to a remote vLLM instance to save local GPU memory.

---

## Distillation Modes

### 1. Logit Matching (Response-based)

The student minimizes the distance between its logits and the teacher's.

- **Standard KD (Off-policy)**: Teacher processes the training dataset once. Loss is usually KL-Divergence between softened distributions.
- **Generalized KD (GKD)**: The student generates its own completions (on-policy), and the teacher provides feedback on those specific generations. This addresses the "distribution shift" problem where the student encounters sequences it never saw during training.
- **Loss Functions**: Forward KL (diversity), Reverse KL (precision/MiniLLM), and JSD (interpolation).

### 2. Hidden State Alignment (Feature-based)

Aligns the internal "thought process" by matching intermediate layer activations.

- **MSE / Cosine Loss**: Computed between corresponding layers of student and teacher.
- **Projection Layers**: If hidden dimensions differ (e.g., student 768 vs teacher 4096), a small learnable linear projection maps student states to teacher space.
- **Attention Map Distillation**: Student mimics the teacher's attention head patterns to ensure it attends to the same context features.

### 3. Reasoning Transfer (Rationale-based)

Distilling the "Chain-of-Thought" (CoT) capability from frontier models like DeepSeek-R1.

- **Trace Distillation**: The dataset contains prompts and teacher-generated reasoning traces (thought + answer). The student is trained via SFT on these traces.
- **Implicit Reasoning**: Experimental mode where the student learns to match the teacher's *final answer* using reasoning, but internalizes the logic so it doesn't necessarily have to output the reasoning tokens at inference (distilling reasoning into weights).

---

## Component Map

```
unbox_platform/distill/
  config.py       # DistillConfig: modes, teacher paths, lmbda (on-policy), beta (JSD)
  train.py        # Entry point: selects trainer based on mode
  trainer.py      # Custom DistillTrainer (extends TRL) for hidden-state hooks
  teacher.py      # Teacher wrapper: local loading or vLLM client
```

---

## Implementation Details

### DistillConfig

Extends `SFTConfig` or `GKDConfig` with distillation-specific parameters:

```python
@dataclass
class DistillTrainConfig(SFTTrainConfig):
    # Teacher setup
    teacher_model_path: str = ""
    teacher_vllm_url: str = ""  # For remote teacher
    
    # Distillation mode
    mode: Literal["logits", "hidden", "reasoning"] = "logits"
    
    # Hyperparameters
    distill_weight: float = 0.5  # Weight of KD loss vs hard label loss
    temperature: float = 2.0     # For logit softening
    on_policy_ratio: float = 0.5 # GKD 'lmbda'
    
    # Feature-based setup
    layer_mapping: dict[int, int] = field(default_factory=dict) # student_layer -> teacher_layer
    projection_dim: int | None = None
```

### Teacher Handling

To handle large teachers (e.g., Llama-3 70B) distilling into small students (e.g., Unbox 1B), the teacher should be:
1. **Quantized**: Load teacher in 4-bit/8-bit if local.
2. **Offloaded**: Use `teacher_vllm_url` to query a teacher running on a separate node/pool.

### Handling Cross-Tokenizer Distillation

When the student and teacher use different tokenizers (e.g., distilling Llama-3 into Unbox), tokens don't align 1-to-1.
- **Strategy**: Perform distillation at the **text level** or use **Universal Logit Distillation (ULD)** which uses Wasserstein distance to align distributions across different vocabularies.

---

## Implementation Roadmap

1. **Phase 1: Rationale SFT**: Implement "DeepSeek-R1 style" distillation where we simply fine-tune the student on teacher-generated reasoning datasets. This is a special case of SFT with `completion_only_loss=True`.
2. **Phase 2: Logit GKD**: Integrate TRL's `GKDTrainer` for on-policy logit distillation. Requires `teacher_model` (local or vLLM).
3. **Phase 3: Hidden State Alignment**: Implement a custom `DistillTrainer` that adds MSE loss between layer activations. This requires adding forward hooks to both models.
4. **Phase 4: Cross-Tokenizer Support**: Implement ULD or logit-mapping for heterogeneous distillation.
