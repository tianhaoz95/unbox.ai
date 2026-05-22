# unbox.ai Product Strategy

## Core Asset

The codebase is intentionally **readable and full-stack** — data pipeline through inference,
in one repo, without Megatron-LM complexity. Most tools are either research toys or production
black boxes. unbox.ai occupies the middle: a complete, hackable system where every component
can be read top-to-bottom and understood.

---

## Product Angles Considered

### Angle 1: Educational Platform (Selected)

Package the codebase as a structured learning resource for engineers who want to understand
LLM internals — not just call APIs.

**Why this fits:**
- Readability is the explicit design goal — it doubles as pedagogy
- The repo already follows a natural curriculum: tokenizer → pretrain → SFT → RL → distillation → inference
- Design docs (`design/`) are ready-made lesson companions
- The unbox-760m training run is a concrete, reproducible story to teach from

**Why angle 1 before others:**
- Fastest to ship (written content, then notebooks, then video)
- Builds an audience before committing to GPU infra costs
- Users who finish the course and say "I want to run this on my data" become the first paying
  customers for a future fine-tuning API (angle 2)

### Angle 2: Fine-tuning / Distillation API (Future)

Host the SFT + DPO/GRPO + distillation pipeline. Users bring data, get back a small deployable
model. Differentiator is distillation — compress a large teacher into a small checkpoint — which
is underserved compared to raw SFT offerings.

### Angle 3: On-prem Training Software (Future)

License the platform as the approachable software stack for teams buying their own GPU hardware
(DGX Spark, etc.). Long sales cycle but high ACV.

---

## Building the Educational Product

### Curriculum Skeleton (already in the repo)

The module sequence maps directly to `unbox_platform/`:

| Module | Source | Design doc |
|---|---|---|
| 1. Tokenizer | `unbox_platform/tokenizer/` | — |
| 2. Data pipeline | `unbox_platform/data/` | — |
| 3. Pretraining | `unbox_platform/train/` | `design/mvp_analysis.md` |
| 4. Supervised fine-tuning | `unbox_platform/sft/` | — |
| 5. RL post-training | `unbox_platform/rl/` | `design/rl_posttraining.md` |
| 6. Distillation | `unbox_platform/distill/` | `design/distillation.md` |
| 7. Inference engine | `unbox_platform/infer/` | `design/infer_completeness_review.md` |

### Format Decision

| Format | Effort | Revenue model | When |
|---|---|---|---|
| Written guide / blog series | Low | Free → funnel | Now |
| Notebooks + annotated code | Medium | GitHub Sponsors / one-time purchase | Month 1–2 |
| Video course | High | Udemy / Maven / own site | After audience validation |
| Live cohort | High | $500–2k/seat | After 1–2 written courses |

**Recommended path:** start with written + notebooks (fast to ship, SEO-friendly, audience-building),
record video on top of what gets traction, then add live cohort once demand is proven.

### Target Learner

The codebase is best matched to:

- **ML engineer at a company** — knows PyTorch, calls HuggingFace APIs daily, wants to understand
  what's inside. Willing to pay for depth. Best fit.
- **CS grad student** — reads papers, hasn't run a real training loop end-to-end. Price-sensitive
  but high volume.

Write for one of these specifically. The "senior engineer pivoting to AI" needs more
hand-holding than the codebase currently provides.

### Monetization

- **Free core → paid advanced**: pretrain module free (public, SEO), RL + distillation + inference
  behind a paywall. Proven funnel structure.
- **Free code, paid cohort**: the repo is the textbook, live sessions with office hours are the
  product. Higher ACV, lower volume.

---

## Concrete First Step

Turn `design/mvp_analysis.md` into a public post explaining the architectural decisions behind
the 760m training run — why Megatron-Core, why the 50% throughput target, what you learned.
That post:

1. Tests whether the audience exists before building a course
2. Drives GitHub stars and early subscribers
3. Is one lesson already written

The unbox-760m training run (pretraining a real model from scratch, documented with sanity
checks at step 10k and 15k) is a story worth telling publicly.

---

## Key Decisions Outstanding

1. **Which learner type to target first** — company ML engineer or grad student
2. **Paid vs. free core** — determines how aggressively to gate advanced modules
3. **Platform** — own site (Ghost/Substack), Maven, or Udemy
