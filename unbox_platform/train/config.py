"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # data_parallel_size is inferred: world_size / (tp * pp)

    # Megatron-Core distributed optimizer (ZeRO-style)
    use_distributed_optimizer: bool = True


@dataclass
class TrainConfig:
    # Paths
    tokenizer_path: str = "checkpoints/tokenizer"
    output_dir: str = "checkpoints/pretrain"
    data_path: str = "data/fineweb_edu_10bt.jsonl"
    dataset_name: str = "sample-10BT"

    # Model
    # (ModelConfig is passed separately; these mirror defaults for config files)

    # Optimization
    batch_size: int = 4                  # per-GPU micro batch size
    grad_accumulation_steps: int = 8
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1           # min_lr = max_lr * min_lr_ratio
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    num_epochs: int = 1
    warmup_steps: int = 2000

    # Precision
    dtype: str = "bfloat16"            # "bfloat16" | "float16" | "float32"

    # Checkpointing
    save_every_steps: int = 1000
    log_every_steps: int = 50
    eval_every_steps: int = 500

    # Parallelism
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    # Sequence length (must match ModelConfig.max_seq_len)
    max_seq_len: int = 2048

    def effective_batch_size(self, world_size: int = 1) -> int:
        return self.batch_size * self.grad_accumulation_steps * world_size
