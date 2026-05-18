# Guide: Publishing Unbox Models to Hugging Face

This guide outlines the process for converting internal Unbox checkpoints into the standard Hugging Face format and publishing them to the Hub.

## Overview

The `unbox_platform.model.hf_adapter` provides the necessary wrappers (`UnboxForCausalLM` and `UnboxConfig`) to make the models compatible with the Hugging Face ecosystem.

## 1. Preparation

Ensure you have the `huggingface_hub` library installed in your environment and log in via the CLI.

```bash
# Install the library if needed
.venv/bin/pip install huggingface_hub

# Login to your account
.venv/bin/huggingface-cli login
```

## 2. Publishing the Base Model

Pre-training saves raw `.pt` checkpoints which contain the state dictionary. To publish these, you must load them into the HF adapter and then push.

### Conversion Script Template

Create a script (e.g., `scripts/publish_base.py`):

```python
from unbox_platform.model.hf_adapter import UnboxForCausalLM, UnboxConfig
from transformers import PreTrainedTokenizerFast

# 1. Configuration
# Ensure the config matches the pre-training hyperparameters
config = UnboxConfig(
    hidden_size=1792,
    num_layers=24,
    num_heads=16,
    num_kv_heads=8,
    # ... other params
)

# 2. Load the model from the raw checkpoint
# This adds the necessary 'model.' prefixes to the state dict
model = UnboxForCausalLM.from_unbox_checkpoint(
    "checkpoints/pretrain/760m/latest.pt", 
    config
)

# 3. Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("checkpoints/tokenizer")

# 4. Push to the Hub
repo_id = "your-username/unbox-760m-base"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

## 3. Publishing the SFT Model

Models trained using `unbox_platform.sft.train` are already saved in the Hugging Face format by the `SFTTrainer`. You can push them directly from the output directory.

```python
from unbox_platform.model.hf_adapter import UnboxForCausalLM
from transformers import PreTrainedTokenizerFast

repo_id = "your-username/unbox-760m-sft"
sft_path = "checkpoints/sft"

# Load the already-formatted model
model = UnboxForCausalLM.from_pretrained(sft_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(sft_path)

# Push to hub
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

## 4. Advanced: Custom Code Support (`trust_remote_code`)

To allow users to load your model without installing `unbox_platform` (using `trust_remote_code=True`), you must include the model files in the repository.

1. Add `auto_map` to your configuration before pushing:
   ```python
   model.config.auto_map = {
       "AutoConfig": "hf_adapter.UnboxConfig",
       "AutoModelForCausalLM": "hf_adapter.UnboxForCausalLM"
   }
   ```
2. Ensure `hf_adapter.py`, `model.py`, and `config.py` are copied into the repository root on the Hub.

## Workflow Summary

| Model Phase | Source Format | Target Format | Tooling |
| :--- | :--- | :--- | :--- |
| **Pre-training** | `.pt` (state dict) | `model.safetensors` | `UnboxForCausalLM.from_unbox_checkpoint()` |
| **SFT** | HF Directory | `model.safetensors` | Direct `push_to_hub()` |

## Model Card Best Practices

When publishing, always update the `README.md` (model card) on Hugging Face:
- **Base Models**: Include training data details (FineWeb-Edu), hardware specs, and final loss/perplexity.
- **SFT Models**: Include the chat template format, the SFT dataset used, and example prompts/responses.
