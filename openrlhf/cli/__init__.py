"""
OpenRLHF CLI Module

Command-line interface for training models with various RLHF techniques.
All training scripts are designed to work with DeepSpeed.

Available training scripts:
- train_dpo: DPO and TPO training (supports formulas 22, 23, 24, 25, 26, 27)
- train_ppo_ray: PPO training with Ray
- train_grpo: GRPO training
- train_rm: Reward model training
- train_sft: Supervised fine-tuning
- train_kd: Knowledge distillation
- train_kto: KTO training
- train_prm: PRM training
- batch_inference: Batch inference
- interactive_chat: Interactive chat
- serve_rm: Serve reward model
- lora_combiner: LoRA model combination

Usage examples:
    deepspeed --module openrlhf.cli.train_dpo --formula 22 --pretrain <model_path> --dataset <dataset_path>
    deepspeed --module openrlhf.cli.train_ppo_ray --model_name_or_path <model_path>
    deepspeed --module openrlhf.cli.train_grpo --model_name_or_path <model_path>
"""

# Import training modules (avoid importing modules with external dependencies)
try:
    from . import train_dpo
    from . import train_kd
    from . import train_kto
    from . import train_ppo_ray
    from . import train_prm
    from . import train_rm
    from . import train_sft
except ImportError as e:
    print(f"Warning: Some CLI modules could not be imported: {e}")

# Optional modules (may have external dependencies)
try:
    from . import batch_inference
    from . import interactive_chat
    from . import lora_combiner
    from . import serve_rm
except ImportError:
    pass  # These modules may have external dependencies

__all__ = [
    "batch_inference",
    "interactive_chat",
    "lora_combiner",
    "serve_rm",
    "train_dpo",
    "train_kd",
    "train_kto",
    "train_ppo_ray",
    "train_prm",
    "train_rm",
    "train_sft",
]

