"""
OpenRLHF - Open-source Reinforcement Learning from Human Feedback

A comprehensive framework for training large language models using RLHF techniques.
Supports DPO, TPO, GRPO, PPO, and other alignment methods.

Usage with DeepSpeed:
    deepspeed --module openrlhf.cli.train_dpo --formula 22 --pretrain <model_path> --dataset <dataset_path>
    deepspeed --module openrlhf.cli.train_ppo_ray --model_name_or_path <model_path>
    deepspeed --module openrlhf.cli.train_grpo --model_name_or_path <model_path>
"""


# Core modules (import with error handling for missing dependencies)
try:
    from . import cli
    from . import openrlhf_datasets as datasets
    from . import models
    from . import trainer
    from . import utils
except ImportError as e:
    print(f"Warning: Some OpenRLHF modules could not be imported: {e}")
    print("This is normal if dependencies like deepspeed are not installed.")
    print("The package structure is available for DeepSpeed usage.")

__all__ = [
    "cli",
    "datasets", 
    "models",
    "trainer",
    "utils",
    "__version__",
    "__author__",
    "__email__",
]
