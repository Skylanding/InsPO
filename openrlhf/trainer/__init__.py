"""
OpenRLHF Trainer Module

Contains various trainer implementations for RLHF training methods.
All trainers are designed to work with DeepSpeed distributed training.

Available trainers:
- DPOTrainer: Standard DPO training
- PPOTrainer: PPO training
- GRPOTrainer: GRPO training
- RMTrainer: Reward model training
- SFTTrainer: Supervised fine-tuning
- KDTrainer: Knowledge distillation
- KTOTrainer: KTO training
- PRMTrainer: PRM training

TPO trainers (from tpo_trainer.py):
- BaseDPOTrainer: Base class for TPO trainers
- DPOTrainer22: TPO formula 22 implementation
- DPOTrainer23: TPO formula 23 implementation
- SimPOTrainer24: SimPO formula 24 implementation
- IPOTrainer25: IPO formula 25 implementation
- RDPOTrainer26: R-DPO formula 26 implementation
- ORPOTrainer27: ORPO formula 27 implementation

Usage:
    from openrlhf.trainer.dpo_trainer import DPOTrainer
    from openrlhf.trainer.tpo_trainer import DPOTrainer22, DPOTrainer23
    from openrlhf.trainer.ppo_trainer import PPOTrainer
"""

# Core trainers
from .dpo_trainer import DPOTrainer
from .ppo_trainer import PPOTrainer
from .grpo_trainer import GRPOTrainer
from .refine_grpo_trainer import RefineGRPOTrainer, RefineGRPOConfig
from .rm_trainer import RewardModelTrainer as RMTrainer
from .sft_trainer import SFTTrainer
from .kd_trainer import KDTrainer
from .kto_trainer import KTOTrainer
from .prm_trainer import ProcessRewardModelTrainer as PRMTrainer

# Graders
from .graders import BaseGrader, RLVRGrader, SimpleMLPGrader

# TPO trainers (imported on demand to avoid dependency issues)
from .tpo_trainer import (
    BaseDPOTrainer,
    DPOTrainer22,
    DPOTrainer23,
    SimPOTrainer24,
    IPOTrainer25,
    RDPOTrainer26,
    ORPOTrainer27,
)

__all__ = [
    "DPOTrainer",
    "PPOTrainer",
    "GRPOTrainer",
    "RefineGRPOTrainer",
    "RefineGRPOConfig",
    "RMTrainer",
    "SFTTrainer",
    "KDTrainer",
    "KTOTrainer",
    "PRMTrainer",
    "BaseGrader",
    "RLVRGrader",
    "SimpleMLPGrader",
]
