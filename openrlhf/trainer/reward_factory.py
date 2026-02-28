#!/usr/bin/env python3
"""
Reward function factory for creating appropriate reward functions based on configuration.
"""

from typing import List, Any

def _get_code_reward():
    try:
        from .grpo_trainer import CodePassrateReward
        return CodePassrateReward
    except ImportError:
        from openrlhf.trainer.grpo_trainer import CodePassrateReward
        return CodePassrateReward

class RewardFactory:
    """Factory class for creating reward functions based on configuration."""
    
    @staticmethod
    def create_reward_functions(reward_mode: str) -> List[Any]:
        """
        Create reward function list based on reward mode.
        
        Args:
            reward_mode: Reward mode ("math", "code", "hardware_verification")
        
        Returns:
            List: List of reward functions
        """
        if reward_mode in ["math", "math_reward"]:
            return RewardFactory._create_math_reward()
        elif reward_mode in ["code", "hardware_verification"]:
            return RewardFactory._create_code_reward()
        else:
            print(f"Warning: Unknown reward mode '{reward_mode}', using default code reward")
            return RewardFactory._create_code_reward()
    
    @staticmethod
    def _create_math_reward() -> List[Any]:
        """Create math reward function."""
        try:
            from .math_reward_class import MathReward
            return [MathReward()]
        except ImportError:
            try:
                from openrlhf.trainer.math_reward_class import MathReward
                return [MathReward()]
            except ImportError:
                print("Warning: MathReward not found, falling back to CodePassrateReward")
                CodePassrateReward = _get_code_reward()
                return [CodePassrateReward()]
    
    @staticmethod
    def _create_code_reward() -> List[Any]:
        """Create code reward function."""
        CodePassrateReward = _get_code_reward()
        return [CodePassrateReward()]

if __name__ == "__main__":
    print("Testing RewardFactory:")
    
    math_rewards = RewardFactory.create_reward_functions("math")
    print(f"Math rewards: {len(math_rewards)} functions")
    
    code_rewards = RewardFactory.create_reward_functions("code")
    print(f"Code rewards: {len(code_rewards)} functions")
    
    default_rewards = RewardFactory.create_reward_functions("unknown")
    print(f"Default rewards: {len(default_rewards)} functions")
