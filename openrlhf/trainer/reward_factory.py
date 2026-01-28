#!/usr/bin/env python3
"""
奖励函数工厂
根据配置创建合适的奖励函数
"""

from typing import List, Any

# 延迟导入以避免循环导入
def _get_code_reward():
    try:
        from .grpo_trainer import CodePassrateReward
        return CodePassrateReward
    except ImportError:
        # 如果相对导入失败，尝试绝对导入
        from openrlhf.trainer.grpo_trainer import CodePassrateReward
        return CodePassrateReward

class RewardFactory:
    """
    奖励函数工厂类
    根据配置创建合适的奖励函数
    """
    
    @staticmethod
    def create_reward_functions(reward_mode: str) -> List[Any]:
        """
        根据奖励模式创建奖励函数列表
        
        Args:
            reward_mode: 奖励模式 ("math", "code", "hardware_verification")
        
        Returns:
            List: 奖励函数列表
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
        """创建数学奖励函数"""
        try:
            from .math_reward_class import MathReward
            return [MathReward()]
        except ImportError:
            try:
                # 尝试绝对导入
                from openrlhf.trainer.math_reward_class import MathReward
                return [MathReward()]
            except ImportError:
                print("Warning: MathReward not found, falling back to CodePassrateReward")
                CodePassrateReward = _get_code_reward()
                return [CodePassrateReward()]
    
    @staticmethod
    def _create_code_reward() -> List[Any]:
        """创建代码奖励函数"""
        CodePassrateReward = _get_code_reward()
        return [CodePassrateReward()]

# 测试函数
if __name__ == "__main__":
    # 测试不同奖励模式
    print("Testing RewardFactory:")
    
    # 测试数学奖励
    math_rewards = RewardFactory.create_reward_functions("math")
    print(f"Math rewards: {len(math_rewards)} functions")
    
    # 测试代码奖励
    code_rewards = RewardFactory.create_reward_functions("code")
    print(f"Code rewards: {len(code_rewards)} functions")
    
    # 测试默认奖励
    default_rewards = RewardFactory.create_reward_functions("unknown")
    print(f"Default rewards: {len(default_rewards)} functions")
