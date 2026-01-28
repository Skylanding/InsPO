#!/usr/bin/env python3
"""
数学推理奖励函数类
用于 OpenRLHF GRPO 训练
"""

import re
from typing import List, Any, Optional

class MathReward:
    """
    数学推理奖励函数类
    兼容 OpenRLHF GRPO 训练器的接口
    """
    
    def __init__(self):
        self.name = "math_reward"
    
    def __call__(self, completions: List[str], prompt: str = "", stage: str = "base", 
                 draft: Optional[str] = None, turn: Optional[str] = None) -> List[float]:
        """
        计算数学答案的奖励分数
        
        Args:
            completions: 生成的答案列表
            prompt: 原始问题
            stage: 阶段 ("base" 或 "refine")
            draft: 草稿答案 (用于 refine 阶段)
            turn: 轮次信息
        
        Returns:
            List[float]: 每个答案的奖励分数
        """
        rewards = []
        
        for completion in completions:
            if not completion or not completion.strip():
                rewards.append(0.0)
                continue
                
            reward = self._calculate_math_reward(completion, prompt, stage, draft)
            rewards.append(reward)
        
        return rewards
    
    def _calculate_math_reward(self, answer: str, prompt: str = "", 
                              stage: str = "base", draft: Optional[str] = None) -> float:
        """
        计算数学答案的奖励分数
        
        奖励标准：
        1. 格式正确性 (40%): 包含计算步骤和最终答案
        2. 数学正确性 (60%): 基于答案格式和结构
        
        Args:
            answer: 生成的答案文本
            prompt: 原始问题
            stage: 阶段
            draft: 草稿答案
        
        Returns:
            float: 奖励分数 (0-1)
        """
        if not answer or not answer.strip():
            return 0.0
        
        # 1. 格式检查 (40% 权重)
        format_score = self._check_answer_format(answer)
        
        # 2. 数学内容检查 (60% 权重)  
        math_score = self._check_math_content(answer)
        
        # 3. 如果是 refine 阶段，检查是否比 draft 更好
        refine_bonus = 0.0
        if stage == "refine" and draft:
            refine_bonus = self._check_refinement_improvement(answer, draft)
        
        # 加权平均
        total_score = 0.4 * format_score + 0.6 * math_score + 0.1 * refine_bonus
        
        return max(0.0, min(1.0, total_score))
    
    def _check_answer_format(self, answer: str) -> float:
        """
        检查答案格式是否正确
        
        期望格式：
        - 包含计算步骤
        - 有最终答案 (#### 数字)
        - 有合理的数学表达式
        """
        score = 0.0
        
        # 检查是否包含最终答案格式 (#### 数字)
        if re.search(r'####\s*-?\d+(?:\.\d+)?', answer):
            score += 0.4
        
        # 检查是否包含计算步骤
        if re.search(r'[+\-*/=]', answer):
            score += 0.3
        
        # 检查是否包含数字
        if re.search(r'\d+', answer):
            score += 0.2
        
        # 检查是否有合理的长度 (不是太短)
        if len(answer.strip()) > 20:
            score += 0.1
        
        return score
    
    def _check_math_content(self, answer: str) -> float:
        """
        检查数学内容的合理性
        """
        score = 0.0
        
        # 检查是否包含数学运算
        math_operations = re.findall(r'[+\-*/=]', answer)
        if len(math_operations) >= 1:
            score += 0.3
        
        # 检查是否有数字计算
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        if len(numbers) >= 2:
            score += 0.3
        
        # 检查是否有合理的数学表达式
        if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', answer):
            score += 0.2
        
        # 检查是否包含数学关键词
        math_keywords = ['calculate', 'solve', 'find', 'compute', 'step', 'answer']
        if any(keyword in answer.lower() for keyword in math_keywords):
            score += 0.2
        
        return score
    
    def _check_refinement_improvement(self, refined_answer: str, draft_answer: str) -> float:
        """
        检查细化答案是否比草稿答案更好
        """
        refined_score = self._calculate_math_reward(refined_answer)
        draft_score = self._calculate_math_reward(draft_answer)
        
        if refined_score > draft_score:
            return 0.1  # 奖励改进
        elif refined_score == draft_score:
            return 0.05  # 保持质量
        else:
            return 0.0  # 没有改进

# 测试函数
if __name__ == "__main__":
    # 测试样例
    reward_func = MathReward()
    
    test_cases = [
        ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72", "base"),
        ("The answer is 72.", "base"),
        ("I don't know how to solve this.", "base"),
        ("48 + 24 = 72\n#### 72", "refine"),
        ("", "base")
    ]
    
    print("Testing MathReward class:")
    for i, (answer, stage) in enumerate(test_cases):
        reward = reward_func([answer], stage=stage)[0]
        print(f"Case {i+1} ({stage}): {reward:.3f}")
        print(f"Text: {answer[:50]}...")
        print()
