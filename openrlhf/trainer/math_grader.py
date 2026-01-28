#!/usr/bin/env python3
"""
数学推理 Grader
基于 TRL GRPO 文档的奖励函数模式，专门用于数学推理任务
"""

import re
import math
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseGrader(ABC):
    """Interface for refinement graders used to compute ΔR."""
    @abstractmethod
    def score_base_batch(self, prompts: List[str], drafts: List[str]) -> List[float]:
        """Return R(x, y1) for each (x, y1)."""
        raise NotImplementedError

    @abstractmethod
    def score_refine_batch(self, prompts: List[str], drafts: List[str], refines: List[str]) -> List[float]:
        """Return R(x, y1, y2) for each (x, y1, y2)."""
        raise NotImplementedError

class MathGrader(BaseGrader):
    """
    数学推理 Grader
    基于 GSM8K 等数学数据集的格式和特点设计
    """
    
    def __init__(self):
        super().__init__()
        self.name = "math_grader"
    
    def score_base_batch(self, prompts: List[str], drafts: List[str]) -> List[float]:
        """
        评分草稿答案 (y1)
        
        Args:
            prompts: 问题列表
            drafts: 草稿答案列表
        
        Returns:
            List[float]: 每个草稿的分数
        """
        scores = []
        for prompt, draft in zip(prompts, drafts):
            score = self._score_math_answer(prompt, draft, stage="base")
            scores.append(score)
        return scores
    
    def score_refine_batch(self, prompts: List[str], drafts: List[str], refinements: List[str]) -> List[float]:
        """
        评分细化答案 (y2)
        
        Args:
            prompts: 问题列表
            drafts: 草稿答案列表
            refinements: 细化答案列表
        
        Returns:
            List[float]: 每个细化答案的分数
        """
        scores = []
        for prompt, draft, refinement in zip(prompts, drafts, refinements):
            score = self._score_math_answer(prompt, refinement, stage="refine", draft=draft)
            scores.append(score)
        return scores
    
    def _score_math_answer(self, prompt: str, answer: str, stage: str = "base", draft: Optional[str] = None) -> float:
        """
        评分数学答案
        
        评分标准：
        1. 格式正确性 (30%): 包含计算步骤和最终答案格式
        2. 数学内容 (50%): 数学运算、逻辑结构
        3. 细化改进 (20%): 如果是 refine 阶段，检查是否比 draft 更好
        
        Args:
            prompt: 原始问题
            answer: 答案文本
            stage: 阶段 ("base" 或 "refine")
            draft: 草稿答案 (用于 refine 阶段)
        
        Returns:
            float: 分数 (0-1)
        """
        if not answer or not answer.strip():
            return 0.0
        
        # 1. 格式检查 (30% 权重)
        format_score = self._check_answer_format(answer)
        
        # 2. 数学内容检查 (50% 权重)
        math_score = self._check_math_content(answer, prompt)
        
        # 3. 细化改进检查 (20% 权重)
        refine_score = 0.0
        if stage == "refine" and draft:
            refine_score = self._check_refinement_improvement(answer, draft)
        
        # 加权平均
        total_score = 0.3 * format_score + 0.5 * math_score + 0.2 * refine_score
        
        return max(0.0, min(1.0, total_score))
    
    def _check_answer_format(self, answer: str) -> float:
        """
        检查答案格式是否正确
        
        期望格式：
        - 包含最终答案 (#### 数字)
        - 有计算步骤
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
        
        # 检查是否有合理的长度
        if len(answer.strip()) > 20:
            score += 0.1
        
        return score
    
    def _check_math_content(self, answer: str, prompt: str) -> float:
        """
        检查数学内容的合理性
        """
        score = 0.0
        
        # 检查是否包含数学运算
        math_operations = re.findall(r'[+\-*/=]', answer)
        if len(math_operations) >= 1:
            score += 0.2
        
        # 检查是否有数字计算
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        if len(numbers) >= 2:
            score += 0.2
        
        # 检查是否有合理的数学表达式
        if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', answer):
            score += 0.2
        
        # 检查是否包含数学关键词
        math_keywords = ['calculate', 'solve', 'find', 'compute', 'step', 'answer', 'total', 'sum']
        if any(keyword in answer.lower() for keyword in math_keywords):
            score += 0.2
        
        # 检查是否包含问题中的数字 (显示理解问题)
        prompt_numbers = re.findall(r'\d+(?:\.\d+)?', prompt)
        if prompt_numbers:
            if any(num in answer for num in prompt_numbers):
                score += 0.2
        
        return score
    
    def _check_refinement_improvement(self, refined_answer: str, draft_answer: str) -> float:
        """
        检查细化答案是否比草稿答案更好
        """
        refined_score = self._score_math_answer("", refined_answer, stage="base")
        draft_score = self._score_math_answer("", draft_answer, stage="base")
        
        if refined_score > draft_score:
            return 1.0  # 显著改进
        elif refined_score == draft_score:
            return 0.5  # 保持质量
        else:
            return 0.0  # 没有改进

# 测试函数
if __name__ == "__main__":
    grader = MathGrader()
    
    # 测试样例
    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "What is 2 + 2?"
    ]
    
    drafts = [
        "Natalia sold 48 clips in April.",
        "2 + 2 = 4"
    ]
    
    refinements = [
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
        "2 + 2 = 4\n#### 4"
    ]
    
    print("Testing MathGrader:")
    
    # 测试草稿评分
    base_scores = grader.score_base_batch(prompts, drafts)
    print(f"Base scores: {base_scores}")
    
    # 测试细化评分
    refine_scores = grader.score_refine_batch(prompts, drafts, refinements)
    print(f"Refine scores: {refine_scores}")
