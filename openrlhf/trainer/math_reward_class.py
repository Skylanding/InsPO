#!/usr/bin/env python3
"""
Math reasoning reward function class for OpenRLHF GRPO training.
"""

import re
from typing import List, Any, Optional

class MathReward:
    """Math reasoning reward function compatible with OpenRLHF GRPO trainer interface."""
    
    def __init__(self):
        self.name = "math_reward"
    
    def __call__(self, completions: List[str], prompt: str = "", stage: str = "base", 
                 draft: Optional[str] = None, turn: Optional[str] = None) -> List[float]:
        """
        Calculate reward scores for math answers.
        
        Args:
            completions: List of generated answers
            prompt: Original question
            stage: Stage ("base" or "refine")
            draft: Draft answer (for refine stage)
            turn: Turn information
        
        Returns:
            List[float]: Reward scores for each answer
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
        Calculate reward score for math answer.
        
        Scoring criteria:
        1. Format correctness (40%): Contains calculation steps and final answer
        2. Math correctness (60%): Based on answer format and structure
        
        Args:
            answer: Generated answer text
            prompt: Original question
            stage: Stage
            draft: Draft answer
        
        Returns:
            float: Reward score (0-1)
        """
        if not answer or not answer.strip():
            return 0.0
        
        format_score = self._check_answer_format(answer)
        math_score = self._check_math_content(answer)
        
        refine_bonus = 0.0
        if stage == "refine" and draft:
            refine_bonus = self._check_refinement_improvement(answer, draft)
        
        total_score = 0.4 * format_score + 0.6 * math_score + 0.1 * refine_bonus
        return max(0.0, min(1.0, total_score))
    
    def _check_answer_format(self, answer: str) -> float:
        """Check if answer format is correct."""
        score = 0.0
        
        if re.search(r'####\s*-?\d+(?:\.\d+)?', answer):
            score += 0.4
        
        if re.search(r'[+\-*/=]', answer):
            score += 0.3
        
        if re.search(r'\d+', answer):
            score += 0.2
        
        if len(answer.strip()) > 20:
            score += 0.1
        
        return score
    
    def _check_math_content(self, answer: str) -> float:
        """Check math content reasonableness."""
        score = 0.0
        
        math_operations = re.findall(r'[+\-*/=]', answer)
        if len(math_operations) >= 1:
            score += 0.3
        
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        if len(numbers) >= 2:
            score += 0.3
        
        if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', answer):
            score += 0.2
        
        math_keywords = ['calculate', 'solve', 'find', 'compute', 'step', 'answer']
        if any(keyword in answer.lower() for keyword in math_keywords):
            score += 0.2
        
        return score
    
    def _check_refinement_improvement(self, refined_answer: str, draft_answer: str) -> float:
        """Check if refined answer is better than draft answer."""
        refined_score = self._calculate_math_reward(refined_answer)
        draft_score = self._calculate_math_reward(draft_answer)
        
        if refined_score > draft_score:
            return 0.1
        elif refined_score == draft_score:
            return 0.05
        else:
            return 0.0

if __name__ == "__main__":
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
