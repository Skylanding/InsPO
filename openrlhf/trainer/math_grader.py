#!/usr/bin/env python3
"""
Math reasoning grader based on TRL GRPO reward function pattern for math reasoning tasks.
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
    """Math reasoning grader designed for GSM8K and similar math datasets."""
    
    def __init__(self):
        super().__init__()
        self.name = "math_grader"
    
    def score_base_batch(self, prompts: List[str], drafts: List[str]) -> List[float]:
        """
        Score draft answers (y1).
        
        Args:
            prompts: List of questions
            drafts: List of draft answers
        
        Returns:
            List[float]: Scores for each draft
        """
        scores = []
        for prompt, draft in zip(prompts, drafts):
            score = self._score_math_answer(prompt, draft, stage="base")
            scores.append(score)
        return scores
    
    def score_refine_batch(self, prompts: List[str], drafts: List[str], refinements: List[str]) -> List[float]:
        """
        Score refined answers (y2).
        
        Args:
            prompts: List of questions
            drafts: List of draft answers
            refinements: List of refined answers
        
        Returns:
            List[float]: Scores for each refined answer
        """
        scores = []
        for prompt, draft, refinement in zip(prompts, drafts, refinements):
            score = self._score_math_answer(prompt, refinement, stage="refine", draft=draft)
            scores.append(score)
        return scores
    
    def _score_math_answer(self, prompt: str, answer: str, stage: str = "base", draft: Optional[str] = None) -> float:
        """
        Score math answer.
        
        Scoring criteria:
        1. Format correctness (30%): Contains calculation steps and final answer format
        2. Math content (50%): Math operations, logical structure
        3. Refinement improvement (20%): If refine stage, check if better than draft
        
        Args:
            prompt: Original question
            answer: Answer text
            stage: Stage ("base" or "refine")
            draft: Draft answer (for refine stage)
        
        Returns:
            float: Score (0-1)
        """
        if not answer or not answer.strip():
            return 0.0
        
        format_score = self._check_answer_format(answer)
        math_score = self._check_math_content(answer, prompt)
        
        refine_score = 0.0
        if stage == "refine" and draft:
            refine_score = self._check_refinement_improvement(answer, draft)
        
        total_score = 0.3 * format_score + 0.5 * math_score + 0.2 * refine_score
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
    
    def _check_math_content(self, answer: str, prompt: str) -> float:
        """Check math content reasonableness."""
        score = 0.0
        
        math_operations = re.findall(r'[+\-*/=]', answer)
        if len(math_operations) >= 1:
            score += 0.2
        
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        if len(numbers) >= 2:
            score += 0.2
        
        if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', answer):
            score += 0.2
        
        math_keywords = ['calculate', 'solve', 'find', 'compute', 'step', 'answer', 'total', 'sum']
        if any(keyword in answer.lower() for keyword in math_keywords):
            score += 0.2
        
        prompt_numbers = re.findall(r'\d+(?:\.\d+)?', prompt)
        if prompt_numbers:
            if any(num in answer for num in prompt_numbers):
                score += 0.2
        
        return score
    
    def _check_refinement_improvement(self, refined_answer: str, draft_answer: str) -> float:
        """Check if refined answer is better than draft answer."""
        refined_score = self._score_math_answer("", refined_answer, stage="base")
        draft_score = self._score_math_answer("", draft_answer, stage="base")
        
        if refined_score > draft_score:
            return 1.0
        elif refined_score == draft_score:
            return 0.5
        else:
            return 0.0

if __name__ == "__main__":
    grader = MathGrader()
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
    
    base_scores = grader.score_base_batch(prompts, drafts)
    print(f"Base scores: {base_scores}")
    
    refine_scores = grader.score_refine_batch(prompts, drafts, refinements)
    print(f"Refine scores: {refine_scores}")
