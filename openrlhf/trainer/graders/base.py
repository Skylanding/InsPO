# graders/base.py
from abc import ABC, abstractmethod
from typing import List

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
