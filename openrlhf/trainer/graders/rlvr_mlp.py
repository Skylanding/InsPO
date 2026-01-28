# graders/rlvr_mlp.py
from typing import List, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseGrader

class RLVRGrader(BaseGrader):
    """
    A thin adapter that wraps a verifier callable (LLM judge / classifier).
    The callable must accept a list[str] and return list[float] scores.
    """
    def __init__(self, base_callable: Callable[[List[str]], List[float]],
                       refine_callable: Optional[Callable[[List[str]], List[float]]] = None,
                       prompt_tmpl_base: str = "Prompt:\n{X}\n\nDraft:\n{Y1}\n\nScore:",
                       prompt_tmpl_refine: str = "Prompt:\n{X}\n\nDraft:\n{Y1}\n\nRefine:\n{Y2}\n\nScore:"):
        self.base_callable = base_callable
        self.refine_callable = refine_callable or base_callable
        self.tmpl_base = prompt_tmpl_base
        self.tmpl_refine = prompt_tmpl_refine

    def score_base_batch(self, prompts: List[str], drafts: List[str]) -> List[float]:
        texts = [self.tmpl_base.format(X=x, Y1=y1) for x, y1 in zip(prompts, drafts)]
        return list(self.base_callable(texts))

    def score_refine_batch(self, prompts: List[str], drafts: List[str], refines: List[str]) -> List[float]:
        texts = [self.tmpl_refine.format(X=x, Y1=y1, Y2=y2) for x, y1, y2 in zip(prompts, drafts, refines)]
        return list(self.refine_callable(texts))


class SimpleMLPGrader(BaseGrader):
    """
    A tiny MLP grader over arbitrary feature vectors.
    You must provide a feature_fn that maps (x,y1) or (x,y1,y2) -> 1D torch.Tensor.
    """
    def __init__(self, feature_fn_base: Callable[[str, str], torch.Tensor],
                       feature_fn_refine: Callable[[str, str, str], torch.Tensor],
                       hidden_sizes=(256, 128),
                       input_dim: Optional[int] = None,
                       device: Optional[str] = None):
        super().__init__()
        self.feature_fn_base = feature_fn_base
        self.feature_fn_refine = feature_fn_refine
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Lazy MLP construction (infer input_dim on first call if not provided)
        self.input_dim = input_dim
        layers = []
        dims = [self.input_dim] + list(hidden_sizes) + [1] if self.input_dim else None
        self.net = None
        self._dims = dims  # stash for lazy build

    def _maybe_build(self, example_vec: torch.Tensor):
        if self.net is None:
            if self.input_dim is None:
                self.input_dim = example_vec.numel()
            dims = [self.input_dim, *([256, 128] if self._dims is None else self._dims[1:-1]), 1]
            layers = []
            for d_in, d_out in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(d_in, d_out))
                if d_out != 1:
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers).to(self.device)
            self.net.eval()

    @torch.no_grad()
    def score_base_batch(self, prompts: List[str], drafts: List[str]) -> List[float]:
        feats = [self.feature_fn_base(x, y1).flatten() for x, y1 in zip(prompts, drafts)]
        ex = feats[0].to(self.device)
        self._maybe_build(ex)
        X = torch.stack([f.to(self.device) for f in feats], dim=0).float()
        s = self.net(X).squeeze(-1)
        return s.cpu().tolist()

    @torch.no_grad()
    def score_refine_batch(self, prompts: List[str], drafts: List[str], refines: List[str]) -> List[float]:
        feats = [self.feature_fn_refine(x, y1, y2).flatten() for x, y1, y2 in zip(prompts, drafts, refines)]
        ex = feats[0].to(self.device)
        self._maybe_build(ex)
        X = torch.stack([f.to(self.device) for f in feats], dim=0).float()
        s = self.net(X).squeeze(-1)
        return s.cpu().tolist()
