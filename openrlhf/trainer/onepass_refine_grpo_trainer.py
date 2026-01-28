# trainers/onepass_refine_grpo_trainer.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from torch.optim import Optimizer

from .graders.base import BaseGrader
from .grpo_trainer import GRPOTrainer, GRPOConfig

class OnePassRefineConfig(GRPOConfig):
    """
    训练期两次采样（y1,y2），推理期一次采样（仅 x->y）。
    """
    def __init__(
        self,
        draft_source: str = "ref",            # {"ref","old","current"} 生成 y1 的来源
        num_refinements_per_prompt: int = 4,  # 每个 prompt 的 y2 数量 (组内归一)
        use_sequence_level_is: bool = True,   # y2 段序列级 log-ratio
        clip_ratio_max: float = 2.0,
        kl_on_y2_only: bool = True,

        # 可选：负样本下压 (y1 作为"拒绝")
        use_negative_draft_push: bool = False,
        neg_push_coef: float = 0.25,          # policy 梯度系数（相对 ΔR 的比例）

        # 可选：一致性 KL（训练期额外一次前向）
        use_consistency_kl: bool = False,
        consistency_kl_coef: float = 0.1,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.draft_source = draft_source
        self.num_refinements_per_prompt = num_refinements_per_prompt
        self.use_sequence_level_is = use_sequence_level_is
        self.clip_ratio_max = clip_ratio_max
        self.kl_on_y2_only = kl_on_y2_only
        self.use_negative_draft_push = use_negative_draft_push
        self.neg_push_coef = neg_push_coef
        self.use_consistency_kl = use_consistency_kl
        self.consistency_kl_coef = consistency_kl_coef


class OnePassRefineGRPOTrainer(GRPOTrainer):
    """
    与 RefineGRPOTrainer 的区别：
    - 采样仍是 x->y1->y2，但优化的是 π(y|x)，即用 [x||y2] 做前向与更新；
    - 因此推理只需一次采样（x->y）。
    """
    def __init__(self,
                 grader: BaseGrader,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.config, OnePassRefineConfig), "Use OnePassRefineConfig for OnePassRefineGRPOTrainer."
        self.grader = grader

    # ---------- 采样：生成 y1 ----------
    @torch.no_grad()
    def _gen_drafts(self, prompts: List[str]) -> List[str]:
        src = self.ref_model if self.config.draft_source in ("ref","old") else self.model
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        out = src.generate(**enc,
                           max_new_tokens=self.config.max_completion_length,
                           do_sample=self.config.do_sample,
                           temperature=self.config.temperature,
                           top_p=self.config.top_p,
                           num_return_sequences=1,
                           pad_token_id=self.tokenizer.eos_token_id,
                           eos_token_id=self.tokenizer.eos_token_id)
        cut = enc['input_ids'].shape[1]
        texts = self.tokenizer.batch_decode(out[:, cut:], skip_special_tokens=True)
        return texts  # [B] 的 y1

    # ---------- 采样：在 (x,y1) 上生成多个 y2 ----------
    @torch.no_grad()
    def _gen_refinements(self, prompts: List[str], drafts: List[str]) -> List[List[str]]:
        cond = [p + d for p, d in zip(prompts, drafts)]
        enc = self.tokenizer(cond, return_tensors="pt", padding=True, truncation=True,
                             max_length=512 + self.config.max_completion_length).to(self.model.device)
        out = self.model.generate(**enc,
                                  max_new_tokens=self.config.max_completion_length,
                                  do_sample=self.config.do_sample,
                                  temperature=self.config.temperature,
                                  top_p=self.config.top_p,
                                  num_return_sequences=self.config.num_refinements_per_prompt,
                                  pad_token_id=self.tokenizer.eos_token_id,
                                  eos_token_id=self.tokenizer.eos_token_id)
        cut = enc['input_ids'].shape[1]
        texts = self.tokenizer.batch_decode(out[:, cut:], skip_special_tokens=True)
        B = len(prompts); G = self.config.num_refinements_per_prompt
        return [texts[i*G:(i+1)*G] for i in range(B)]  # [B][G]

    # ---------- 评分：ΔR ----------
    def _compute_delta_rewards(self, prompts: List[str], drafts: List[str], y2_groups: List[List[str]]) -> List[List[float]]:
        base_scores = self.grader.score_base_batch(prompts, drafts)  # len B
        all_deltas = []
        for i, (x, y1, y2s) in enumerate(zip(prompts, drafts, y2_groups)):
            refine_scores = self.grader.score_refine_batch([x]*len(y2s), [y1]*len(y2s), y2s)
            deltas = [float(r) - float(base_scores[i]) for r in refine_scores]
            all_deltas.append(deltas)
        return all_deltas

    # ---------- 打包（关键差异）：用 [x||y2] 做训练 ----------
    def _pack_onepass(self,
                      prompts: List[str],
                      y2_groups: List[List[str]],
                      deltas: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        ids_list, attn_list, mask_list, labels_list, adv_list = [], [], [], [], []

        for x, y2s, ds in zip(prompts, y2_groups, deltas):
            px = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=512).input_ids.shape[1]
            for y2, delta in zip(y2s, ds):
                text = x + y2
                enc = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True,
                                     max_length=512 + self.config.max_completion_length)
                ids = enc["input_ids"].squeeze(0)
                attn = enc["attention_mask"].squeeze(0)

                loss_mask = torch.zeros_like(ids)
                start, end = px, ids.numel()  # 仅 y2 段
                if end > start:
                    loss_mask[start:end] = 1

                ids_list.append(ids); attn_list.append(attn); mask_list.append(loss_mask)
                labels_list.append(ids); adv_list.append(float(delta))

        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id).to(self.model.device)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0).to(self.model.device)
        loss_mask = pad_sequence(mask_list, batch_first=True, padding_value=0).to(self.model.device)
        labels_full = pad_sequence(labels_list, batch_first=True, padding_value=pad_id).to(self.model.device)
        advantages = torch.tensor(adv_list, dtype=torch.float32, device=self.model.device)

        # 自回归对齐
        inputs_shift = input_ids[:, :-1]
        attn_shift = attention_mask[:, 1:]
        labels = labels_full[:, 1:]
        loss_mask = loss_mask[:, 1:]

        return inputs_shift, attn_shift, labels, loss_mask, advantages

    # ---------- y2-only 损失（在 [x||y2] 上；序列级 IS + KL@ref） ----------
    def _compute_y2_loss(self,
                         logits: torch.Tensor, ref_logits: torch.Tensor,
                         labels: torch.Tensor, loss_mask: torch.Tensor,
                         advantages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logp = F.log_softmax(logits, dim=-1)
        logq = F.log_softmax(ref_logits, dim=-1)

        lp = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
        lq = torch.gather(logq, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        loss_mask = loss_mask.float()
        active = loss_mask.sum().clamp_min(1.0)

        if getattr(self.config, "use_sequence_level_is", True):
            log_ratio = (lp - lq) * loss_mask
            log_ratio_seq = log_ratio.sum(dim=1)  # [B]
            ratio = torch.exp(log_ratio_seq)
            if self.config.clip_ratio_max:
                ratio = ratio.clamp(max=self.config.clip_ratio_max)
            adv = (advantages * self.config.gamma)
            policy_loss = -(ratio * adv).mean()
        else:
            log_ratio = (lp - lq) * loss_mask
            ratio = torch.exp(log_ratio).clamp(max=self.config.clip_ratio_max)
            adv = (advantages * self.config.gamma).unsqueeze(-1).expand_as(ratio)
            policy_loss = -(ratio * adv * loss_mask).sum() / active

        # KL（仅 y2 段）
        p = torch.exp(logp)
        kl_t = (p * (logp - logq)).sum(dim=-1) * loss_mask
        if self.config.loss_type in ("dapo", "bnpo"):
            kl_loss = kl_t.sum() / active
        elif self.config.loss_type == "dr_grpo":
            B = logits.size(0)
            kl_loss = kl_t.sum() / (B * self.config.max_completion_length)
        else:
            kl_loss = kl_t.sum() / active

        total = policy_loss + self.config.beta * kl_loss
        return total, policy_loss.detach(), kl_loss.detach()

    # ---------- 可选：负样本下压（在 [x||y1] 上，优势为 -neg_coef*ΔR） ----------
    def _neg_push_loss(self,
                       prompts: List[str],
                       drafts: List[str],
                       deltas: List[List[float]]) -> Optional[torch.Tensor]:
        if not self.config.use_negative_draft_push:
            return None

        # 取每组的平均 ΔR（也可取 max ΔR），作为 y1 的负优势权重
        neg_adv = []
        ids_list, attn_list, mask_list, labels_list = [], [], [], []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for x, y1, ds in zip(prompts, drafts, deltas):
            d = float(np.mean(ds))
            adv = - self.config.neg_push_coef * d
            text = x + y1
            enc = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True,
                                 max_length=512 + self.config.max_completion_length)
            ids = enc["input_ids"].squeeze(0)
            attn = enc["attention_mask"].squeeze(0)

            px = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=512).input_ids.shape[1]
            loss_mask = torch.zeros_like(ids)
            start, end = px, ids.numel()
            if end > start:
                loss_mask[start:end] = 1

            ids_list.append(ids); attn_list.append(attn); mask_list.append(loss_mask)
            labels_list.append(ids); neg_adv.append(adv)

        if len(ids_list) == 0:
            return None

        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id).to(self.model.device)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0).to(self.model.device)
        loss_mask = pad_sequence(mask_list, batch_first=True, padding_value=0).to(self.model.device)
        labels_full = pad_sequence(labels_list, batch_first=True, padding_value=pad_id).to(self.model.device)
        advantages = torch.tensor(neg_adv, dtype=torch.float32, device=self.model.device)

        inputs_shift = input_ids[:, :-1]
        attn_shift = attention_mask[:, 1:]
        labels = labels_full[:, 1:]
        loss_mask = loss_mask[:, 1:]

        out = self.model(input_ids=inputs_shift, attention_mask=attn_shift)
        logits = out.logits
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=inputs_shift, attention_mask=attn_shift)
            ref_logits = ref_out.logits

        total, _, _ = self._compute_y2_loss(logits, ref_logits, labels, loss_mask, advantages)
        return total

    # ---------- 可选：一致性 KL（在 y2 段，π(·|x) vs π(·|x,y1) ） ----------
    @torch.no_grad()
    def _teacher_logits_xy1(self, prompts: List[str], drafts: List[str], y2_groups: List[List[str]]):
        # 构建 [x||y1||y2] 的 teacher logits（不回传梯度）
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids_list, attn_list = [], []
        for x, y1, y2s in zip(prompts, drafts, y2_groups):
            for y2 in y2s:
                text = x + y1 + y2
                enc = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True,
                                     max_length=512 + 2*self.config.max_completion_length)
                ids_list.append(enc["input_ids"].squeeze(0)); attn_list.append(enc["attention_mask"].squeeze(0))
        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id).to(self.model.device)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0).to(self.model.device)
        inputs_shift = input_ids[:, :-1]; attn_shift = attention_mask[:, 1:]
        teacher = self.model(input_ids=inputs_shift, attention_mask=attn_shift).logits.detach()
        return teacher, input_ids  # logits@xy1, 用于对齐 y2 区间

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()

        # 读取 prompts
        if isinstance(batch, list) and "prompt" in batch[0]:
            prompts = [b["prompt"] for b in batch]
        elif "prompt" in batch:
            prompts = batch["prompt"]
        elif "input_ids" in batch:
            prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["input_ids"]]
        else:
            raise ValueError("Batch must contain 'prompt' or 'input_ids'.")

        # 采样 y1 与 y2
        drafts = self._gen_drafts(prompts)                         # [B]
        y2_groups = self._gen_refinements(prompts, drafts)         # [B][G]

        # 评分 ΔR，并组内归一
        deltas = self._compute_delta_rewards(prompts, drafts, y2_groups)
        normed = []
        for ds in deltas:
            arr = np.asarray(ds, dtype=np.float32)
            if self.config.scale_rewards == "group":
                m, s = arr.mean(), arr.std()
                arr = (arr - m) / (s + 1e-8) if s > 0 else (arr - m)
            elif self.config.scale_rewards == "batch":
                m, s = arr.mean(), arr.std()
                arr = (arr - m) / (s + 1e-8) if s > 0 else (arr - m)
            if self.config.reward_clip_range:
                lo, hi = self.config.reward_clip_range
                arr = np.clip(arr, lo, hi)
            normed.append(arr.tolist())

        # 打包为 [x||y2]（关键差异）
        inputs_shift, attn_shift, labels, loss_mask, advantages = self._pack_onepass(prompts, y2_groups, normed)

        # 前向（current/ref）——一遍式上下文
        out = self.model(input_ids=inputs_shift, attention_mask=attn_shift)
        logits = out.logits
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=inputs_shift, attention_mask=attn_shift)
            ref_logits = ref_out.logits

        # 主损失（仅 y2 段）
        total, pol_loss, kl_loss = self._compute_y2_loss(logits, ref_logits, labels, loss_mask, advantages)
        loss = total

        # 可选：一致性 KL（额外计算 π(·|x,y1) 的 teacher logits）
        if self.config.use_consistency_kl:
            with torch.no_grad():
                teacher_logits_xy1, full_xy1 = self._teacher_logits_xy1(prompts, drafts, y2_groups)
            # 为对齐 y2 段：我们也构建 [x||y2] 的 logits（上面已有 logits），在同一 token 索引处做 KL
            # 简化做法：对 labels==y2 的位置逐 token 做 KL(p_x || p_xy1)
            logp_x = F.log_softmax(logits, dim=-1)
            logp_xy1 = F.log_softmax(teacher_logits_xy1, dim=-1)  # teacher
            p_x = torch.exp(logp_x)
            # KL(p_x || p_xy1) 仅在 loss_mask==1 的地方
            kl_cons_t = (p_x * (logp_x - logp_xy1)).sum(dim=-1) * loss_mask
            kl_cons = kl_cons_t.sum() / loss_mask.sum().clamp_min(1.0)
            loss = loss + self.config.consistency_kl_coef * kl_cons
        else:
            kl_cons = torch.tensor(0.0, device=self.model.device)

        # 可选：负样本下压（在 [x||y1] 上，用 -λΔR）
        neg_loss = self._neg_push_loss(prompts, drafts, normed)
        if neg_loss is not None:
            loss = loss + neg_loss

        # 反传与优化
        self.optim.zero_grad()
        loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optim.step()
        if self.scheduler:
            self.scheduler.step()

        mean_delta = float(np.mean([np.mean(g) for g in deltas])) if len(deltas) else 0.0
        metrics = {
            "loss": loss.item(),
            "policy_loss": pol_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_deltaR": mean_delta,
            "y2_active_tokens": loss_mask.sum().item(),
            "consistency_kl": kl_cons.item() if self.config.use_consistency_kl else 0.0,
            "neg_push_loss": neg_loss.item() if neg_loss is not None else 0.0,
        }
        return metrics
