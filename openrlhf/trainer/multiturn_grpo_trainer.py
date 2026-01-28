"""
Multi-Turn GRPO Trainer

基于现有 GRPO 训练器的多回合生成↔验证训练器
实现 Gen₁→Ver₁→Gen₂→Ver₂→... 的多回合训练机制
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel
import numpy as np
import copy

from .grpo_trainer import GRPOTrainer, GRPOConfig


@dataclass
class MultiTurnGRPOConfig(GRPOConfig):
    """多回合 GRPO 配置"""
    # 多回合参数
    max_train_turns: int = 3  # 训练最多回合数（Gen/Ver 各算一回合）
    max_infer_turns: int = 6  # 推理可扩深
    use_turn_aware_advantage: bool = True  # Turn-Aware PPO
    gen_abs_weight: float = 0.0  # r_abs 权重（论文默认 0）
    gen_imp_weight: float = 1.0  # r_imp 权重（论文默认 1）
    ver_weight: float = 1.0  # 验证回合权重
    stop_when_pass_1: bool = True  # 通过率=1 提前停止
    
    # Monte Carlo 合并（简单拷贝标量回合回报到该回合所有 token）
    gamma: float = 1.0
    lam: float = 1.0
    
    # 行为策略 & 参考策略
    use_old_model_snapshot: bool = True  # PPO 分母：显式旧策略快照
    old_model_sync_frequency: int = 4  # 每N步同步old_model


class MultiTurnGRPOTrainer(GRPOTrainer):
    """
    多回合 GRPO 训练器
    
    基于现有 GRPO 训练器，扩展支持多回合生成↔验证训练
    """
    
    def __init__(self, model, ref_model, tokenizer, config: MultiTurnGRPOConfig, strategy=None, **kwargs):
        """
        初始化多回合 GRPO 训练器
        
        Args:
            model: 主模型
            ref_model: 参考模型
            tokenizer: 分词器
            config: 多回合 GRPO 配置
            strategy: 训练策略
        """
        # 使用父类初始化，但使用多回合配置
        super().__init__(model, ref_model, tokenizer, config, strategy, **kwargs)
        
        # 多回合特定初始化
        self.multiturn_config = config
        self.turn_tags = {
            "gen_open": "<GEN>\n",
            "gen_close": "</GEN>\n",
            "ver_open": "<VER>\n",
            "ver_close": "</VER>\n",
            "tool_feedback_open": "<TOOL>\n",
            "tool_feedback_close": "</TOOL>\n",
        }
        
        # 回合计数器
        self.turn_count = 0
        
    def rollout_turns(self, prompts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        多回合生成↔验证 rollout
        
        Returns:
            List[List[Dict]]: 每个 prompt 的回合序列
            turns[p] = [
                {"type":"gen","segment":"…y_gen1…","pass_before":0.0,"pass_after":p1},
                {"type":"ver","segment":"…y_ver1…","pass_before":p1,"pass_after":p1'},
                {"type":"gen","segment":"…y_gen2…","pass_before":p1',"pass_after":p2},
                ...
            ]
        """
        all_prompts_turns = []
        
        for x in prompts:
            ctx = f"User:\n{x}\n\n"
            pass_hist = []  # 仅存 Gen 回合后的 passrate (用于 improvement)
            turns = []
            
            for k in range(self.multiturn_config.max_train_turns):
                # ---- Gen_k ----
                gen_in = ctx + self.turn_tags["gen_open"]
                yk = self._sample(
                    self.model if k > 0 else self.ref_model, 
                    gen_in,
                    max_new_tokens=self.config.max_completion_length
                )
                ctx += self.turn_tags["gen_open"] + yk + self.turn_tags["gen_close"]
                
                # 执行器：生成/复用测试 + 运行，得到 passrate_k
                pr_k, tool_fb = self._evaluate_and_feedback(x, ctx, yk, turn=f"gen_{k+1}")
                turns.append(dict(
                    type="gen", 
                    segment=yk,
                    pass_before=pass_hist[-1] if pass_hist else 0.0,
                    pass_after=pr_k,
                    tool_feedback=tool_fb
                ))
                pass_hist.append(pr_k)
                
                if self.multiturn_config.stop_when_pass_1 and pr_k >= 1.0:
                    break
                
                # ---- Ver_k ----
                ver_in = ctx + self.turn_tags["ver_open"]
                vk = self._sample(
                    self.model, 
                    ver_in,
                    max_new_tokens=self.config.max_completion_length
                )
                ctx += self.turn_tags["ver_open"] + vk + self.turn_tags["ver_close"]
                
                # 验证回合：产测试（或解释）+ 执行得到本回合有效通过率
                prk_ver, tool_fb2 = self._evaluate_and_feedback(
                    x, ctx, vk, turn=f"ver_{k+1}", is_verification=True
                )
                turns.append(dict(
                    type="ver", 
                    segment=vk,
                    pass_before=None,
                    pass_after=prk_ver,
                    tool_feedback=tool_fb2
                ))
            
            all_prompts_turns.append(turns)
        
        return all_prompts_turns
    
    def compute_turn_rewards(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        计算回合级奖励
        
        Args:
            turns: 单个 prompt 的回合列表
            
        Returns:
            更新了 reward_scalar 的回合列表
        """
        abs_w, imp_w = self.multiturn_config.gen_abs_weight, self.multiturn_config.gen_imp_weight
        gen_pass_hist = []
        
        for i, t in enumerate(turns):
            if t["type"] == "gen":
                pr = float(t["pass_after"])
                pr_prev2 = gen_pass_hist[-1] if len(gen_pass_hist) >= 1 else 0.0
                r = abs_w * pr + imp_w * (pr - pr_prev2)
                t["reward_scalar"] = r
                gen_pass_hist.append(pr)
            else:
                r = float(t["pass_after"])
                t["reward_scalar"] = self.multiturn_config.ver_weight * r
        
        # Turn-aware 回注：把 Gen_k 的奖励也加到 Ver_{k-1}
        if self.multiturn_config.use_turn_aware_advantage:
            for i, t in enumerate(turns):
                if t["type"] == "gen":
                    j = i - 1
                    if j >= 0 and turns[j]["type"] == "ver":
                        turns[j]["reward_scalar"] += t["reward_scalar"]
        
        return turns
    
    def prepare_turn_batch(self, prompts: List[str], all_turns: List[List[Dict[str, Any]]]) -> Tuple:
        """
        为所有 prompt 的所有回合构建训练样本
        
        Returns:
            input_ids, attention_mask, loss_mask, labels, advantages
        """
        ids_list, attn_list, mask_list, labels_list, adv_list = [], [], [], [], []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        MAX = 512 + self.config.max_completion_length
        
        for x, turns in zip(prompts, all_turns):
            ctx_left = f"User:\n{x}\n\n"
            
            for t in turns:
                # 1) 构建上下文 + 当前回合段（严格角色化）
                if t["type"] == "gen":
                    seg_open, seg_close = self.turn_tags["gen_open"], self.turn_tags["gen_close"]
                else:
                    seg_open, seg_close = self.turn_tags["ver_open"], self.turn_tags["ver_close"]
                
                # 输入 = (ctx_left + seg_open) → 生成 seg → seg_close 只出现在 labels
                inp_text = ctx_left + seg_open
                seg_text = t["segment"]
                full_text = inp_text + seg_text + seg_close
                
                ids_full = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
                ids_inp = self.tokenizer(inp_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
                
                start = ids_inp.numel()
                end = start + self.tokenizer(seg_text, add_special_tokens=False, return_tensors="pt").input_ids[0].numel()
                
                # 左截断以适应 MAX
                if ids_full.numel() > MAX:
                    cut = ids_full.numel() - MAX
                    ids_full = ids_full[cut:]
                    start = max(0, start - cut)
                    end = max(start, end - cut)
                
                attn = torch.ones_like(ids_full)
                loss_mask = torch.zeros_like(ids_full)
                loss_mask[start:end] = 1
                
                ids_list.append(ids_full)
                attn_list.append(attn)
                labels_list.append(ids_full.clone())
                mask_list.append(loss_mask)
                adv_list.append(float(t["reward_scalar"]))
                
                # 2) 更新 ctx_left：把"当前回合完整段"并上 TOOL 反馈
                ctx_left = full_text
                if t.get("tool_feedback"):
                    ctx_left += (self.turn_tags["tool_feedback_open"] + 
                               t["tool_feedback"] + 
                               self.turn_tags["tool_feedback_close"])
        
        # pad & stack & device
        max_len = max(x.numel() for x in ids_list)
        B = len(ids_list)
        
        def pad_stack(lst, fill):
            out = torch.full((B, max_len), fill, dtype=torch.long)
            for i, x in enumerate(lst):
                out[i, :x.numel()] = x
            return out
        
        input_ids = pad_stack(ids_list, pad_id)
        attention_mask = pad_stack(attn_list, 0)
        loss_mask = pad_stack(mask_list, 0)
        labels = pad_stack(labels_list, pad_id)
        advantages = torch.tensor(adv_list, dtype=torch.float32)
        
        device = self.model.device
        return (input_ids.to(device), attention_mask.to(device),
                loss_mask.to(device), labels.to(device), advantages.to(device))
    
    def train_step(self, prompts: List[str]) -> Dict[str, Any]:
        """
        多回合训练步骤
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            训练指标字典
        """
        # 1) rollout 多回合
        all_turns = self.rollout_turns(prompts)
        
        # 2) 回合级奖励
        all_turns = [self.compute_turn_rewards(t) for t in all_turns]
        
        # 3) 构建回合段样本
        batch = self.prepare_turn_batch(prompts, all_turns)
        
        # 4) 计算损失（使用父类的 compute_refinement_loss）
        loss_dict = self.compute_refinement_loss(*batch)
        loss_dict["total_loss"].backward()
        
        # 5) 同步 old_model
        if self.step_count % self.multiturn_config.old_model_sync_frequency == 0:
            self._sync_old_model()
        
        # 6) 统计
        mean_adv = batch[-1].mean().item()
        return dict(
            total_loss=loss_dict["total_loss"].item(),
            policy_loss=loss_dict["policy_loss"].item(),
            kl_loss=loss_dict["kl_loss"].item(),
            mean_ratio=loss_dict["mean_ratio"].item(),
            mean_advantage=mean_adv,
            active_tokens=int((batch[2][:,1:].float()).sum().item()),
            turn_count=self.turn_count,
        )
    
    def inference(self, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
        """
        多回合推理（Test-time scaling）
        
        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大新token数
            
        Returns:
            生成的响应列表
        """
        responses = []
        
        for x in prompts:
            ctx = f"User:\n{x}\n\n"
            best_code, best_pass = "", 0.0
            
            for k in range(self.multiturn_config.max_infer_turns):
                # Gen_k
                gen_in = ctx + self.turn_tags["gen_open"]
                yk = self._sample(self.model, gen_in, max_new_tokens=max_new_tokens//2)
                ctx += self.turn_tags["gen_open"] + yk + self.turn_tags["gen_close"]
                
                pr_k, tool_fb = self._evaluate_and_feedback(x, ctx, yk, turn=f"gen_{k+1}")
                if pr_k > best_pass:
                    best_pass, best_code = pr_k, yk
                if self.multiturn_config.stop_when_pass_1 and pr_k >= 1.0:
                    break
                
                # Ver_k
                ver_in = ctx + self.turn_tags["ver_open"]
                vk = self._sample(self.model, ver_in, max_new_tokens=max_new_tokens//2)
                ctx += self.turn_tags["ver_open"] + vk + self.turn_tags["ver_close"]
                
                prk_ver, tool_fb2 = self._evaluate_and_feedback(
                    x, ctx, vk, turn=f"ver_{k+1}", is_verification=True
                )
                
                # tool 反馈可拼到 ctx，作为下轮提示
                ctx += (self.turn_tags["tool_feedback_open"] + tool_fb + tool_fb2 +
                       self.turn_tags["tool_feedback_close"])
            
            responses.append(best_code if best_code else yk)
        
        return responses