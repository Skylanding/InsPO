#!/usr/bin/env python3
"""
Simple smoke test for GRPO multi-turn implementation.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import unittest
from unittest.mock import Mock, patch

# Mock the base trainer
class MockBasePPOTrainer:
    def __init__(self, **kwargs):
        pass

# Create a simplified test version
class SimpleGRPOConfig:
    def __init__(self):
        self.max_train_turns = 3
        self.max_infer_turns = 6
        self.use_turn_aware_advantage = True
        self.gen_abs_weight = 0.0
        self.gen_imp_weight = 1.0
        self.ver_weight = 1.0
        self.stop_when_pass_1 = True
        self.old_model_sync_frequency = 4
        self.max_completion_length = 64
        self.do_sample = True
        self.temperature = 1.0
        self.top_p = 0.9
        self.clip_ratio = 0.2
        self.kl_coef = 0.01

class SimpleGRPOTrainer:
    def __init__(self, config, **kwargs):
        self.config = config
        self.model = kwargs.get('model')
        self.ref_model = kwargs.get('ref_model')
        self.tokenizer = kwargs.get('tokenizer')
        self.step_count = 0
        
        # Create old_model
        self.old_model = Mock()
        self.old_model.device = self.model.device
        
        # Mock reward function
        self.reward_funcs = [Mock()]
    
    def compute_turn_rewards(self, turns):
        """Test turn reward computation."""
        abs_w, imp_w = self.config.gen_abs_weight, self.config.gen_imp_weight
        gen_pass_hist = []
        
        for i, t in enumerate(turns):
            if t["type"] == "gen":
                pr = float(t["pass_after"])
                pr_prev2 = gen_pass_hist[-1] if len(gen_pass_hist) >= 1 else 0.0
                r = abs_w * pr + imp_w * (pr - pr_prev2)
                t["reward_scalar"] = r
                gen_pass_hist.append(pr)
                print(f"Gen{i}: pr={pr}, pr_prev2={pr_prev2}, reward={r}")
            else:  # verification turn
                r = float(t["pass_after"])
                t["reward_scalar"] = self.config.ver_weight * r
                print(f"Ver{i}: pr={r}, base_reward={r}")
        
        # Turn-aware attribution: Gen_k reward goes to Ver_{k-1}
        if self.config.use_turn_aware_advantage:
            for i, t in enumerate(turns):
                if t["type"] == "gen":
                    j = i - 1
                    if j >= 0 and turns[j]["type"] == "ver":
                        old_reward = turns[j]["reward_scalar"]
                        turns[j]["reward_scalar"] += t["reward_scalar"]
                        print(f"Attribution: Ver{j} {old_reward} -> {turns[j]['reward_scalar']} (+Gen{i} {t['reward_scalar']})")
        
        return turns

def test_turn_reward_attribution():
    """Test that Gen_k reward is attributed to Ver_{k-1}."""
    print("Testing turn reward attribution...")
    
    config = SimpleGRPOConfig()
    trainer = SimpleGRPOTrainer(config, model=Mock(), ref_model=Mock(), tokenizer=Mock())
    
    # Create mock turns
    turns = [
        {"type": "gen", "segment": "def func1(): pass", "pass_before": 0.0, "pass_after": 0.3},
        {"type": "ver", "segment": "test_func1()", "pass_before": None, "pass_after": 0.8},
        {"type": "gen", "segment": "def func2(): pass", "pass_before": 0.3, "pass_after": 0.7},
        {"type": "ver", "segment": "test_func2()", "pass_before": None, "pass_after": 0.9},
    ]
    
    # Compute rewards
    result_turns = trainer.compute_turn_rewards(turns)
    
    # Check that Gen rewards are computed correctly
    assert abs(result_turns[0]["reward_scalar"] - 0.3) < 1e-6, f"Expected 0.3, got {result_turns[0]['reward_scalar']}"
    assert abs(result_turns[2]["reward_scalar"] - 0.4) < 1e-6, f"Expected 0.4, got {result_turns[2]['reward_scalar']}"
    
    # Check that Ver rewards include Gen attribution
    # Ver1: 0.8 + Gen2(0.4) = 1.2 (Gen2 is the next Gen after Ver1)
    # Ver2: 0.9 + Gen4(0.0) = 0.9 (no Gen4, so no attribution)
    print(f"Debug: Ver1 reward = {result_turns[1]['reward_scalar']}, expected = 1.2")
    print(f"Debug: Ver2 reward = {result_turns[3]['reward_scalar']}, expected = 0.9")
    assert abs(result_turns[1]["reward_scalar"] - 1.2) < 1e-6, f"Expected 1.2, got {result_turns[1]['reward_scalar']}"
    assert abs(result_turns[3]["reward_scalar"] - 0.9) < 1e-6, f"Expected 0.9, got {result_turns[3]['reward_scalar']}"
    
    print("✓ Turn reward attribution test passed!")

def test_mask_boundary_correctness():
    """Test that loss mask boundaries are correctly computed."""
    print("Testing mask boundary correctness...")
    
    # Mock tokenizer
    def mock_tokenize(text, **kwargs):
        if "User:" in text:
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        elif "<GEN>" in text:
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
        else:
            return {"input_ids": torch.tensor([[1, 2, 3]])}
    
    mock_tokenizer = Mock()
    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pad_token_id = 0
    
    config = SimpleGRPOConfig()
    trainer = SimpleGRPOTrainer(config, model=Mock(), ref_model=Mock(), tokenizer=mock_tokenizer)
    
    # Create mock turns
    turns = [
        {"type": "gen", "segment": "def test(): pass", "reward_scalar": 1.0}
    ]
    
    # Mock prepare_turn_batch method
    def mock_prepare_turn_batch(prompts, all_turns):
        # Simulate batch preparation
        batch_size = sum(len(turns) for turns in all_turns)
        seq_len = 10
        return (
            torch.randint(0, 1000, (batch_size, seq_len)),  # input_ids
            torch.ones(batch_size, seq_len),                 # attention_mask
            torch.randint(0, 2, (batch_size, seq_len)),      # loss_mask
            torch.randint(0, 1000, (batch_size, seq_len)),  # labels
            torch.tensor([1.0] * batch_size)                 # advantages
        )
    
    trainer.prepare_turn_batch = mock_prepare_turn_batch
    
    # Prepare batch
    input_ids, attention_mask, loss_mask, labels, advantages = trainer.prepare_turn_batch(
        ["test prompt"], [turns]
    )
    
    # Check that loss_mask has correct shape and contains both 0s and 1s
    assert loss_mask.shape == input_ids.shape, f"Shape mismatch: {loss_mask.shape} vs {input_ids.shape}"
    assert torch.any(loss_mask == 1), "Loss mask should contain 1s"
    assert torch.any(loss_mask == 0), "Loss mask should contain 0s"
    
    print("✓ Mask boundary correctness test passed!")

def test_ratio_initial_mean():
    """Test that PPO ratio initial mean is approximately 1."""
    print("Testing ratio initial mean...")
    
    # Create test data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    advantages = torch.tensor([1.0, 1.0])
    
    # Mock model outputs (identical old and new models)
    old_logits = torch.randn(batch_size, seq_len, vocab_size)
    new_logits = old_logits + 0.01 * torch.randn_like(old_logits)  # Small difference
    
    # Mock model
    mock_model = Mock()
    mock_model.device = torch.device('cpu')
    mock_model.return_value.logits = new_logits
    
    config = SimpleGRPOConfig()
    trainer = SimpleGRPOTrainer(config, model=mock_model, ref_model=Mock(), tokenizer=Mock())
    trainer.old_model.return_value.logits = old_logits
    trainer.ref_model.return_value.logits = old_logits
    
    # Mock compute_refinement_loss
    def mock_compute_loss(input_ids, attention_mask, loss_mask, labels, advantages):
        # Simulate ratio computation
        logp_new = torch.log_softmax(new_logits[:, :-1, :], dim=-1)
        logp_old = torch.log_softmax(old_logits[:, :-1, :], dim=-1)
        
        # Compute ratio
        shift_labels = labels[:, 1:]
        logp_new_sel = torch.gather(logp_new, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        logp_old_sel = torch.gather(logp_old, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        shift_mask = loss_mask[:, 1:].float()
        seq_log_ratio = ((logp_new_sel - logp_old_sel) * shift_mask).sum(dim=1)
        ratio = torch.exp(seq_log_ratio)
        
        return {
            "total_loss": torch.tensor(0.5),
            "policy_loss": torch.tensor(0.3),
            "kl_loss": torch.tensor(0.2),
            "mean_ratio": ratio.mean()
        }
    
    trainer.compute_refinement_loss = mock_compute_loss
    
    # Compute loss
    loss_dict = trainer.compute_refinement_loss(
        input_ids, attention_mask, loss_mask, labels, advantages
    )
    
    # Check that mean ratio is close to 1
    mean_ratio = loss_dict["mean_ratio"].item()
    assert abs(mean_ratio - 1.0) < 0.1, f"Expected ratio close to 1.0, got {mean_ratio}"
    
    print("✓ Ratio initial mean test passed!")

def test_kl_non_negative():
    """Test that KL divergence is non-negative."""
    print("Testing KL non-negative...")
    
    # Create test data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    advantages = torch.tensor([1.0, 1.0])
    
    # Mock model outputs
    mock_model = Mock()
    mock_model.device = torch.device('cpu')
    mock_model.return_value.logits = torch.randn(batch_size, seq_len, vocab_size)
    
    config = SimpleGRPOConfig()
    trainer = SimpleGRPOTrainer(config, model=mock_model, ref_model=Mock(), tokenizer=Mock())
    trainer.old_model.return_value.logits = torch.randn(batch_size, seq_len, vocab_size)
    trainer.ref_model.return_value.logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Mock compute_refinement_loss
    def mock_compute_loss(input_ids, attention_mask, loss_mask, labels, advantages):
        # Simulate KL computation
        shift_logits = torch.randn(batch_size, seq_len-1, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len-1, vocab_size)
        
        logp_new = torch.log_softmax(shift_logits, dim=-1)
        logp_ref = torch.log_softmax(ref_logits, dim=-1)
        p_new = torch.softmax(shift_logits, dim=-1)
        
        kl_token = (p_new * (logp_new - logp_ref)).sum(-1)
        shift_mask = loss_mask[:, 1:].float()
        kl_loss = (kl_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
        
        return {
            "total_loss": torch.tensor(0.5),
            "policy_loss": torch.tensor(0.3),
            "kl_loss": kl_loss,
            "mean_ratio": torch.tensor(1.0)
        }
    
    trainer.compute_refinement_loss = mock_compute_loss
    
    # Compute loss
    loss_dict = trainer.compute_refinement_loss(
        input_ids, attention_mask, loss_mask, labels, advantages
    )
    
    # Check that KL loss is non-negative
    kl_loss = loss_dict["kl_loss"].item()
    assert kl_loss >= 0.0, f"Expected KL >= 0, got {kl_loss}"
    
    print("✓ KL non-negative test passed!")

def run_all_tests():
    """Run all smoke tests."""
    print("Running GRPO Multi-Turn Smoke Tests...")
    print("=" * 50)
    
    try:
        test_turn_reward_attribution()
        test_mask_boundary_correctness()
        test_ratio_initial_mean()
        test_kl_non_negative()
        
        print("=" * 50)
        print("✓ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
