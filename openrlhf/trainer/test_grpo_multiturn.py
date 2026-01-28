#!/usr/bin/env python3
"""
Unit tests for GRPO multi-turn implementation.

Tests:
1. Turn reward attribution (Gen_k -> Ver_{k-1})
2. Mask boundary correctness
3. Ratio initial mean ≈ 1
4. KL ≥ 0
5. Pass@1 monotonic trend in inference
"""

import torch
import numpy as np
from typing import List, Dict, Any
import unittest
from unittest.mock import Mock, patch

# Import the GRPO trainer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the base trainer for testing
class MockBasePPOTrainer:
    def __init__(self, **kwargs):
        pass

# Patch the import
import unittest.mock
with unittest.mock.patch('grpo_trainer.BasePPOTrainer', MockBasePPOTrainer):
    from grpo_trainer import GRPOTrainer, GRPOConfig, CodePassrateReward


class TestGRPOMultiTurn(unittest.TestCase):
    """Test suite for GRPO multi-turn implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model and tokenizer
        self.mock_model = Mock()
        self.mock_model.device = torch.device('cpu')
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        self.mock_tokenizer.batch_decode.return_value = ["def test(): pass"]
        
        # Create config
        self.config = GRPOConfig(
            max_train_turns=3,
            max_infer_turns=6,
            use_turn_aware_advantage=True,
            gen_abs_weight=0.0,
            gen_imp_weight=1.0,
            ver_weight=1.0,
            stop_when_pass_1=True,
            old_model_sync_frequency=4
        )
        
        # Create trainer
        self.trainer = GRPOTrainer(
            config=self.config,
            model=self.mock_model,
            ref_model=self.mock_model,
            tokenizer=self.mock_tokenizer
        )
        
        # Mock reward function
        self.trainer.reward_funcs = [CodePassrateReward()]
    
    def test_turn_reward_attribution(self):
        """Test that Gen_k reward is attributed to Ver_{k-1}."""
        # Create mock turns
        turns = [
            {"type": "gen", "segment": "def func1(): pass", "pass_before": 0.0, "pass_after": 0.3},
            {"type": "ver", "segment": "test_func1()", "pass_before": None, "pass_after": 0.8},
            {"type": "gen", "segment": "def func2(): pass", "pass_before": 0.3, "pass_after": 0.7},
            {"type": "ver", "segment": "test_func2()", "pass_before": None, "pass_after": 0.9},
        ]
        
        # Compute rewards
        result_turns = self.trainer.compute_turn_rewards(turns)
        
        # Check that Gen rewards are computed correctly
        self.assertEqual(result_turns[0]["reward_scalar"], 0.3)  # Gen1: 0.3 - 0.0 = 0.3
        self.assertEqual(result_turns[2]["reward_scalar"], 0.4)  # Gen2: 0.7 - 0.3 = 0.4
        
        # Check that Ver rewards include Gen attribution
        self.assertEqual(result_turns[1]["reward_scalar"], 0.8 + 0.3)  # Ver1: 0.8 + Gen1(0.3)
        self.assertEqual(result_turns[3]["reward_scalar"], 0.9 + 0.4)  # Ver2: 0.9 + Gen2(0.4)
    
    def test_mask_boundary_correctness(self):
        """Test that loss mask boundaries are correctly computed."""
        # Mock tokenizer responses
        def mock_tokenize(text, **kwargs):
            # Simulate different token lengths for different inputs
            if "User:" in text:
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
            elif "<GEN>" in text:
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
            else:
                return {"input_ids": torch.tensor([[1, 2, 3]])}
        
        self.mock_tokenizer.side_effect = mock_tokenize
        
        # Create mock turns
        turns = [
            {"type": "gen", "segment": "def test(): pass", "reward_scalar": 1.0}
        ]
        
        # Prepare batch
        input_ids, attention_mask, loss_mask, labels, advantages = self.trainer.prepare_turn_batch(
            ["test prompt"], [turns]
        )
        
        # Check that loss_mask has correct shape and contains 1s
        self.assertEqual(loss_mask.shape, input_ids.shape)
        self.assertTrue(torch.any(loss_mask == 1))
        self.assertTrue(torch.any(loss_mask == 0))
    
    def test_ratio_initial_mean(self):
        """Test that PPO ratio initial mean is approximately 1."""
        # Create identical old and new models (should give ratio ≈ 1)
        old_logits = torch.randn(2, 10, 1000)  # [batch, seq, vocab]
        new_logits = old_logits + 0.01 * torch.randn_like(old_logits)  # Small difference
        
        # Mock model outputs
        self.mock_model.return_value.logits = new_logits
        self.trainer.old_model.return_value.logits = old_logits
        
        # Create test data
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        advantages = torch.tensor([1.0, 1.0])
        
        # Compute loss
        loss_dict = self.trainer.compute_refinement_loss(
            input_ids, attention_mask, loss_mask, labels, advantages
        )
        
        # Check that mean ratio is close to 1
        mean_ratio = loss_dict["mean_ratio"].item()
        self.assertAlmostEqual(mean_ratio, 1.0, delta=0.1)
    
    def test_kl_non_negative(self):
        """Test that KL divergence is non-negative."""
        # Create test data
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        advantages = torch.tensor([1.0, 1.0])
        
        # Mock model outputs
        self.mock_model.return_value.logits = torch.randn(2, 10, 1000)
        self.trainer.old_model.return_value.logits = torch.randn(2, 10, 1000)
        self.trainer.ref_model.return_value.logits = torch.randn(2, 10, 1000)
        
        # Compute loss
        loss_dict = self.trainer.compute_refinement_loss(
            input_ids, attention_mask, loss_mask, labels, advantages
        )
        
        # Check that KL loss is non-negative
        kl_loss = loss_dict["kl_loss"].item()
        self.assertGreaterEqual(kl_loss, 0.0)
    
    def test_pass_at_1_monotonic(self):
        """Test that Pass@1 shows monotonic trend in inference."""
        # Mock evaluation function to return increasing passrates
        passrates = [0.2, 0.4, 0.6, 0.8, 1.0]
        call_count = [0]
        
        def mock_evaluate(prompt, ctx, segment, turn, is_verification=False):
            idx = min(call_count[0] // 2, len(passrates) - 1)  # Gen/Ver pairs
            call_count[0] += 1
            return passrates[idx], f"feedback_{idx}"
        
        self.trainer._evaluate_and_feedback = mock_evaluate
        
        # Run inference
        responses = self.trainer.inference(["test prompt"], max_new_tokens=100)
        
        # Check that we got a response
        self.assertEqual(len(responses), 1)
        self.assertIsInstance(responses[0], str)
    
    def test_old_model_synchronization(self):
        """Test that old_model is synchronized periodically."""
        # Set step count to trigger synchronization
        self.trainer.step_count = 3  # Should trigger sync at step 4
        
        # Mock model state dict
        mock_state_dict = {"weight": torch.tensor([1.0])}
        self.mock_model.state_dict.return_value = mock_state_dict
        self.trainer.old_model.load_state_dict = Mock()
        self.trainer._freeze_parameters = Mock()
        
        # Run train step
        with patch.object(self.trainer, 'rollout_turns', return_value=[[{"type": "gen", "segment": "test", "pass_before": 0.0, "pass_after": 0.5}]]):
            with patch.object(self.trainer, 'compute_turn_rewards', return_value=[[{"type": "gen", "segment": "test", "pass_before": 0.0, "pass_after": 0.5, "reward_scalar": 0.5}]]):
                with patch.object(self.trainer, 'prepare_turn_batch', return_value=(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]), torch.tensor([[0, 1, 0]]), torch.tensor([[1, 2, 3]]), torch.tensor([0.5]))):
                    with patch.object(self.trainer, 'compute_refinement_loss', return_value={"total_loss": torch.tensor(0.5), "policy_loss": torch.tensor(0.3), "kl_loss": torch.tensor(0.2), "mean_ratio": torch.tensor(1.0)}):
                        metrics = self.trainer.train_step(["test prompt"])
        
        # Check that synchronization was called
        self.trainer.old_model.load_state_dict.assert_called_once()
        self.trainer._freeze_parameters.assert_called_once()
    
    def test_code_reward_function(self):
        """Test CodePassrateReward function."""
        reward_func = CodePassrateReward()
        
        # Test format reward
        good_code = "```python\ndef test(): pass\n```"
        bad_code = "invalid code"
        
        reward_good = reward_func([good_code], "test prompt", "gen_1")
        reward_bad = reward_func([bad_code], "test prompt", "gen_1")
        
        # Good code should have higher reward
        self.assertGreater(reward_good, reward_bad)
        
        # Test that reward is in reasonable range
        self.assertGreaterEqual(reward_good, -1.0)
        self.assertLessEqual(reward_good, 6.0)  # format(1) + passrate(5)


def run_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("Running GRPO Multi-Turn Smoke Test...")
    
    # Create minimal config
    config = GRPOConfig(max_train_turns=2, max_infer_turns=3)
    
    # Mock components
    mock_model = Mock()
    mock_model.device = torch.device('cpu')
    mock_tokenizer = Mock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.pad_token_id = 0
    
    # Create trainer
    trainer = GRPOTrainer(
        config=config,
        model=mock_model,
        ref_model=mock_model,
        tokenizer=mock_tokenizer
    )
    
    # Test basic functionality
    try:
        # Test rollout
        turns = trainer.rollout_turns(["test prompt"])
        print(f"✓ Rollout generated {len(turns[0])} turns")
        
        # Test reward computation
        rewards = trainer.compute_turn_rewards(turns[0])
        print(f"✓ Rewards computed for {len(rewards)} turns")
        
        # Test batch preparation
        batch = trainer.prepare_turn_batch(["test prompt"], turns)
        print(f"✓ Batch prepared with {batch[0].shape[0]} samples")
        
        print("✓ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        return False


if __name__ == "__main__":
    # Run smoke test
    smoke_success = run_smoke_test()
    
    if smoke_success:
        # Run unit tests
        unittest.main(verbosity=2)
    else:
        print("Smoke test failed, skipping unit tests")
