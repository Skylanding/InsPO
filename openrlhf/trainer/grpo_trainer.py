"""
GRPO Trainer with Refinement Logic

Implements Generalized Reinforcement Preference Optimization (GRPO) with refinement-based training.
The key idea is to generate draft actions (y1) first, then generate refined actions (y2) conditioned on (x, y1),
and use refinement advantage for PPO/GRPO updates.

Mathematical formulation:
- Refinement advantage: A(x,y1,y2) = R(x,y1,y2) - R(x,y1)
- Conditional ratio: r(θ) = π_θ(y2|x,y1) / π_θ_old(y2|x,y1)
- PPO-Clip objective: L_CLIP(θ) applied to refinement policy π(y2|x,y1)
"""

import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel
import numpy as np
import copy

from .ppo_trainer import BasePPOTrainer


class CodePassrateReward:
    """
    Code execution reward function with passrate and format scoring.
    """
    def __init__(self, executor=None, use_gold_filter: bool = True):
        self.exec = executor
        self.use_gold_filter = use_gold_filter
    
    def _format_ok(self, code: str) -> bool:
        """Check if code has proper formatting (e.g., fenced code blocks)"""
        # Simple heuristic: check for code fences or proper indentation
        return "```" in code or code.strip().startswith(("def ", "class ", "import ", "from "))
    
    def _build_or_reuse_tests(self, prompt: str, draft: str = None, turn: str = None) -> List[str]:
        """Build or reuse test cases for code evaluation"""
        # Placeholder: return dummy tests
        return ["test_case_1", "test_case_2"]
    
    def _has_gold(self, prompt: str) -> bool:
        """Check if prompt has gold standard answer"""
        # Placeholder: assume no gold standard
        return False
    
    def _filter_tests_with_gold(self, tests: List[str], prompt: str) -> List[str]:
        """Filter tests using gold standard"""
        # Placeholder: return all tests
        return tests
    
    def __call__(self, candidates: List[str], prompt: str, stage: str,
                 draft: Optional[str] = None, turn: Optional[str] = None) -> float:
        """
        Compute reward for code candidates.
        
        Args:
            candidates: List of generated code candidates
            prompt: Input prompt
            stage: Stage identifier {"base","refine","gen_k","ver_k"}
            draft: Previous draft (for refinement)
            turn: Turn identifier
            
        Returns:
            float: Reward score
        """
        code = candidates[0]
        
        # 1) Format reward
        fmt = self._format_ok(code)
        r_format = 1.0 if fmt else -1.0
        
        # 2) Passrate reward
        tests = self._build_or_reuse_tests(prompt, draft, turn)
        if self.use_gold_filter and self._has_gold(prompt):
            tests = self._filter_tests_with_gold(tests, prompt)
        
        # Placeholder: simulate passrate
        import random
        passrate = random.uniform(0.0, 1.0)
        
        # Scale passrate by 5 as in paper
        return 1.0 * r_format + 5.0 * passrate

# Turn tags for structured generation
TURN_TAGS = dict(
    gen_open = "<GEN>\n",
    gen_close = "</GEN>\n",
    ver_open = "<VER>\n",
    ver_close = "</VER>\n",
    tool_feedback_open = "<TOOL>\n",
    tool_feedback_close = "</TOOL>\n",
)


@dataclass
class GRPOConfig:
    """Configuration for GRPO with refinement logic"""
    # Draft generation parameters
    draft_source: str = "ref"  # {"ref", "old", "current"} - source strategy for generating y1
    num_drafts_per_prompt: int = 1
    num_refinements_per_draft: int = 4  # G - number of refinements per draft
    max_completion_length: int = 64
    # Whether to mask tokens that were truncated from generation when computing loss
    mask_truncated_completions: bool = False
    
    # Refinement advantage parameters
    use_sequence_level_is: bool = True  # Use sequence-level importance sampling for y2 segment
    use_refinement_advantage: bool = True  # Use ΔR = R(x,y1,y2) - R(x,y1) as advantage
    
    # Training parameters
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    scale_rewards: str = "group"  # Group normalization for refinement advantages
    
    # Generation parameters
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.9
    
    # Loss computation
    importance_sampling_level: str = "sequence"  # {"sequence", "token"}
    kl_stable: bool = True  # Use stable KL computation
    
    # Loss computation parameters
    loss_type: str = "dr_grpo"  # {"dr_grpo", "dapo", "bnpo"}
    
    # Reward scaling
    reward_clip_min: float = -5.0
    reward_clip_max: float = 5.0
    # Select reward function behavior (string identifier)
    reward_function: str = "hardware_verification"
    
    # Model parameters
    model_name_or_path: str = ""
    ref_model_name_or_path: str = ""
    dataset_name: str = ""
    output_dir: str = "./checkpoint"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 250
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    report_to: str = "wandb"
    wandb_project: str = "grpo"
    wandb_run_name: str = "grpo_run"
    bf16: bool = True
    gradient_checkpointing: bool = True
    flash_attn: bool = True
    max_len: int = 2048
    num_completions_per_prompt: int = 4
    seed: int = 42


class GRPOTrainer:
    """
    GRPO Trainer with refinement-based training logic.
    
    Key features:
    1. Two-stage generation: draft (y1) -> refinement (y2)
    2. Refinement advantage: ΔR = R(x,y1,y2) - R(x,y1)
    3. Sequence-level importance sampling for y2 segment
    4. Group normalization for refinement advantages
    """
    
    def __init__(self, model, ref_model, tokenizer, config: GRPOConfig, strategy=None, **kwargs):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.strategy = strategy
        # Optional components from caller
        self.train_dataloader = kwargs.get('train_dataloader')
        self.eval_dataloader = kwargs.get('eval_dataloader')
        self.scheduler = kwargs.get('scheduler')
        
        # Initialize behavior policy (old policy) for PPO ratio computation
        self.old_model = kwargs.get('old_model')
        self._old_model_initialized = False
        
        # Step counter for old_model synchronization
        self.step_count = 0
        
        # Initialize reward functions based on reward_mode
        self.reward_funcs = self._initialize_reward_functions()
    
    def _initialize_reward_functions(self):
        """
        根据配置初始化奖励函数
        
        Returns:
            List: 奖励函数列表
        """
        # 获取奖励模式，默认为 code
        reward_mode = getattr(self.config, 'reward_function', 'hardware_verification')
        
        # 使用奖励函数工厂创建奖励函数
        try:
            from .reward_factory import RewardFactory
            return RewardFactory.create_reward_functions(reward_mode)
        except ImportError:
            print("Warning: RewardFactory not found, using default CodePassrateReward")
            return [CodePassrateReward()]
    
    def _ensure_old_model_initialized(self):
        """Ensure old_model is properly initialized"""
        if self.old_model is None or not self._old_model_initialized:
            # Build a fresh base model (unwrap DeepSpeed/FSDP .module and Actor wrappers)
            base_model = getattr(self.model, "module", self.model)
            hf_model = getattr(base_model, "model", base_model)
            
            # Instantiate the same HF model class using its config
            model_cls = hf_model.__class__
            hf_config = hf_model.config
            self.old_model = model_cls(hf_config)
            
            # For ZeRO-3, skip loading weights initially and use random weights
            # The old_model will be synced later during training
            
            # Match dtype/device of the HF model
            base_dtype = next(hf_model.parameters()).dtype
            base_device = next(hf_model.parameters()).device
            self.old_model.to(base_dtype)
            self.old_model.to(base_device)
            self._freeze_parameters(self.old_model)
            self._old_model_initialized = True
        
        # Initialize optimizer
        if self.strategy and hasattr(self.strategy, "create_optimizer"):
            self.optimizer = self.strategy.create_optimizer(
                getattr(self.model, "model", self.model),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
            )
        else: 
            import torch.optim as optim
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, betas=(0.9, 0.95))

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is not provided for GRPOTrainer")
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        # Initialize old_model before training starts
        self._ensure_old_model_initialized()

        start_epoch = consumed_samples // args.train_batch_size // max(1, num_update_steps_per_epoch)
        step = consumed_samples // args.train_batch_size * max(1, getattr(self.strategy, 'accumulated_gradient', 1)) + 1
        epoch_bar = tqdm(range(start_epoch, args.max_epochs), desc="Train epoch", disable=not (self.strategy and self.strategy.is_rank_0()))
        for epoch in range(start_epoch, args.max_epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()), desc=f"Train step of epoch {epoch}", disable=not (self.strategy and self.strategy.is_rank_0()))
            for batch in self.train_dataloader:
                # derive prompts
                if isinstance(batch, list) and batch and isinstance(batch[0], dict) and 'prompt' in batch[0]:
                    prompts = [b['prompt'] for b in batch]
                elif isinstance(batch, dict) and 'prompt' in batch:
                    prompts = batch['prompt']
                else:
                    prompts = batch

                metrics = self.train_step(prompts)

                # step
                if self.strategy is not None:
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                else:
                    self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)

                # Sync old_model periodically
                if self.step_count % self.config.old_model_sync_frequency == 0:
                    self._sync_old_model()

                self.step_count += 1
                if self.strategy and self.strategy.is_rank_0():
                    step_bar.set_postfix(metrics)
                step_bar.update(); step += 1
            epoch_bar.update()
        epoch_bar.close()
        
    def generate_draft_and_refinements(self, prompts: List[str]) -> List[List[Tuple[str, List[str]]]]:
        """
        Generate drafts (y1) and refinements (y2) for each prompt.
        
        Returns:
            List[List[Tuple[str, List[str]]]]: Structure: [prompt][draft] -> (y1, [y2_i])
        """
        # 1) Generate drafts y1
        src_model = {
            "ref": self.ref_model, 
            "old": self.ref_model, 
            "current": self.model
        }[self.config.draft_source]
        
        with torch.no_grad():
            enc = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.model.device)
            
            # Unwrap DeepSpeed engine to get the underlying model
            actual_src_model = getattr(src_model, 'module', src_model)
            actual_src_model = getattr(actual_src_model, 'model', actual_src_model)
            
            y1_out = actual_src_model.generate(
                **enc,
                max_new_tokens=self.config.max_completion_length,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=self.config.num_drafts_per_prompt
            )
        
        B = len(prompts)
        D = self.config.num_drafts_per_prompt
        y1_texts = self.tokenizer.batch_decode(
            y1_out[:, enc['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        y1_groups = [y1_texts[i*D:(i+1)*D] for i in range(B)]
        
        # 2) Generate refinements y2 for each (x, y1) - batch processing
        all_y2_groups = []
        for i, x in enumerate(prompts):
            y2_list_per_prompt = []
            
            # Batch all y1 texts for this prompt
            y1_texts_for_prompt = y1_groups[i]
            if not y1_texts_for_prompt:
                all_y2_groups.append([])
                continue
                
            # Prepare batch inputs
            batch_texts = []
            for y1 in y1_texts_for_prompt:
                cond_text = x + y1
                batch_texts.append(cond_text)
            
            # Batch tokenization
            batch_cond = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512 + self.config.max_completion_length
            ).to(self.model.device)
            
            with torch.no_grad():
                # Unwrap DeepSpeed engine to get the underlying model
                actual_model = getattr(self.model, 'module', self.model)
                actual_model = getattr(actual_model, 'model', actual_model)
                
                # Generate for entire batch
                y2_out = actual_model.generate(
                    **batch_cond,
                    max_new_tokens=self.config.max_completion_length,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_return_sequences=self.config.num_refinements_per_draft,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and organize results
            y2_texts = self.tokenizer.batch_decode(
                y2_out[:, batch_cond['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Group results by draft
            refs_per_draft = self.config.num_refinements_per_draft
            for j, y1 in enumerate(y1_texts_for_prompt):
                start_idx = j * refs_per_draft
                end_idx = start_idx + refs_per_draft
                y2_for_this_draft = y2_texts[start_idx:end_idx]
                y2_list_per_prompt.append((y1, y2_for_this_draft))
            
            all_y2_groups.append(y2_list_per_prompt)
        
        return all_y2_groups
    
    def compute_refinement_rewards(self, prompts: List[str], y1y2_groups: List[List[Tuple[str, List[str]]]]) -> List[List[List[float]]]:
        """
        Compute refinement advantages: ΔR = R(x,y1,y2) - R(x,y1)
        
        Args:
            prompts: List of input prompts
            y1y2_groups: Generated drafts and refinements
            
        Returns:
            List[List[List[float]]]: Refinement advantages [prompt][draft][refinement]
        """
        all_deltas = []  # [P][D][G]
        
        for x, draft_pack in zip(prompts, y1y2_groups):
            deltas_this_prompt = []
            for (y1, y2_list) in draft_pack:
                # Evaluate R(x,y1) once as baseline
                base_reward = 0.0
                if getattr(self, 'reward_funcs', None):
                    for rf in self.reward_funcs:
                        r = rf([y1], prompt=x, stage="base")
                        base_reward += (r[0] if isinstance(r, (list, tuple)) else r)
                
                # Evaluate R(x,y1,y2) for each y2 and compute difference
                deltas = []
                for y2 in y2_list:
                    refine_reward = 0.0
                    if getattr(self, 'reward_funcs', None):
                        for rf in self.reward_funcs:
                            r = rf([y2], prompt=x, draft=y1, stage="refine")
                            refine_reward += (r[0] if isinstance(r, (list, tuple)) else r)
                    
                    delta = refine_reward - base_reward
                    deltas.append(delta)
                
                # Group normalization within same (x,y1) pair
                if self.config.scale_rewards == "group":
                    g = np.array(deltas, dtype=np.float32)
                    std = g.std() if g.std() > 1e-8 else 1.0
                    g = (g - g.mean()) / std
                    deltas = g.tolist()
                
                deltas_this_prompt.append(deltas)
            all_deltas.append(deltas_this_prompt)
        
        return all_deltas
    
    def _freeze_parameters(self, model):
        """Freeze all parameters in the model"""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    def _sync_old_model(self):
        """Synchronize old_model with current model"""
        if not self._old_model_initialized:
            return
            
        try:
            # Get the underlying HF model from DeepSpeed engine
            base_model = getattr(self.model, "module", self.model)
            hf_model = getattr(base_model, "model", base_model)
            
            # For ZeRO-3, use a different approach to get model weights
            if hasattr(self.model, '_consolidated_16bit_state_dict'):
                # Try consolidated state dict first
                try:
                    consolidated_dict = self.model._consolidated_16bit_state_dict()
                    if consolidated_dict is not None:
                        self.old_model.load_state_dict(consolidated_dict, strict=False)
                        return
                except:
                    pass
            
            # Fallback: copy parameters directly
            try:
                for old_param, hf_param in zip(self.old_model.parameters(), hf_model.parameters()):
                    if old_param.shape == hf_param.shape:
                        old_param.data.copy_(hf_param.data)
            except:
                # If direct copy fails, skip sync
                pass
                
        except:
            # Silent failure - just skip sync
            pass
        
        self._freeze_parameters(self.old_model)
    
    def _sample(self, model, input_text: str, max_new_tokens: int) -> str:
        """Sample from a model with given input text"""
        with torch.no_grad():
            # Unwrap DeepSpeed engine to get the underlying model
            actual_model = getattr(model, 'module', model)
            # Further unwrap Actor wrapper if present
            actual_model = getattr(actual_model, 'model', actual_model)
            
            enc = self.tokenizer(
                [input_text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(actual_model.device)
            
            output = actual_model.generate(
                **enc, 
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.batch_decode(
                output[:, enc['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )[0]
            
        return generated_text
    
    def _evaluate_and_feedback(self, prompt: str, context: str, segment: str, 
                               turn: str, is_verification: bool = False) -> Tuple[float, str]:
        """Evaluate code segment and return passrate and tool feedback"""
        # Use existing reward functions
        if hasattr(self, 'reward_funcs') and self.reward_funcs:
            reward_func = self.reward_funcs[0]
            try:
                reward = reward_func([segment], prompt, stage=turn, draft=None, turn=turn)
                # Convert reward to passrate (assuming reward range [-1, 6])
                passrate = max(0.0, min(1.0, (reward + 1) / 7))
                feedback = f"Pass rate: {passrate:.2f}, Reward: {reward:.2f}"
            except Exception as e:
                print(f"Reward function error: {e}")
                passrate = 0.0
                feedback = f"Evaluation error: {str(e)}"
        else:
            # Default evaluation: simple format check
            if "```" in segment and ("module" in segment.lower() or "function" in segment.lower()):
                passrate = 0.8
                feedback = "Format check passed"
            else:
                passrate = 0.2
                feedback = "Format check failed"
        
        return passrate, feedback
    
    
    
    
    def prepare_refinement_batch(self, prompts: List[str], y1y2_groups: List[List[Tuple[str, List[str]]]], 
                                deltas_all: List[List[List[float]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training batch with refinement samples.
        
        Returns:
            Tuple of (input_ids, attention_mask, loss_mask, labels, advantages)
        """
        ids_list, attn_list, mask_list, labels_list, adv_list = [], [], [], [], []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        
        for x, draft_pack, deltas_for_prompt in zip(prompts, y1y2_groups, deltas_all):
            for (y1, y2_list), group_delta in zip(draft_pack, deltas_for_prompt):
                for delta, y2 in zip(group_delta, y2_list):
                    # Stable boundary computation: tokenize separately then concatenate
                    ids_x = self.tokenizer(x, add_special_tokens=False, return_tensors="pt").input_ids[0]
                    ids_y1 = self.tokenizer(y1, add_special_tokens=False, return_tensors="pt").input_ids[0]
                    ids_y2 = self.tokenizer(y2, add_special_tokens=False, return_tensors="pt").input_ids[0]
                    full = torch.cat([ids_x, ids_y1, ids_y2], dim=0)
                    
                    MAX = 512 + 2 * self.config.max_completion_length
                    if full.numel() > MAX:
                        cut = full.numel() - MAX
                        full = full[cut:]
                        start = max(0, ids_x.numel() + ids_y1.numel() - cut)
                    else:
                        start = ids_x.numel() + ids_y1.numel()
                    end = full.numel()
                    
                    ids = full
                    attn = torch.ones_like(ids)
                    loss_mask = torch.zeros_like(ids)
                    if end > start:
                        loss_mask[start:end] = 1  # Only y2 segment
                    
                    ids_list.append(ids)
                    attn_list.append(attn)
                    mask_list.append(loss_mask)
                    labels_list.append(ids.clone())
                    adv_list.append(float(delta))
        
        # Pad sequences to same length
        max_len = max(len(ids) for ids in ids_list)
        
        padded_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        padded_attn = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        padded_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        padded_labels = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        
        for i, (ids, attn, mask, labels) in enumerate(zip(ids_list, attn_list, mask_list, labels_list)):
            seq_len = len(ids)
            padded_ids[i, :seq_len] = ids
            padded_attn[i, :seq_len] = attn
            padded_mask[i, :seq_len] = mask
            padded_labels[i, :seq_len] = labels
        
        # Move tensors to device
        device = self.model.device
        advantages = torch.tensor(adv_list, dtype=torch.float32, device=device)
        
        return (padded_ids.to(device), padded_attn.to(device), 
                padded_mask.to(device), padded_labels.to(device), advantages)
    
    def compute_refinement_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                              loss_mask: torch.Tensor, labels: torch.Tensor, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss with refinement logic (y2-only).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            loss_mask: Loss mask (1 for y2 segment, 0 otherwise) [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            advantages: Refinement advantages [batch_size]
            
        Returns:
            Dict containing loss components
        """
        # Forward pass
        outputs = self.model(sequences=input_ids, attention_mask=attention_mask, return_output=True)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Shift for autoregressive training
        shift_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[:, 1:]      # [batch_size, seq_len-1]
        shift_mask = loss_mask[:, 1:].float()  # [batch_size, seq_len-1] - right-shifted mask
        
        # Compute log probabilities for current policy
        logp_new = F.log_softmax(shift_logits, dim=-1)
        logp_new_sel = torch.gather(logp_new, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Compute log probabilities for old policy (behavior policy)
        with torch.no_grad():
            # Use old_model as behavior policy for PPO ratio denominator
            old_outputs = self.old_model(sequences=input_ids, attention_mask=attention_mask, return_output=True)
            old_logits = old_outputs.logits[:, :-1, :]
            logp_old = F.log_softmax(old_logits, dim=-1)
            logp_old_sel = torch.gather(logp_old, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Compute importance sampling ratio (sequence-level)
        seq_log_ratio = (logp_new_sel - logp_old_sel) * shift_mask
        seq_log_ratio = seq_log_ratio.sum(dim=1)  # Sum over y2 tokens
        ratio = torch.exp(seq_log_ratio)
        
        # Global normalization of advantages (already group-normalized in compute_refinement_rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO-Clip objective
        clip = self.config.clip_ratio
        clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        
        # Correct KL divergence computation (forward KL against reference model)
        with torch.no_grad():
            ref_outputs = self.ref_model(sequences=input_ids, attention_mask=attention_mask, return_output=True)
            ref_logits = ref_outputs.logits[:, :-1, :]
            logp_ref = F.log_softmax(ref_logits, dim=-1)
        
        p_new = torch.softmax(shift_logits, dim=-1)
        kl_token = (p_new * (logp_new - logp_ref)).sum(-1)  # [batch_size, seq_len-1]
        kl_loss = (kl_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
        
        total_loss = policy_loss + self.config.kl_coef * kl_loss
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "mean_ratio": ratio.mean(),
            "mean_advantage": advantages.mean(),
            "y2_active_tokens": int(shift_mask.sum().item()),
        }
    
    def train_step(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Single training step with refinement logic.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            Dict containing training metrics
        """
        # 1. Generate drafts and refinements
        y1y2_groups = self.generate_draft_and_refinements(prompts)
        
        # 2. Compute refinement advantages
        deltas_all = self.compute_refinement_rewards(prompts, y1y2_groups)
        
        # 3. Prepare refinement batch
        input_ids, attention_mask, loss_mask, labels, advantages = self.prepare_refinement_batch(
            prompts, y1y2_groups, deltas_all
        )
        
        # 4. Compute loss
        loss_dict = self.compute_refinement_loss(
            input_ids, attention_mask, loss_mask, labels, advantages
        )
        
        # 5. Backward pass
        loss_dict["total_loss"].backward()
        
        # 6. Logging
        metrics = {
            "total_loss": loss_dict["total_loss"].item(),
            "policy_loss": loss_dict["policy_loss"].item(),
            "kl_loss": loss_dict["kl_loss"].item(),
            "mean_ratio": loss_dict["mean_ratio"].item(),
            "mean_advantage": advantages.mean().item(),
            "active_tokens": int((loss_mask[:, 1:].float()).sum().item()),
            "step_count": self.step_count,
        }
        
        return metrics
    
    def inference(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        """
        Inference with two-stage generation (draft -> refinement).
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for prompt in prompts:
            # Stage 1: Generate draft
            with torch.no_grad():
                enc = self.tokenizer(
                    [prompt], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.model.device)
                
                # Unwrap DeepSpeed engine to get the underlying model
                actual_ref_model = getattr(self.ref_model, 'module', self.ref_model)
                actual_ref_model = getattr(actual_ref_model, 'model', actual_ref_model)
                
                draft_out = actual_ref_model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens // 2,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                draft_text = self.tokenizer.batch_decode(
                    draft_out[:, enc['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )[0]
            
            # Stage 2: Generate refinement conditioned on (prompt, draft)
            with torch.no_grad():
                cond_text = prompt + draft_text
                cond_enc = self.tokenizer(
                    [cond_text], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512 + max_new_tokens // 2
                ).to(self.model.device)
                
                # Unwrap DeepSpeed engine to get the underlying model
                actual_model = getattr(self.model, 'module', self.model)
                actual_model = getattr(actual_model, 'model', actual_model)
                
                refine_out = actual_model.generate(
                    **cond_enc,
                    max_new_tokens=max_new_tokens // 2,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                refine_text = self.tokenizer.batch_decode(
                    refine_out[:, cond_enc['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )[0]
            
            # Return full response (draft + refinement)
            responses.append(draft_text + refine_text)
        
        return responses
    
    
