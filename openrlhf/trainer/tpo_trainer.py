import torch
import os
from tqdm import tqdm
from openrlhf.utils.distributed_sampler import DistributedSampler


class BaseDPOTrainer:
    """
    公共方法基类，实现核心条件概率构造和log概率计算。
    供22和23共享。
    """

    def __init__(self, model, ref_model, strategy, tokenizer, optim, train_dataloader, eval_dataloader, scheduler,
                 max_norm=0.5, beta=0.01, max_epochs=2, save_hf_ckpt=False, disable_ds_ckpt=False, args=None):
        self.strategy = strategy
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.max_norm = max_norm
        self.beta = beta
        self.max_epochs = max_epochs
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.args = args if args is not None else strategy.args

        # 是否启用详细指标计算（会影响训练速度）
        self.enable_detailed_metrics = getattr(strategy.args, 'enable_detailed_metrics', False)

        # 重要：确保参考模型是固定的，不参与训练
        if self.ref_model is not None:
            self.ref_model.eval()
            # 冻结参考模型的所有参数
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def compute_logprob(self, model, input_ids, attention_mask, labels):
        # 使用更高效的方式调用模型，避免不必要的 return_output
        try:
            # 直接调用模型，让 Actor 类处理参数
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[..., :-1, :]
            labels_shifted = labels[..., 1:]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_logprobs = torch.gather(logprobs, -1, labels_shifted.unsqueeze(-1)).squeeze(-1)
            mask = (labels_shifted != -100).float()
            seq_logprob = (selected_logprobs * mask).sum(dim=1)
            return seq_logprob
        except Exception as e:
            # 如果直接调用失败，回退到标准方式
            outputs = model(sequences=input_ids, attention_mask=attention_mask, return_output=True)
            logits = outputs.logits[..., :-1, :]
            labels_shifted = labels[..., 1:]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_logprobs = torch.gather(logprobs, -1, labels_shifted.unsqueeze(-1)).squeeze(-1)
            mask = (labels_shifted != -100).float()
            seq_logprob = (selected_logprobs * mask).sum(dim=1)
            return seq_logprob

    def compute_dpo_metrics(self, chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps):
        """
        计算DPO相关的指标
        """
        # 计算reward (log probability difference)
        chosen_rewards = chosen_logps - reference_chosen_logps
        rejected_rewards = rejected_logps - reference_rejected_logps
        
        # 计算reward difference (这是DPO loss的核心)
        reward_diff = chosen_rewards - rejected_rewards
        
        # 计算准确率 (chosen reward > rejected reward的比例)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        # 计算reward统计信息
        chosen_reward_mean = chosen_rewards.mean()
        rejected_reward_mean = rejected_rewards.mean()
        reward_diff_mean = reward_diff.mean()
        
        # 计算logprob统计信息
        chosen_logp_mean = chosen_logps.mean()
        rejected_logp_mean = rejected_logps.mean()
        ref_chosen_logp_mean = reference_chosen_logps.mean()
        ref_rejected_logp_mean = reference_rejected_logps.mean()
        
        metrics = {
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
            'reward_diff': reward_diff,
            'accuracy': accuracy,
            'chosen_reward_mean': chosen_reward_mean,
            'rejected_reward_mean': rejected_reward_mean,
            'reward_diff_mean': reward_diff_mean,
            'chosen_logp_mean': chosen_logp_mean,
            'rejected_logp_mean': rejected_logp_mean,
            'ref_chosen_logp_mean': ref_chosen_logp_mean,
            'ref_rejected_logp_mean': ref_rejected_logp_mean,
        }
        
        return metrics

    def compute_tpo_loss(self, chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, 
                         logp_y2_given_y1=None, ref_logp_y2=None, logp_y1_given_y2=None, ref_logp_y1_given_y2=None):
        """
        计算TPO loss，支持公式22和公式23
        """
        # 基础DPO指标
        dpo_metrics = self.compute_dpo_metrics(
            chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
        
        # 计算基础DPO loss
        dpo_loss = -torch.nn.functional.logsigmoid(dpo_metrics['reward_diff']).mean()
        
        # 如果提供了TPO特定参数，计算TPO loss
        if logp_y2_given_y1 is not None and ref_logp_y2 is not None:
            beta_tensor = torch.tensor(self.beta, device=chosen_logps.device)
            
            if logp_y1_given_y2 is not None and ref_logp_y1_given_y2 is not None:
                # 公式23: loss = - log σ(β * ((log π(y₂|y₁,x) - log π_ref(y₂|x)) - (log π(y₁|y₂,x) - log π_ref(y₁|x))))
                tpo_logits = beta_tensor * ((logp_y2_given_y1 - ref_logp_y2) - (logp_y1_given_y2 - ref_logp_y1_given_y2))
            else:
                # 公式22: loss = - log σ(β * (log π(y₂|y₁,x) - log π_ref(y₂|x)))
                tpo_logits = beta_tensor * (logp_y2_given_y1 - ref_logp_y2)
            
            tpo_loss = -torch.nn.functional.logsigmoid(tpo_logits).mean()
            
            return tpo_loss, dpo_metrics, tpo_logits
        else:
            # 只返回DPO loss
            return dpo_loss, dpo_metrics, None

    def compute_detailed_metrics(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """
        计算更详细的训练指标，包括序列长度、token分布等
        """
        batch_size = chosen_ids.size(0)
        
        # 计算序列长度统计
        chosen_lengths = c_mask.sum(dim=1)
        reject_lengths = r_mask.sum(dim=1)
        prompt_lengths = prompt_id_lens
        
        # 计算有效token数量
        total_chosen_tokens = chosen_lengths.sum().item()
        total_reject_tokens = reject_lengths.sum().item()
        total_prompt_tokens = prompt_lengths.sum().item()
        
        # 计算平均长度
        avg_chosen_len = chosen_lengths.float().mean().item()
        avg_reject_len = reject_lengths.float().mean().item()
        avg_prompt_len = prompt_lengths.float().mean().item()
        
        # 计算长度差异
        length_diff = (chosen_lengths - reject_lengths).abs().float().mean().item()
        
        # 计算token分布统计
        chosen_token_dist = chosen_ids[c_mask.bool()].bincount()
        reject_token_dist = reject_ids[r_mask.bool()].bincount()
        
        # 计算词汇表覆盖率
        chosen_vocab_size = (chosen_token_dist > 0).sum().item()
        reject_vocab_size = (reject_token_dist > 0).sum().item()
        
        metrics = {
            'batch_size': batch_size,
            'total_chosen_tokens': total_chosen_tokens,
            'total_reject_tokens': total_reject_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'avg_chosen_len': avg_chosen_len,
            'avg_reject_len': avg_reject_len,
            'avg_prompt_len': avg_prompt_len,
            'length_diff': length_diff,
            'chosen_vocab_size': chosen_vocab_size,
            'reject_vocab_size': reject_vocab_size,
        }
        
        return metrics

    def get_conditioned_inputs_and_labels(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id
        batch_size = chosen_ids.size(0)
        device = chosen_ids.device

        input_ids1_list = []
        labels1_list = []
        input_ids2_list = []
        labels2_list = []
        input_ids3_list = []
        labels3_list = []

        for i in range(batch_size):
            chosen_seq = chosen_ids[i]
            reject_seq = reject_ids[i]
            chosen_len = c_mask[i].sum().item()
            reject_len = r_mask[i].sum().item()
            prompt_len = prompt_id_lens[i]

            # 确保长度是有效的
            chosen_len = max(1, min(chosen_len, chosen_seq.size(0)))
            reject_len = max(1, min(reject_len, reject_seq.size(0)))
            prompt_len = max(0, min(prompt_len, min(chosen_len, reject_len)))

            chosen_seq = chosen_seq[:int(chosen_len)].tolist()
            reject_seq = reject_seq[:int(reject_len)].tolist()

            prompt_tokens = chosen_seq[:prompt_len]
            y1_tokens = chosen_seq[prompt_len:]
            y2_tokens = reject_seq[prompt_len:]

            # 确保y1_tokens和y2_tokens不为空
            if len(y1_tokens) == 0:
                y1_tokens = [pad_token_id]
            if len(y2_tokens) == 0:
                y2_tokens = [pad_token_id]

            # prompt + y1
            input_ids1 = prompt_tokens + y1_tokens
            labels1 = [-100] * prompt_len + y1_tokens
            max_len1 = max(len(input_ids1), len(chosen_seq))
            input_ids1 += [pad_token_id] * (max_len1 - len(input_ids1))
            labels1 += [-100] * (max_len1 - len(labels1))

            # prompt + y1 + y2
            input_ids2 = prompt_tokens + y1_tokens + y2_tokens
            labels2 = [-100] * (len(prompt_tokens) + len(y1_tokens)) + y2_tokens
            max_len2 = max(len(input_ids2), len(chosen_seq) + len(reject_seq))
            input_ids2 += [pad_token_id] * (max_len2 - len(input_ids2))
            labels2 += [-100] * (max_len2 - len(labels2))

            # prompt + y2 + y1
            input_ids3 = prompt_tokens + y2_tokens + y1_tokens
            labels3 = [-100] * (len(prompt_tokens) + len(y2_tokens)) + y1_tokens
            max_len3 = max_len2
            input_ids3 += [pad_token_id] * (max_len3 - len(input_ids3))
            labels3 += [-100] * (max_len3 - len(labels3))

            input_ids1_list.append(torch.tensor(input_ids1, device=device))
            labels1_list.append(torch.tensor(labels1, device=device))
            input_ids2_list.append(torch.tensor(input_ids2, device=device))
            labels2_list.append(torch.tensor(labels2, device=device))
            input_ids3_list.append(torch.tensor(input_ids3, device=device))
            labels3_list.append(torch.tensor(labels3, device=device))

        input_ids1_batch = torch.stack(input_ids1_list)
        labels1_batch = torch.stack(labels1_list)
        input_ids2_batch = torch.stack(input_ids2_list)
        labels2_batch = torch.stack(labels2_list)
        input_ids3_batch = torch.stack(input_ids3_list)
        labels3_batch = torch.stack(labels3_list)

        attention_mask1 = (input_ids1_batch != pad_token_id).long()
        attention_mask2 = (input_ids2_batch != pad_token_id).long()
        attention_mask3 = (input_ids3_batch != pad_token_id).long()

        return (input_ids1_batch, attention_mask1, labels1_batch), (input_ids2_batch, attention_mask2, labels2_batch), (input_ids3_batch, attention_mask3, labels3_batch)

    def get_ref_inputs_and_labels(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Build (p+y1) and (p+y2) with labels only on the answer span (non-conditional ref targets)."""
        tok = self.tokenizer; pad = tok.pad_token_id; dev = chosen_ids.device
        B = chosen_ids.size(0)
        ids_y1x, lab_y1x, ids_y2x, lab_y2x = [], [], [], []
        for i in range(B):
            p = int(prompt_id_lens[i])
            cseq = chosen_ids[i, :c_mask[i].sum()].tolist()
            rseq = reject_ids[i, :r_mask[i].sum()].tolist()
            prompt, y1, y2 = cseq[:p], cseq[p:], rseq[p:]
            if len(y1)==0: y1=[pad]
            if len(y2)==0: y2=[pad]
            s1 = prompt + y1; l1 = [-100]*len(prompt) + y1
            s2 = prompt + y2; l2 = [-100]*len(prompt) + y2
            ids_y1x.append(torch.tensor(s1, device=dev)); lab_y1x.append(torch.tensor(l1, device=dev))
            ids_y2x.append(torch.tensor(s2, device=dev)); lab_y2x.append(torch.tensor(l2, device=dev))
        def pad_stack(S, L):
            mx = max(len(s) for s in S)
            ids = torch.full((B, mx), pad, dtype=torch.long, device=dev)
            lab = torch.full((B, mx), -100, dtype=torch.long, device=dev)
            for i,(s,l) in enumerate(zip(S,L)):
                ids[i,:len(s)] = s; lab[i,:len(l)] = l
            att = (ids != pad).long()
            return ids, att, lab
        return pad_stack(ids_y1x, lab_y1x), pad_stack(ids_y2x, lab_y2x)


    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None):
        logs_dict = logs_dict or {}
        client_states = client_states or {}

        if global_step % args.logging_steps == 0:
            # wandb日志
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard日志
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save checkpoint
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(len(eval_dataloader)),
                desc=f"Eval stage of global_step {steps}",
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                # 计算chosen和rejected的logprobs
                chosen_logps = self.compute_logprob(self.model, chosen_ids, c_mask, labels=chosen_ids)
                rejected_logps = self.compute_logprob(self.model, reject_ids, r_mask, labels=reject_ids)
                
                with torch.no_grad():
                    ref_chosen_logps = self.compute_logprob(self.ref_model, chosen_ids, c_mask, labels=chosen_ids)
                    ref_rejected_logps = self.compute_logprob(self.ref_model, reject_ids, r_mask, labels=reject_ids)

                # 计算DPO指标
                dpo_metrics = self.compute_dpo_metrics(
                    chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
                )
                
                # 使用DPO指标计算loss和accuracy
                loss = -torch.nn.functional.logsigmoid(dpo_metrics['reward_diff']).mean()
                acc = dpo_metrics['accuracy'].item()
                
                acc_sum += acc
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times if times > 0 else 0,
                "acc_mean": acc_sum / times if times > 0 else 0,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

class DPOTrainer22(BaseDPOTrainer):
    """
    实现公式22：
    loss = - log σ(β * (log π(y₂|y₁,x) - log π_ref(y₂|x)))
    """

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.max_epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        
        acc_sum = 0
        loss_sum = 0
        
        for epoch in range(start_epoch, self.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            self.ref_model.eval()

            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                
                # 优化 prompt_id_lens 处理，减少类型检查开销
                if not hasattr(prompt_id_lens, 'to'):
                    # 如果是列表，转换为张量
                    prompt_id_lens = torch.tensor(prompt_id_lens, device=chosen_ids.device, dtype=torch.long)
                else:
                    # 如果已经是张量，直接移动设备
                    prompt_id_lens = prompt_id_lens.to(chosen_ids.device)

                (input_ids1, att_mask1, labels1), (input_ids2, att_mask2, labels2), _ = self.get_conditioned_inputs_and_labels(
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )

                # 只计算TPO loss所需的logprobs，避免重复计算
                logp_y2_given_y1 = self.compute_logprob(self.model, input_ids2, att_mask2, labels2)
                with torch.no_grad():
                    # ref_logp_y2 = self.compute_logprob(self.ref_model, input_ids2, att_mask2, labels2)
                    (_,_,_), (ids_y2x, att_y2x, lab_y2x) = self.get_ref_inputs_and_labels(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
                    ref_logp_y2 = self.compute_logprob(self.ref_model, ids_y2x, att_y2x, lab_y2x)


                # 计算TPO loss (公式22) - 修正实现
                beta_tensor = torch.tensor(self.beta, device=chosen_ids.device)
                # TPO公式22: loss = -log σ(β * (log π(y₂|y₁,x) - log π_ref(y₂|x)))
                # 我们希望 log π(y₂|y₁,x) < log π_ref(y₂|x)，即 (log π(y₂|y₁,x) - log π_ref(y₂|x)) < 0
                log_ratio = logp_y2_given_y1 - ref_logp_y2
                dpo_logits = beta_tensor * log_ratio
                
                # 使用更稳定的loss计算
                # 当 log_ratio < 0 时，我们希望loss小
                # 当 log_ratio > 0 时，我们希望loss大
                loss = -torch.nn.functional.logsigmoid(-dpo_logits).mean()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # 计算准确率 - 修正逻辑：TPO希望模型给rejected response更低概率
                # 所以当 logp_y2_given_y1 < ref_logp_y2 时，模型表现好
                # 但我们需要一个更有意义的accuracy指标
                acc = (log_ratio < 0).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()

                # 简化的日志记录，只保留核心指标
                logs_dict = {
                    "loss": loss.item(),
                    "acc": acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    # TPO特定指标
                    "tpo_logits_mean": dpo_logits.mean().item(),
                    "tpo_accuracy": acc,
                    # 调试信息
                    "logp_y2_given_y1_mean": logp_y2_given_y1.mean().item(),
                    "ref_logp_y2_mean": ref_logp_y2.mean().item(),
                    "log_ratio_mean": log_ratio.mean().item(),
                    "log_ratio_std": log_ratio.std().item(),
                }

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


class DPOTrainer23(BaseDPOTrainer):
    """
    实现公式23：
    loss = - log σ(β * ((log π(y₂|y₁,x) - log π_ref(y₂|x)) - (log π(y₁|y₂,x) - log π_ref(y₁|x))))
    """

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.max_epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        
        acc_sum = 0
        loss_sum = 0
        
        for epoch in range(start_epoch, self.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            self.ref_model.eval()

            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                
                # 优化 prompt_id_lens 处理，减少类型检查开销
                if not hasattr(prompt_id_lens, 'to'):
                    # 如果是列表，转换为张量
                    prompt_id_lens = torch.tensor(prompt_id_lens, device=chosen_ids.device, dtype=torch.long)
                else:
                    # 如果已经是张量，直接移动设备
                    prompt_id_lens = prompt_id_lens.to(chosen_ids.device)

                (input_ids1, att_mask1, labels1), (input_ids2, att_mask2, labels2), (input_ids3, att_mask3, labels3) = self.get_conditioned_inputs_and_labels(
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )

                # 只计算TPO loss所需的logprobs，避免重复计算
                logp_y2_given_y1 = self.compute_logprob(self.model, input_ids2, att_mask2, labels2)
                logp_y1_given_y2 = self.compute_logprob(self.model, input_ids3, att_mask3, labels3)

                with torch.no_grad():
                    ref_logp_y2 = self.compute_logprob(self.ref_model, input_ids2, att_mask2, labels2)
                    ref_logp_y1_given_y2 = self.compute_logprob(self.ref_model, input_ids3, att_mask3, labels3)

                # 计算TPO loss (公式23) - 直接计算，不使用复杂函数
                beta_tensor = torch.tensor(self.beta, device=chosen_ids.device)
                dpo_logits = beta_tensor * ((logp_y2_given_y1 - ref_logp_y2) - (logp_y1_given_y2 - ref_logp_y1_given_y2))
                # 目标：Δ 趋于负 => 用 -log σ(-βΔ) 或 softplus(βΔ)
                loss = -torch.nn.functional.logsigmoid(-dpo_logits).mean()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # 计算准确率 - 修正逻辑：TPO希望模型给rejected response更低概率
                # 公式23的准确率：当 (logp_y2_given_y1 - ref_logp_y2) < (logp_y1_given_y2 - ref_logp_y1_given_y2) 时表现好
                acc = ((logp_y2_given_y1 - ref_logp_y2) < (logp_y1_given_y2 - ref_logp_y1_given_y2)).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()

                # 简化的日志记录，只保留核心指标
                logs_dict = {
                    "loss": loss.item(),
                    "acc": acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    # TPO特定指标 (公式23)
                    "tpo_logits_mean": dpo_logits.mean().item(),
                    "tpo_accuracy": acc,
                    "logp_y2_given_y1_mean": logp_y2_given_y1.mean().item(),
                    "logp_y1_given_y2_mean": logp_y1_given_y2.mean().item(),
                }

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


class SimPOTrainer24(BaseDPOTrainer):
    """
    实现公式(24)的Trainer。

    它继承自 BaseDPOTrainer 以复用核心方法，但在 fit 循环中实现了自己独特的、
    不依赖 ref_model 的损失函数，并自行计算序列长度。
    """

    def __init__(self, model, ref_model, strategy, tokenizer, optim, train_dataloader, eval_dataloader, scheduler,
                 max_norm=0.5, beta=0.01, gamma=0.1, max_epochs=2, save_hf_ckpt=False, disable_ds_ckpt=False, args=None):
        # 调用父类的 __init__，即使我们不会使用 ref_model，也要保持签名一致
        super().__init__(model, ref_model, strategy, tokenizer, optim, train_dataloader, eval_dataloader, scheduler,
                         max_norm, beta, max_epochs, save_hf_ckpt, disable_ds_ckpt, args)
        
        # 为公式(24)增加新的超参数 gamma
        self.gamma = gamma
        self.strategy.print(f"DPOTrainer24 initialized with beta={self.beta}, gamma={self.gamma}")


    def fit(self, args):
        # tqdm progress bars
        epoch_bar = tqdm(range(self.max_epochs), desc="Epochs", disable=not self.strategy.is_rank_0())
        global_step = 0
        
        for epoch in range(self.max_epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()), desc="Steps", disable=not self.strategy.is_rank_0())
            
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            
            for step, batch in enumerate(self.train_dataloader):
                # 从batch中获取ids (这是trainer22和23也依赖的)
                prompt_ids = batch['prompt_ids']
                chosen_ids = batch['chosen_ids']
                reject_ids = batch['reject_ids']
                
                # ==================== 核心修正 ====================
                # 参照trainer22/23的实现，不假设batch中有长度信息。
                # 我们直接从 token id 列表中计算出长度。
                # chosen_ids 和 reject_ids 是一个list of tensors
                y1_lens = torch.tensor([ids.size(0) for ids in chosen_ids], dtype=torch.float32).to(self.strategy.device)
                y2_lens = torch.tensor([ids.size(0) for ids in reject_ids], dtype=torch.float32).to(self.strategy.device)
                # ===============================================

                # 1. 使用基类方法构造 TPO 输入和标签
                (inputs_y2, mask_y2, labels_y2), (inputs_y1, mask_y1, labels_y1), _ = \
                    self.get_conditioned_inputs_and_labels(prompt_ids, chosen_ids, reject_ids)
                
                # 2. 计算条件对数概率 (只使用 self.model)
                # 我们完全不使用 self.ref_model
                logp_y2_given_y1 = self.compute_logprob(self.model, inputs_y2, mask_y2, labels_y2)
                logp_y1_given_y2 = self.compute_logprob(self.model, inputs_y1, mask_y1, labels_y1)
                
                # 3. 直接在这里实现公式(24)的损失计算逻辑
                # 确保长度不为0，避免除零错误
                y1_lens_clamped = torch.clamp(y1_lens, min=1)
                y2_lens_clamped = torch.clamp(y2_lens, min=1)
                
                # 计算长度归一化的对数概率
                term1 = (self.beta / y2_lens_clamped) * logp_y2_given_y1
                term2 = (self.beta / y1_lens_clamped) * logp_y1_given_y2
                
                logits = term1 - term2 - self.gamma
                
                loss = -torch.nn.functional.logsigmoid(logits).mean()
                
                # 4. 反向传播和优化器步骤
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # 5. 日志记录
                acc = (logits > 0).float().mean().item()

                if self.strategy.is_rank_0():
                    logs_dict = {
                        "loss": loss.item(),
                        "acc": acc,
                        "lr": self.scheduler.get_last_lr()[0],
                        # 公式24特定指标
                        "logits_mean": logits.mean().item(),
                    }
                    
                    if self._wandb is not None:
                        self._wandb.log(logs_dict, step=global_step)
                    
                    step_bar.set_postfix(logs_dict)
                
                step_bar.update()
                global_step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()


# =========================
# 25) IPO-23 Trainer
# =========================
class IPOTrainer25(BaseDPOTrainer):
    """
    公式25 (IPO-23)：
    L = ( [log πθ(y2|y1,x) - log πref(y2|x)] - [log πθ(y1|y2,x) - log πref(y1|x)] - 1/(2τ) )^2
    """
    def __init__(self, *args, tau: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.max_epochs),
                         desc="Train epoch", disable=not self.strategy.is_rank_0())

        acc_sum = 0.0
        loss_sum = 0.0

        for epoch in range(start_epoch, self.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(range(len(self.train_dataloader)),
                            desc=f"[25-IPO] Train step of epoch {epoch}",
                            disable=not self.strategy.is_rank_0())

            self.model.train()
            self.ref_model.eval()

            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                device = torch.cuda.current_device()
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)
                reject_ids = reject_ids.squeeze(1).to(device)
                r_mask = r_mask.squeeze(1).to(device)
                if not hasattr(prompt_id_lens, 'to'):
                    prompt_id_lens = torch.tensor(prompt_id_lens, device=device, dtype=torch.long)
                else:
                    prompt_id_lens = prompt_id_lens.to(device)

                # 条件化三路：拿到 y2|y1 与 y1|y2
                (ids1, att1, lab1), (ids2, att2, lab2), (ids3, att3, lab3) = \
                    self.get_conditioned_inputs_and_labels(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)

                logp_y2gy1 = self.compute_logprob(self.model, ids2, att2, lab2)
                logp_y1gy2 = self.compute_logprob(self.model, ids3, att3, lab3)

                # 参考模型：y2|x 与 y1|x（使用正确的构造器，只监督y段）
                with torch.no_grad():
                    (ids_y2x, att_y2x, lab_y2x), (ids_y1x, att_y1x, lab_y1x) = \
                        self.get_ref_inputs_and_labels(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
                    ref_logp_y2x = self.compute_logprob(self.ref_model, ids_y2x, att_y2x, lab_y2x)
                    ref_logp_y1x = self.compute_logprob(self.ref_model, ids_y1x, att_y1x, lab_y1x)

                # IPO-23：平方项
                delta = (logp_y2gy1 - ref_logp_y2x) - (logp_y1gy2 - ref_logp_y1x) - (1.0 / (2.0 * self.tau))
                loss = (delta ** 2).mean()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # 命中率：希望 (y2|y1) 相对分数 < (y1|y2) 相对分数
                acc = ((logp_y2gy1 - ref_logp_y2x) < (logp_y1gy2 - ref_logp_y1x)).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()

                logs = {
                    "loss": loss.item(),
                    "acc": acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    "ipo_delta_mean": delta.mean().item(),
                }
                logs = self.strategy.all_reduce(logs)
                step_bar.set_postfix(logs)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    logs["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0.0
                    acc_sum = 0.0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


# =========================
# 26) R-DPO-23 Trainer
# =========================
class RDPOTrainer26(BaseDPOTrainer):
    """
    公式26 (R-DPO-23)：
    L = -log σ( β * ([log πθ(y2|y1,x) - log πref(y2|x)] - [log πθ(y1|y2,x) - log πref(y1|x)]) + α*(|y2|-|y1|) )
    """
    def __init__(self, *args, alpha: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # self.beta 已在父类中

    @staticmethod
    def _count_valid(labels: torch.Tensor) -> torch.Tensor:
        # 统计每个样本监督段长度（label != -100）
        return (labels != -100).sum(dim=1).float().clamp_min(1.0)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.max_epochs),
                         desc="Train epoch", disable=not self.strategy.is_rank_0())

        acc_sum = 0.0
        loss_sum = 0.0

        for epoch in range(start_epoch, self.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(range(len(self.train_dataloader)),
                            desc=f"[26-RDPO] Train step of epoch {epoch}",
                            disable=not self.strategy.is_rank_0())

            self.model.train()
            self.ref_model.eval()

            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                device = torch.cuda.current_device()
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)
                reject_ids = reject_ids.squeeze(1).to(device)
                r_mask = r_mask.squeeze(1).to(device)
                if not hasattr(prompt_id_lens, 'to'):
                    prompt_id_lens = torch.tensor(prompt_id_lens, device=device, dtype=torch.long)
                else:
                    prompt_id_lens = prompt_id_lens.to(device)

                # 条件化三路：y2|y1, y1|y2
                (ids1, att1, lab1), (ids2, att2, lab2), (ids3, att3, lab3) = \
                    self.get_conditioned_inputs_and_labels(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)

                logp_y2gy1 = self.compute_logprob(self.model, ids2, att2, lab2)
                logp_y1gy2 = self.compute_logprob(self.model, ids3, att3, lab3)

                # 参考模型：必须是 y|x，而不是 y|另一条
                with torch.no_grad():
                    (ids_y2x, att_y2x, lab_y2x), (ids_y1x, att_y1x, lab_y1x) = \
                        self.get_ref_inputs_and_labels(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
                    ref_logp_y2x = self.compute_logprob(self.ref_model, ids_y2x, att_y2x, lab_y2x)
                    ref_logp_y1x = self.compute_logprob(self.ref_model, ids_y1x, att_y1x, lab_y1x)

                # |y| 直接用监督段长度（label != -100）统计，保证和 logprob 的归一口径一致
                len_y2 = self._count_valid(lab_y2x)   # 等价于 |y2|
                len_y1 = self._count_valid(lab_y1x)   # 等价于 |y1|
                
                # z = β * Δ_cond^{23} + α*(|y2|-|y1|)
                beta_t = torch.tensor(self.beta, device=chosen_ids.device)
                z = beta_t * ((logp_y2gy1 - ref_logp_y2x) - (logp_y1gy2 - ref_logp_y1x)) + self.alpha * (len_y2 - len_y1)

                # 目标同样是"更偏向 y1 好于 y2"
                loss = -torch.nn.functional.logsigmoid(-z).mean()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (z > 0).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()

                logs = {
                    "loss": loss.item(),
                    "acc": acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    "rdpo_logit_mean": z.mean().item(),
                }
                logs = self.strategy.all_reduce(logs)
                step_bar.set_postfix(logs)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    logs["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0.0
                    acc_sum = 0.0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

# =========================
# 27) ORPO-23 Trainer
# =========================
class ORPOTrainer27(BaseDPOTrainer):
    """
    公式27 (ORPO-23)：
    L = - avg_log πθ(y1|x) + λ * ( -log σ( logit(pθ(y2|y1,x)) - logit(pθ(y1|y2,x)) ) )
    其中 avg_log 与 pθ 均基于长度归一化；不需要参考模型。
    """
    def __init__(self, *args, lambda_pair: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_pair = lambda_pair

    @staticmethod
    def _avg_from_labels(sum_logp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B, T], -100 掩掉非监督段
        denom = (labels != -100).sum(dim=1).float().clamp_min(1.0)
        return sum_logp / denom

    @staticmethod
    def _safe_logit_from_avglogp(avg_logp: torch.Tensor) -> torch.Tensor:
        # p = exp(avg_logp) ∈ (0,1]; logit = log p - log (1-p)
        p = torch.exp(avg_logp).clamp(max=1 - 1e-8)
        return avg_logp - torch.log1p(-p + 1e-12)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.max_epochs),
                         desc="Train epoch", disable=not self.strategy.is_rank_0())

        acc_sum = 0.0
        loss_sum = 0.0

        for epoch in range(start_epoch, self.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(range(len(self.train_dataloader)),
                            desc=f"[27-ORPO] Train step of epoch {epoch}",
                            disable=not self.strategy.is_rank_0())

            self.model.train()

            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                device = torch.cuda.current_device()
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)
                reject_ids = reject_ids.squeeze(1).to(device)
                r_mask = r_mask.squeeze(1).to(device)
                if not hasattr(prompt_id_lens, 'to'):
                    prompt_id_lens = torch.tensor(prompt_id_lens, device=device, dtype=torch.long)
                else:
                    prompt_id_lens = prompt_id_lens.to(device)

                # 条件化三路：得到 (p+y1), (p+y1+y2), (p+y2+y1)
                (ids1, att1, lab1), (ids2, att2, lab2), (ids3, att3, lab3) = \
                    self.get_conditioned_inputs_and_labels(chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)

                # 1) SFT 锚：- avg_log πθ(y1|x)
                sum_y1x = self.compute_logprob(self.model, ids1, att1, lab1)   # 仅监督 y1 段
                avg_y1x = self._avg_from_labels(sum_y1x, lab1)
                sft_term = (-avg_y1x).mean()

                # 2) 条件化 pairwise-logit：y2|y1 与 y1|y2（avg 后做 logit 差）
                sum_y2gy1 = self.compute_logprob(self.model, ids2, att2, lab2)
                sum_y1gy2 = self.compute_logprob(self.model, ids3, att3, lab3)
                avg_y2gy1 = self._avg_from_labels(sum_y2gy1, lab2)
                avg_y1gy2 = self._avg_from_labels(sum_y1gy2, lab3)

                logit_pair = self._safe_logit_from_avglogp(avg_y2gy1) - self._safe_logit_from_avglogp(avg_y1gy2)
                pair_term = torch.nn.functional.softplus(-logit_pair).mean()  # -log σ(·)

                loss = sft_term + self.lambda_pair * pair_term

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # 我们希望 logit(p(y2|y1,x)) - logit(p(y1|y2,x)) < 0 才是"好"
                acc = (logit_pair < 0).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()

                logs = {
                    "loss": loss.item(),
                    "acc": acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    "orpo_sft": sft_term.item(),
                    "orpo_pair": pair_term.item(),
                    "orpo_margin_mean": logit_pair.mean().item(),
                }
                logs = self.strategy.all_reduce(logs)
                step_bar.set_postfix(logs)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    logs["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0.0
                    acc_sum = 0.0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

