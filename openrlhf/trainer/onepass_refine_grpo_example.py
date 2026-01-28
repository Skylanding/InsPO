# 使用示例：OnePassRefineGRPOTrainer
"""
训练期两次采样（x->y1->y2），推理期一次采样（仅 x->y）的GRPO训练器

核心思想：
- 采样：仍然用 (x -> y1 -> y2) 产生训练样本
- 优化：但前向与损失都在 [x||y2] 上计算（不是 [x||y1||y2]）
- 优势：ΔR = R(x,y1,y2) - R(x,y1) 由 grader 提供并做组内归一
- KL/比率：仅在 y2 段统计，锚定到 reference 的一遍式上下文 x
"""

from trainers.onepass_refine_grpo_trainer import OnePassRefineGRPOTrainer, OnePassRefineConfig
from graders.rlvr_mlp import RLVRGrader, SimpleMLPGrader

# 示例：用 RLVRGrader（把你的 judge/reward model 封装成 callable）
def my_verifier(texts: List[str]) -> List[float]:
    """
    接入你的 RLVR / RM，返回分数
    这里用简单的示例实现
    """
    scores = []
    for text in texts:
        # 简单的启发式评分：检查代码格式和长度
        if "```" in text and len(text) > 50:
            scores.append(0.8)
        elif "def " in text or "class " in text:
            scores.append(0.6)
        else:
            scores.append(0.2)
    return scores

# 创建 grader
grader = RLVRGrader(base_callable=my_verifier)

# 配置
config = OnePassRefineConfig(
    model_name_or_path="Qwen/Qwen2-0.5B-Instruct",
    loss_type="dapo",
    draft_source="ref",                    # 草稿来源：reference model
    num_refinements_per_prompt=4,          # 每个 prompt 的 y2 数量
    use_sequence_level_is=True,            # 序列级重要性采样
    clip_ratio_max=2.0,                    # 比率裁剪上限
    beta=0.01,                             # KL 惩罚系数
    
    # 可选增强
    use_negative_draft_push=True,          # 负样本下压
    neg_push_coef=0.25,                    # 负样本系数
    use_consistency_kl=True,               # 一致性 KL
    consistency_kl_coef=0.1,               # 一致性 KL 系数
    
    # 基础参数
    max_completion_length=64,              # 最大生成长度
    do_sample=True,                        # 是否采样
    temperature=1.0,                       # 温度
    top_p=0.9,                            # top-p
    scale_rewards="group",                 # 奖励标准化方式
    reward_clip_range=(-10.0, 10.0),     # 奖励裁剪范围
)

# 创建训练器
trainer = OnePassRefineGRPOTrainer(
    grader=grader,
    model=model,                           # 当前策略模型
    ref_model=ref_model,                   # 参考模型
    strategy=strategy,                     # 分布式策略
    tokenizer=tokenizer,                   # 分词器
    optim=optimizer,                       # 优化器
    train_dataloader=train_loader,         # 训练数据加载器
    eval_dataloader=eval_loader,           # 验证数据加载器
    scheduler=scheduler,                   # 学习率调度器
    reward_funcs=[],                       # ΔR 由 grader 提供
    config=config,                         # 配置
)

# 开始训练
trainer.train()

# ===== 实验建议 =====
"""
A/B 对比实验：

A = 原始 GRPOTrainer（单段普通 GRPO）
B = OnePassRefineGRPOTrainer（两次采样训练，单次采样推理）

消融实验开关：
1. use_negative_draft_push（关/开）
2. use_consistency_kl（关/开）  
3. use_sequence_level_is（token vs sequence）

诊断指标：
- mean_deltaR：平均细化优势
- y2_active_tokens：y2 段活跃 token 数
- kl_loss：KL 散度损失
- consistency_kl：一致性 KL 损失
- neg_push_loss：负样本下压损失

离线评测：
- AE2 LC/WR：代码通过率
- AH：人类评估
- MT-Bench：多轮对话评估
"""

# ===== 关键差异总结 =====
"""
与原始 RefineGRPOTrainer 的区别：

1. 采样策略：相同
   - 仍然用 (x -> y1 -> y2) 产生训练样本

2. 优化目标：不同
   - 原始：优化 π(y2|x,y1)，推理需要两次采样
   - 本实现：优化 π(y|x)，推理只需一次采样

3. 前向计算：不同
   - 原始：在 [x||y1||y2] 上计算损失
   - 本实现：在 [x||y2] 上计算损失

4. 优势计算：相同
   - 都使用 ΔR = R(x,y1,y2) - R(x,y1)

5. 推理效率：不同
   - 原始：推理时需要 x->y1->y2 两次生成
   - 本实现：推理时只需 x->y 一次生成

这样能在不改变推理成本的前提下，把"基于两次采样的细化优势"蒸馏进一遍式策略。
"""
