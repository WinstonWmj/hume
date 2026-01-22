# CometPolicy 训练流程分析

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CometPolicy 架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────┐    ┌────────────────────────────────┐    │
│  │      PI0Pytorch (冻结)        │    │    CometValueQueryHead (训练)   │    │
│  │  ┌─────────────────────────┐ │    │  ┌────────────────────────────┐│    │
│  │  │ PaliGemmaWithExpertModel│ │───>│  │      VQHBackbone           ││    │
│  │  │  - SigLIP VisionEncoder │ │    │  │  (4层 GemmaModel)          ││    │
│  │  │  - PaliGemma (2B)       │ │    │  └────────────────────────────┘│    │
│  │  │  - GemmaExpert (300M)   │ │    │  ┌────────────────────────────┐│    │
│  │  └─────────────────────────┘ │    │  │        CalQL               ││    │
│  │  ┌─────────────────────────┐ │    │  │  - Policy Network          ││    │
│  │  │  Action Projections     │ │    │  │  - Critics (Twin Q)        ││    │
│  │  │  - action_in_proj       │ │    │  │  - Target Critics          ││    │
│  │  │  - action_out_proj      │ │    │  │  - Temperature             ││    │
│  │  │  - time_mlp             │ │    │  └────────────────────────────┘│    │
│  │  └─────────────────────────┘ │    └────────────────────────────────┘    │
│  └──────────────────────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 训练脚本调用流程

```
train_vqh_s1.behavior.comet.sh
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  accelerate launch src/hume/training/train_vqh_s1.py          │
│    --policy.type=comet                                        │
│    --pretrained_comet_path=/.../pi05-b1kpt12-cs32             │
│    --config_path=config/behavior_comet.json                   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  train_vqh_s1.py::train()                                     │
│    1. cfg.validate()           # 验证配置                      │
│    2. Accelerator()            # 初始化分布式训练               │
│    3. make_dataset(cfg)        # 创建数据集                    │
│    4. make_policy()            # 创建 CometPolicy             │
│    5. 加载 comet 预训练权重到 PI0 模型                         │
│    6. make_optimizer_and_scheduler() # 创建优化器              │
│    7. Training Loop                                           │
└───────────────────────────────────────────────────────────────┘
```

## 3. 核心函数调用链

### 3.1 策略创建流程

```python
# train_vqh_s1.py:309-321
policy_type = getattr(cfg.policy, 'type', 'hume')  # 'comet'
policy_cls = CometPolicy

policy = make_policy(
    cfg=cfg.policy,
    ds_meta=dataset.meta,
    policy_cls=CometPolicy,
)
```

```
make_policy()
    │
    ▼
CometPolicy.__init__(config, dataset_stats)
    │
    ├──> PI0Pytorch(pi0_config)
    │        │
    │        ├──> PaliGemmaWithExpertModel()
    │        │        ├──> PaliGemmaForConditionalGeneration (VLM)
    │        │        └──> GemmaForCausalLM (Action Expert)
    │        │
    │        ├──> action_in_proj (Linear: 32 -> 1024)
    │        ├──> action_out_proj (Linear: 1024 -> 32)
    │        └──> time_mlp (for pi05 AdaRMS)
    │
    └──> CometValueQueryHead(paligemma_with_expert, config)
             │
             ├──> VQHBackbone (4层 GemmaModel)
             ├──> CalQL (Actor-Critic + CQL)
             └──> query_embedding (可学习参数)
```

### 3.2 加载预训练权重

```python
# train_vqh_s1.py:324-351
weight_paths = glob.glob(os.path.join(cfg.pretrained_comet_path, "*.safetensors"))

for weight_path in weight_paths:
    checkpoint_state_dict = safetensors.torch.load_file(weight_path)
    
    # 过滤匹配的权重
    filtered_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in pi0_state_dict and value.shape == pi0_state_dict[key].shape:
            filtered_state_dict[key] = value
    
    policy.pi0_model.load_state_dict(filtered_state_dict, strict=False)

# 冻结 PI0 模型
if cfg.freeze_pi0:
    policy._set_requires_grad()  # PI0 所有参数 requires_grad = False
```

### 3.3 训练循环 (Training Loop)

```python
# train_vqh_s1.py:540-560
for _ in range(step, cfg.steps):
    batch = next(dl_iter)
    
    train_tracker, output_dict = update_policy(
        train_tracker,
        policy,
        batch,
        optimizers,
        grad_clip_norm,
        accelerator,
        lr_schedulers,
        cfg
    )
```

### 3.4 策略更新流程 (update_policy)

```
update_policy()
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  policy.forward(batch)                                          │
│    │                                                            │
│    ├─1─> normalize_inputs(batch)                                │
│    ├─2─> normalize_targets(batch)                               │
│    │                                                            │
│    ├─3─> prepare_images(batch) ──> images, img_masks            │
│    ├─4─> prepare_language(batch) ──> lang_tokens, lang_masks    │
│    ├─5─> prepare_images(batch, "*.vqh") ──> vqh_images          │
│    │                                                            │
│    └─6─> value_query_head.forward(...)                          │
│              │                                                  │
│              ▼                                                  │
│         ┌─────────────────────────────────────────────────┐     │
│         │  CometValueQueryHead.forward()                  │     │
│         │    │                                            │     │
│         │    ├─> process_next_obs()   # 拼接当前和下一观测 │     │
│         │    │     images: [B, ...] + [B, ...] = [2B, ...]│     │
│         │    │                                            │     │
│         │    ├─> embed_prefix()       # 图像+语言嵌入     │     │
│         │    │     ├─> embed_image()  (SigLIP)           │     │
│         │    │     ├─> embed_language_tokens()           │     │
│         │    │     └─> 插入 query_embedding              │     │
│         │    │                                            │     │
│         │    ├─> vqh_backbone.forward()                  │     │
│         │    │     └─> 4层 GemmaModel Transformer        │     │
│         │    │                                            │     │
│         │    ├─> 提取 query_embedding 位置的输出         │     │
│         │    │     encoded_obs = out[:B]                 │     │
│         │    │     encoded_next_obs = out[B:]            │     │
│         │    │                                            │     │
│         │    └─> calql(cal_ql_batch)                     │     │
│         │          ├─> temperature_loss                  │     │
│         │          ├─> policy_loss (Actor)               │     │
│         │          └─> critic_loss (Twin Q + CQL)        │     │
│         └─────────────────────────────────────────────────┘     │
│                                                                  │
│  返回: total_loss, temperature_loss, policy_loss, critic_loss   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  反向传播和优化器更新                                            │
│    │                                                            │
│    ├─> accelerator.backward(                                    │
│    │       chunk_loss + temperature_loss + policy_loss +        │
│    │       critic_loss                                          │
│    │   )                                                        │
│    │                                                            │
│    ├─> trunk_optimizer.step()                                   │
│    ├─> actor_optimizer.step()                                   │
│    ├─> critic_optimizer.step()                                  │
│    └─> temperature_optimizer.step()                             │
│                                                                  │
│  目标网络软更新 (每 target_critic_update_period 步):            │
│    └─> soft_update(target_critics, critics, tau)                │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 数据流图

```
                          Batch 数据
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
   observation.images    observation.state    action/reward
   [B, 3, H, W] × N          [B, 25]          [B, T, 23]
         │                    │                    │
         ▼                    │                    │
   resize_with_pad()          │                    │
   normalize [-1,1]           │                    │
         │                    │                    │
         ▼                    ▼                    │
   ┌─────────────────────────────────────────┐    │
   │     PaliGemmaWithExpertModel            │    │
   │                                         │    │
   │   SigLIP Vision Encoder                 │    │
   │   [B, 256, 2048] (image tokens)         │    │
   │           │                             │    │
   │           ▼                             │    │
   │   + Language Tokens [B, T_lang, 2048]   │    │
   │           │                             │    │
   │           ▼                             │    │
   │   + Query Embedding [B, 1, 2048]        │    │
   │           │                             │    │
   │           ▼                             │    │
   │   Attention Masks (causal)              │    │
   └─────────────────────────────────────────┘    │
                    │                             │
                    ▼                             │
         ┌──────────────────────┐                 │
         │    VQHBackbone       │                 │
         │   (4层 Gemma)        │                 │
         │                      │                 │
         │  输入: [2B, S, 2048] │                 │
         │  输出: [2B, S, 2048] │                 │
         └──────────────────────┘                 │
                    │                             │
                    ▼                             │
         提取 query position 的输出               │
         encoded_obs: [B, 2048]                   │
         encoded_next_obs: [B, 2048]              │
                    │                             │
                    ▼                             ▼
         ┌─────────────────────────────────────────────┐
         │                   CalQL                     │
         │                                             │
         │  Policy Network (Actor):                    │
         │    encoded_obs ──> μ, σ ──> actions        │
         │                                             │
         │  Critics (Twin Q):                          │
         │    [encoded_obs, actions] ──> Q1, Q2       │
         │                                             │
         │  TD Target:                                 │
         │    r + γ * target_Q(next_obs, π(next_obs)) │
         │                                             │
         │  CQL Regularization:                        │
         │    penalize OOD actions                     │
         └─────────────────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────────────────────────────┐
         │              Losses                         │
         │                                             │
         │  temperature_loss: 自动调节熵正则化强度      │
         │  policy_loss: max E[Q(s,a)] - α*H(π)       │
         │  critic_loss: TD error + CQL penalty        │
         └─────────────────────────────────────────────┘
```

## 5. 可训练参数分组

| 参数组 | 模块 | 优化器 | 学习率 |
|--------|------|--------|--------|
| trunk | VQHBackbone + query_embedding | trunk_optimizer | 5e-5 |
| actor | CalQL.policy (Actor Network) | actor_optimizer | 1e-5 |
| critics | CalQL.critics (Twin Q Networks) | critic_optimizer | 1e-5 |
| temperature | CalQL.temperature | temperature_optimizer | 2e-5 |

**注意**: PI0Pytorch 模型完全冻结 (`freeze_pi0=True`)，不参与训练。

## 6. 关键配置参数

```json
{
    "policy": {
        "type": "comet",
        "chunk_size": 32,           // PI0 action horizon
        "n_action_steps": 32,       // 预测的动作步数
        "vqh_chunk_size": 32,       // VQH 使用的动作片段长度 (no receding horizon)
        "next_obs_offset": 32,      // TD learning 的时间偏移 (应与 vqh_chunk_size 一致)
        "pi05": true,               // 使用 pi0.5 版本 (AdaRMS)
        "freeze_pi0": true,         // 冻结 PI0 backbone
        "train_vqh_only": true,     // 只训练 ValueQueryHead
        
        // 学习率设置 (针对小参数量的 VQHBackbone)
        "optimizer_lr": 2e-4,       // trunk 学习率
        "actor_lr": 3e-4,           // Actor 网络学习率
        "critic_lr": 3e-4,          // Critic 网络学习率
        "temp_lr": 3e-4,            // Temperature 学习率
        
        // CalQL 设置 (action_dim = 32*23 = 736)
        "cql_alpha": 5.0,           // CQL penalty 权重
        "cql_n_actions": 10,        // OOD 动作采样数量
        "target_entropy_coef": -0.2, // target_entropy = coef * action_dim (高维需要更保守)
        "discount": 0.99,           // TD target 折扣因子
        
        // CalQL 网络架构 (增强以处理高维动作空间)
        "policy_hidden_dims": [512, 512, 256],   // Policy 网络: 3层 MLP
        "critic_hidden_dims": [1024, 512, 256],  // Critic 网络: 3层 MLP (更宽)
        "use_layer_norm": true                   // LayerNorm 提升训练稳定性
    }
}
```

## 7. 文件结构

```
hume/
├── config/
│   └── behavior_comet.json          # CometPolicy 专用配置
├── scripts/
│   └── train_vqh_s1.behavior.comet.sh  # 训练启动脚本
└── src/hume/
    ├── models/
    │   ├── configuration_hume.py    # CometConfig 定义
    │   ├── modeling_hume.py         # CometPolicy, CometValueQueryHead
    │   ├── value_query.py           # VQHBackbone, CalQL
    │   └── comet_models/
    │       ├── __init__.py
    │       ├── gemma_config.py      # Gemma 配置
    │       ├── gemma_pytorch.py     # PaliGemmaWithExpertModel
    │       └── pi0_pytorch.py       # PI0Pytorch 主模型
    └── training/
        └── train_vqh_s1.py          # 训练脚本
```

## 8. 与原 HumePolicy 的对比

| 特性 | HumePolicy | CometPolicy |
|------|------------|-------------|
| 主干网络 | System2 + FastVisuoMatching | PI0Pytorch |
| 训练阶段 | 两阶段 (S1 + S2) | 单阶段 (VQH only) |
| 预训练权重 | Hume-System2 | openpi-comet |
| 冻结策略 | 可选冻结 S2 | 完全冻结 PI0 |
| Action Expert | 自定义 Gemma Expert | 内置 Gemma Expert (pi0.5) |
| Vision Encoder | SigLIP + DINOv2 | SigLIP only |

