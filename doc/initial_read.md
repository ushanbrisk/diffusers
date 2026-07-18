# Wan2.2-TI2V-5B 源码阅读路线（由粗到细 4 层递进）

> 目标：在 ~1 小时内理解一次完整推理的架构，再按兴趣深入算法。
> 所有路径都以 `sglang_research` env 的 site-packages 为基准（即 `pip install -e .` 后真实加载的代码）。

---

## 0. 准备工作

你的研究目录里有两份代码：

```
/home/luke/code_project/wan-research/
├── diffusers/    # HF 实现的 Wan（sglang 实际加载用的）★ 主看
└── Wan2.2/       # 阿里官方参考实现（独立代码，不走 diffusers）→ 跟 diffusers 版对比看架构差异
```

> **Tip**：VSCode 打开 `wan-research/diffusers` 作为 workspace，Python 解释器选 `sglang_research`。
> 这样跳转、类型检查都基于你 clone 的本地源码，**0 风险**。

关键路径速查：

| 模块 | 路径 |
|---|---|
| Pipeline（T2V 总入口） | `diffusers/src/diffusers/pipelines/wan/pipeline_wan.py` |
| Pipeline（I2V） | `diffusers/src/diffusers/pipelines/wan/pipeline_wan_i2v.py` |
| DiT 主干 | `diffusers/src/diffusers/models/transformers/transformer_wan.py` |
| DiT 变体（VACE） | `diffusers/src/diffusers/models/transformers/transformer_wan_vace.py` |
| VAE（视频编解码） | `diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py` |

---

## Level 1（10 分钟）：Pipeline 总览 —— 一次推理的 7 个阶段

**文件**：`diffusers/src/diffusers/pipelines/wan/pipeline_wan.py`
**入口**：`class WanPipeline` → `def __call__(...)` 在 **L383**

### `__call__` 方法逐段解读（L383 - L671）

```python
383: def __call__(
384:     self,
385:     prompt: str | list[str] = None,
386:     negative_prompt: str | list[str] = None,
387:     height: int = 480,           # ★ 默认 480（sglang 跑 480p 路线）
388:     width: int = 832,            # ★ 默认 832（横向 16:9）
389:     num_frames: int = 81,        # ★ 81 帧 ≈ 5s @ 16fps
390:     num_inference_steps: int = 50,  # ★ 50 步去噪（也可以 30/40，越多越慢越精细）
391:     guidance_scale: float = 5.0,    # ★ CFG 强度，>1 才生效
392:     guidance_scale_2: float | None = None,  # ★ Wan2.2 第二阶段（低噪声）的 CFG
403:     max_sequence_length: int = 512,  # ★ T5 文本最大 token 数
404: ):
```

**7 个阶段的源码位置 + 注释**：

| 阶段 | 行号 | 代码 | 作用 |
|---|---|---|---|
| **① 输入校验** | `L478` | `# 1. Check inputs. Raise error if not correct` | 检查 prompt / height / width 是否合法，调 `self.check_inputs(...)` (L280) |
| **② 定义 batch 参数** | `L524` | `# 2. Define call parameters` | 把 `prompt`、`negative_prompt` 拍平成 batch，确定 device / dtype |
| **③ 文本编码** | `L532-547` | `# 3. Encode input prompt` + `prompt_embeds, negative_prompt_embeds = self.encode_prompt(...)` | **T5 encoder** 把 prompt 转成 embedding（shape `[B, 512, 4096]`），同时算 positive + negative（CFG 用） |
| **④ 准备 timesteps** | `L549` | `# 4. Prepare timesteps` | 调 `self.scheduler.set_timesteps(num_inference_steps)` 生成 50 个 t 值（flow matching 是均匀采样） |
| **⑤ 准备 latent** | `L553-572` | `# 5. Prepare latent variables` + `latents = self.prepare_latents(...)` | 在 latent 空间采样一个**纯噪声** tensor，shape `[B, 16, T, H/8, W/8]`（VAE 下采样 8 倍） |
| **⑥ 去噪循环** | `L573-642` | `# 6. Denoising loop` | **核心**：循环 50 次，每次跑一次 transformer + scheduler.step |
| **⑦ 解码 + 后处理** | `L660-661` | `video = self.vae.decode(latents, ...)` + `video = self.video_processor.postprocess_video(...)` | VAE 把 latent 解码回像素空间，转 numpy/PIL |

### 去噪循环（L573 - L642）展开 —— Wan2.2 的核心

Wan2.2 比 Wan2.1 **多了一个低噪声阶段**（dual-transformer），代码在这里分流：

```python
577:        if self.config.boundary_ratio is not None:
578:            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
579:        else:
580:            boundary_timestep = None
                                              # ↑ boundary_ratio 把时间步切两段：
                                              #   高噪声段（前 87.2%）→ self.transformer
                                              #   低噪声段（后 12.8%）→ self.transformer_2
                                              #   Wan2.1 没有这个 → boundary_timestep = None

583:        for i, t in enumerate(timesteps):
589:            if boundary_timestep is None or t >= boundary_timestep:
590:                # wan2.1 or high-noise stage in wan2.2
591:                current_model = self.transformer
592:                current_guidance_scale = guidance_scale
593:            else:
594:                # low-noise stage in wan2.2
595:                current_model = self.transformer_2
596:                current_guidance_scale = guidance_scale_2

598:            latent_model_input = latents.to(transformer_dtype)
                                              # ↑ cast 到 transformer 的 dtype（bf16）

599:            if self.config.expand_timesteps:
601:                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
602:                # batch_size, seq_len
603:                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                                              # ↑ 视频 diffusion 的 trick：每个 spatial patch 有自己的 timestep

607:            with current_model.cache_context("cond"):
608:                noise_pred = current_model(
609:                    hidden_states=latent_model_input,
610:                    timestep=timestep,
611:                    encoder_hidden_states=prompt_embeds,    # T5 的文本 embedding
612:                    attention_kwargs=attention_kwargs,
613:                    return_dict=False,
614:                )[0]
                                              # ↑ ★ 第一次 forward：条件分支（cond）
                                              #   输出 noise_pred = 模型预测的"噪声/速度"

616:            if self.do_classifier_free_guidance:
617:                with current_model.cache_context("uncond"):
618:                    noise_uncond = current_model(
619:                        hidden_states=latent_model_input,
620:                        timestep=timestep,
621:                        encoder_hidden_states=negative_prompt_embeds,  # 空 prompt 的 embedding
622:                        attention_kwargs=attention_kwargs,
623:                        return_dict=False,
624:                    )[0]
625:                noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
                                              # ↑ ★ CFG 公式：放大 cond 和 uncond 的差
                                              #   guidance_scale=5.0 时很激进

627:            # compute the previous noisy sample x_t -> x_t-1
628:            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                                              # ↑ ★ Flow matching 的欧拉步：x_{t-1} = x_t + dt * v(...)
                                              #   不同 scheduler 公式不同（Euler / FlowMatch / DPMSolver）
```

### 数据流图（看完这一节必须能画出来）

```
                        ┌─────────────────────────────────────────┐
prompt ("a cat walks") │                                         │
        ↓              │        WanPipeline.__call__             │
[T5 text encoder]      │                                         │
        ↓              └─────────────────────────────────────────┘
text_emb [B, 512, 4096]                          ↓
                              ┌────────────────────────────────────┐
        ┌────────────────────→│   for i, t in timesteps:         │
        │                     │     cond   = transformer(latent,  │
        │                     │                              t,   │
        │                     │                              text_emb)
        │                     │     uncond = transformer(latent,  │
        │                     │                              t,   │
        │                     │                              neg_emb)
        │                     │     noise   = uncond + scale*    │
        │                     │                (cond - uncond)   │
        │                     │     latent  = scheduler.step(...) │
        │                     └────────────────────────────────────┘
        │                                       ↓ × N (默认 50)
   latents [B, 16, T, H/8, W/8]                 │
        ↑                                       │
[randn 噪声]                                    │
                                                ↓
                              ┌────────────────────────────────────┐
                              │ vae.decode(latents) → video [B,3,T,H,W]
                              └────────────────────────────────────┘
```

**Level 1 完成标志**：你能向别人用 1 分钟讲清楚 Wan2.2 一次推理干了什么。

---

## Level 2（20 分钟）：Transformer 主干 —— DiT 内部数据变换

**文件**：`diffusers/src/diffusers/models/transformers/transformer_wan.py`（约 700 行）

按这个顺序读，**只看结构，不看数学**：

### 2.1 主类 `WanTransformer3DModel`（文件上半部）

```python
class WanTransformer3DModel(ModelMixin, ConfigMixin):
    """
    Wan 视频 DiT 模型。
    """
    def __init__(self, ...):
        # ① patch embedding（3D 版）
        self.patch_embedding = nn.Conv3d(in_channels, hidden_size, ...)
            # ↑ 把 [B, 16, T, H, W] latent → [B, hidden, T', H', W']
            #   下采样倍率通常 (1, 2, 2)，即 patch=(1,2,2)
            #   时间维度不变，高度宽度各 ×2

        # ② 时间 + 文本 embedding 融合
        self.time_text_embedding = WanTimeTextImageEmbedding(...)
            # ↑ 把 scalar timestep + 文本特征拼成一个调制向量

        # ③ N 个 block（默认 30）
        self.blocks = nn.ModuleList([
            WanTransformerBlock(...) for _ in range(num_layers)
        ])

        # ④ 输出 norm + projection
        self.norm_out = ...
        self.proj_out = nn.Linear(hidden_size, patch_size**2 * out_channels)
            # ↑ unpatchify：把 hidden 维度映射回 patch 像素
```

### 2.2 单个 Block `WanTransformerBlock.forward`

每个 block 含 3 个 sub-module：
```
hidden_states
    ↓
norm1 → modulation (timestep 注入) → WanAttention (spatial-temporal self-attn + cross-attn)
    ↓ residual
norm2 → modulation                → MLP (FFN)
    ↓ residual
output
```

### 2.3 完整 forward（你要找的方法）

在 `WanTransformer3DModel` 类里搜 `def forward(`：

```python
def forward(
    self,
    hidden_states: torch.Tensor,         # [B, 16, T, H, W] latent
    timestep: torch.Tensor,              # [B] 或 [B, seq_len] (expand_timesteps)
    encoder_hidden_states: torch.Tensor, # [B, 512, 4096] 文本 embedding
    attention_kwargs: dict | None = None,
    return_dict: bool = False,
):
    # ① patchify：[B, 16, T, H, W] → [B, T*H*W/p², hidden]
    hidden_states = self.patch_embedding(hidden_states)

    # ② timestep embedding
    temb = self.time_text_embedding(timestep, encoder_hidden_states)

    # ③ N 个 block 串行
    for block in self.blocks:
        hidden_states = block(hidden_states, temb, ...)

    # ④ unpatchify：[B, T*H*W/p², hidden] → [B, 16, T, H, W]
    hidden_states = self.unpatchify(hidden_states)
    return hidden_states
```

### 2.4 关键概念速记

| 名词 | 意思 | 在哪 |
|---|---|---|
| **patchify** | 把 latent 切成 3D patch（ViT 思路） | `patch_embedding` (Conv3d) |
| **timestep modulation** | 把 scalar 时间步注入到每个 block | `scale_shift` 参数 |
| **spatial-temporal attention** | 自注意力同时覆盖 T+H+W 三维 | `WanAttention.forward` |
| **cross-attention** | latent 序列 attend 到文本序列 | `WanAttention.forward` 里的 to_q / to_k/to_v |
| **RoPE 3D** | 旋转位置编码（T, H, W 各自频率） | `WanRotaryPosEmbed` |
| **unpatchify** | 把 patch 拼回 latent | `proj_out` Linear |

**Level 2 完成标志**：你能说出"Wan DiT 有 N 个 block，每个 block 含 self-attn + cross-attn + MLP，timestep 通过 modulation 注入"。

---

## Level 3（15 分钟）：VAE 视频编解码

**文件**：`diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py`

### 3.1 为什么有 VAE

视频帧是 `[B, 3, 168, 480, 832]`（5s × 16fps × 480p）= 0.21 G 像素 / 视频。
直接在像素空间做 diffusion，attention 计算量不可承受。
**VAE 把像素压缩到 latent** `[B, 16, 42, 60, 104]`（×8 空间压缩，×4 时间压缩），diffusion 在 latent 跑。

### 3.2 看哪些方法

| 方法 | 作用 | 关键点 |
|---|---|---|
| `encode(self, x)` | 视频 → latent | 输出 `latent_dist`，采样得到 latent；**causal temporal conv** 让 t 帧只依赖 0..t 帧（防止"看到未来"） |
| `_encode(self, x)` | 实际跑 encode | 看 `self.encoder`（一堆 3D conv + ResNet block） |
| `decode(self, z)` | latent → 视频 | 反向过程 |
| `tiled_encode` / `tiled_decode` | 切块编码/解码 | 显存爆了就调它，对应 sglang 启动时的 `--vae-tiling` flag |

### 3.3 VAE 为什么是显存大头

5B 模型在 latent 空间跑只要 ~10GB，但 VAE decode 把 42 帧 latent 一次性还原成 168 帧高清视频时，**3D conv 中间激活要全部存下来** → 720P 时峰值 23.20 GiB → OOM。

**解决方案**（已经在 sglang 里配了）：
- `--vae-tiling`：把视频切成小块分批 decode，拼回去
- `--text-encoder-cpu-offload`：text encoder 卸到 CPU

看 `tiled_decode` 方法理解切块逻辑。

**Level 3 完成标志**：你能解释"为什么需要 VAE" + "为什么 720P 会 OOM" + "tiling 怎么救场"。

---

## Level 4（按需深入）：关键算法

前三层完成后你已经懂整体架构了。下面这些是单点深入，按兴趣挑。

### 4.1 3D RoPE（旋转位置编码）

**位置**：`transformer_wan.py` → class `WanRotaryPosEmbed`

视频 = T + H + W 三维，每维需要独立的位置编码频率。

```python
def forward(self, hidden_states):
    # 1) 生成 T H W 三个维度的 grid
    grid_t = torch.arange(num_frames)
    grid_h = torch.arange(height)
    grid_w = torch.arange(width)
    grid = torch.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)   # [3, T, H, W]

    # 2) 嵌入到高维（half/half rotation）
    emb = freqs.tensor_product(grid, ...)  # 类似 RoPE 的 cos/sin

    # 3) 切片成 cos/sin pair
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin
```

**看点**：为什么 T 维频率最低（视频连续）、W 维频率最高（横向分辨率大）？

### 4.2 Attention 分类

**位置**：`transformer_wan.py` → class `WanAttention` (L175)

```python
class WanAttention(nn.Module, AttentionModuleMixin):
    def __init__(self, ...):
        self.to_q = nn.Linear(dim, dim)        # latent → query
        self.to_k = nn.Linear(dim, dim)        # 哪个序列的 K？
        self.to_v = nn.Linear(dim, dim)
```

**关键判断**：`to_k / to_v` 的输入是 `hidden_states` 还是 `encoder_hidden_states`？
- **self-attn**：`to_k = to_v = self.to_k(self.to_v_proj(hidden_states))` —— latent 自己 attend 自己
- **cross-attn**：`to_k = to_v = self.to_k(text_emb)` —— latent attend 到文本

### 4.3 Flow Matching scheduler step

**位置**：`diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`

Wan2.2 **不是** DDPM，是 **flow matching**。看 `step()` 方法：

```python
def step(self, model_output, timestep, sample, ...):
    # model_output 是模型预测的"速度" v(x_t, t)
    # flow matching: dx/dt = v(x_t, t)
    # 欧拉步: x_{t+dt} = x_t + dt * v(x_t, t)
    dt = next_sigma - sigma
    prev_sample = sample + dt * model_output
    return prev_sample
```

跟 DDPM 的 `x_{t-1} = (x_t - (1-α̅_t)ε / √α̅_t) / √α_t` 完全不一样。

### 4.4 Classifier-Free Guidance (CFG)

**位置**：pipeline_wan.py L616-625

```python
noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
#   = uncond + scale * (cond - uncond)
# scale=1 → 完全用 cond
# scale=5 → 激进放大（默认）
# scale>10 → 过饱和、伪影
```

**为什么需要 uncond forward**：让模型"知道不想要什么"，guidance_scale 越大"prompt 跟随度"越高，但画质会下降。

### 4.5 expand_timesteps（视频 diffusion 的 trick）

**位置**：pipeline_wan.py L599-605

```python
if self.config.expand_timesteps:
    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
    # batch_size, seq_len
    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
```

**意思**：把一个 scalar `t` 扩展成 `[seq_len]`，每个 spatial patch 有自己的 timestep。**为什么**：让模型可以"分块"用不同的去噪强度（边界 patch 用更小 t，更精细）。

---

## 阅读技巧

### 1. 数据 shape 速记表（贴在桌上）

```
prompt str
   ↓ T5 encoder
text_emb: [B, 512, 4096]                          # 512 tokens, 4096 dim

noisy latent: [B, 16, T, H/8, W/8]                # 例 [B, 16, 42, 60, 104]
   ↓ patchify (Conv3d, kernel=(1,2,2), stride=(1,2,2))
patches: [B, 16, T, H/16, W/16]                   # [B, 16, 42, 30, 52]
   ↓ reshape → [B, T*30*52, hidden]               # [B, 65520, 5120]
   ↓ add RoPE
   ↓ transformer × 30 blocks (self-attn + cross-attn + MLP)
out: [B, T*30*52, 5120]
   ↓ unpatchify Linear → reshape
latent: [B, 16, 42, 60, 104]
   ↓ VAE decode
video: [B, 3, 168, 480, 832]                      # 5s × 16fps × 480p
```

### 2. 改一行看效果

```bash
# 在 transformer_wan.py 任意 WanAttnProcessor.forward 里加：
print(f"[{__name__}] x.shape={x.shape}")

# 跑一次推理
python -c "
import torch
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained('/ssd/model_download/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/<snapshot>', torch_dtype=torch.bfloat16)
pipe.to('cuda')
out = pipe(prompt='a cat walks', num_inference_steps=5)  # 5 步就够看 shape
"
# 终端会刷出一堆 shape，比读 10 遍代码强
```

### 3. 跳过的优先级

| 看不懂 | 怎么办 |
|---|---|
| `WanAttnProcessor2_0` 里的 scaled_dot_product_attention | 跳，知道是 Flash Attention 就够 |
| `WanRotaryPosEmbed` 的 tensor_product | 跳，看 4.1 节概念就够 |
| `scheduler.sigma` 那堆数学 | 跳，看 4.3 节欧拉步就够 |
| `tiled_decode` 切块逻辑 | 跳，知道有 tiling 这回事就够 |

### 4. 对照看（diffusers vs Wan-Video 官方）

`wan-research/Wan2.2/` 里是阿里官方版本（独立代码），他们的 `wan/modules/model.py` 等文件可以跟 diffusers 版对着看，验证你对 DiT 结构的理解。

---

## 时间预算

| Level | 目标 | 预计时间 | 状态 |
|---|---|---|---|
| L1 | 画得出数据流图 | 10 min | ☐ |
| L2 | 解释 DiT 内部结构 | 20 min | ☐ |
| L3 | 解释 VAE 角色 | 15 min | ☐ |
| L4 | 选 1 个算法深入 | 30+ min | ☐ |

完成 L1-L3 ≈ **45 分钟**，架构就吃透了。L4 按兴趣推进。