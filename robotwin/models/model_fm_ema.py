"""
Flow-matching VLA model with next-chunk prediction.

The next-chunk loss performs a second forward pass with detached or EMA
parameters through torch.func.functional_call. Gradients flow back to the first
forward pass only through obs_tokens_evolved.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Standard Transformer SinCos positional encoding.
    Returns: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)


# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VisualFeatureAdapter(nn.Module):
    """Perceiver-style adapter: learnable queries cross-attend to vision tokens, output (B, num_queries, hidden_dim)."""
    def __init__(self, feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        self.input_norm = nn.LayerNorm(feat_dim)
        self.feature_proj = nn.Linear(feat_dim, hidden_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, features, extra_query=None):
        """
        features:    (B, N, feat_dim)
        extra_query: (B, K, hidden_dim) optional extra query tokens (e.g. task-cond token)
                     concatenated after the learnable queries.
        Returns: (B, num_queries [+K], hidden_dim)
        """
        B = features.shape[0]
        memory = self.input_norm(features)
        memory = self.feature_proj(memory)
        tgt = self.query_embed.expand(B, -1, -1)
        if extra_query is not None:
            tgt = torch.cat([tgt, extra_query], dim=1)   # (B, num_queries + K, hidden_dim)
        out = self.transformer_decoder(tgt, memory)
        return out


class MultiLayerConcatFusion(nn.Module):
    """
    multi layer DINO feature fusion (concat mode):
      - LayerNorm
      - concat in channel dim → (B, N, L*feat_dim)
      - Linear / MLP  → (B, N, out_dim)
    """
    def __init__(self, feat_dim, num_layers, out_dim, proj_type="linear", pre_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        if pre_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        in_dim = feat_dim * num_layers
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported concat proj_type: {proj_type}")

    def forward(self, feats_list):
        """feats_list: List[(B, N, feat_dim)]"""
        assert len(feats_list) == self.num_layers, \
            f"MultiLayerConcatFusion expects {self.num_layers} layers, got {len(feats_list)}"
        if self.pre_norm:
            feats_list = [ln(f) for ln, f in zip(self.layer_norms, feats_list)]
        x = torch.cat(feats_list, dim=-1)   # (B, N, L*feat_dim)
        return self.proj(x)                 # (B, N, out_dim)


class FutureFeatureDecoder(nn.Module):
    def __init__(self, num_queries, hidden_dim, out_dim,
                 num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.out_dim = out_dim

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        self.kv_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_tokens):
        """
        cond_tokens: (B, M, hidden_dim) — adapter output or obs_tokens_evolved
        Returns: (B, num_queries, out_dim) predicted future frame tokens
        """
        B = cond_tokens.shape[0]
        memory = self.kv_norm(cond_tokens)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.decoder(tgt, memory)
        out = self.out_proj(self.out_norm(out))
        return out


def future_feature_cosine_loss(pred, target):
    """
    Per-token cosine loss = mean(1 - cos_sim).
    pred / target: (B, M, D). use float32
    """
    assert pred.shape == target.shape, \
        f"future-feat pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)} "
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    cos = (pred * target).sum(dim=-1)        # (B, M)
    return (1.0 - cos).mean()


class DiTBlock(nn.Module):
    """
    DiT block with Self-Attention + MLP, conditioned via adaLN-zero on time embedding.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, attn_mask=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb_t).chunk(6, dim=1)

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn1(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class VLAModel(nn.Module):
    def __init__(self,
                 action_dim=14,
                 proprio_dim=16,
                 hidden_dim=512,
                 num_heads=4,
                 depth=12,
                 action_len=32,
                 proprio_len=1,
                 num_registers=2,
                 dino_feat_dims=(1024,),     # feat_dim, len = feat layers
                 vlm_num_queries=64,
                 adapter_depth=2,
                 state_inject_start=0,
                 task_cond_dim=None,
                 # --- task condition insertion point ---
                 task_cond_insert="action_head",   # "action_head" | "adapter" | "both"
                 # --- multi layer feature fusion mode ---
                 fusion_mode="per_layer",
                 concat_proj_type="linear",
                 concat_pre_norm=True,
                 concat_out_dim=None,
                 # --- future frame pred ---
                 use_future_feat=False,
                 future_feat_attach="cond_tokens",    # "adapter" | "cond_tokens"
                 future_feat_num_queries=None,        # = future frame patch token num
                 future_feat_out_dim=None,            # = DINO hidden_size; None → = dino_feat_dims[0]
                 future_feat_depth=2,
                 future_feat_heads=4,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.num_registers = num_registers
        self.state_inject_start = state_inject_start
        self.num_dino_layers = len(dino_feat_dims)
        self.use_task_cond = task_cond_dim is not None

        # --- task condition insertion point ---
        assert task_cond_insert in ("action_head", "adapter", "both"), \
            f"Unknown task_cond_insert: {task_cond_insert}"
        self.task_cond_insert = task_cond_insert
        # Only meaningful when task cond is actually used.
        self.inject_task_cond_action_head = self.use_task_cond and task_cond_insert in ("action_head", "both")
        self.inject_task_cond_adapter = self.use_task_cond and task_cond_insert in ("adapter", "both")

        self.fusion_mode = fusion_mode
        assert fusion_mode in ("per_layer", "concat"), f"Unknown fusion_mode: {fusion_mode}"

        self.use_future_feat = use_future_feat
        self.future_feat_attach = future_feat_attach
        if use_future_feat:
            assert future_feat_attach in ("adapter", "cond_tokens"), \
                f"Unknown future_feat_attach: {future_feat_attach}"

        # --- 1. Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- 2. Projections & PosEmb ---
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)

        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))

        # Type embeddings: action, proprio + DINO
        self.type_emb_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_proprio = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        num_dino_type_emb = 1 if fusion_mode == "concat" else self.num_dino_layers
        self.type_emb_dino_layers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(num_dino_type_emb)
        ])

        nn.init.normal_(self.type_emb_action, std=0.02)
        nn.init.normal_(self.type_emb_proprio, std=0.02)
        for emb in self.type_emb_dino_layers:
            nn.init.normal_(emb, std=0.02)

        # --- 3. Register Tokens ---
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- 3b. Task Condition Token (DINOv3 CLS diff feature) ---
        if self.use_task_cond:
            self.task_cond_proj = nn.Sequential(
                nn.LayerNorm(task_cond_dim),
                nn.Linear(task_cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.type_emb_task_cond = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.type_emb_task_cond, std=0.02)

        # --- 4. DINO adapters ---
        if fusion_mode == "concat":
            # one adapter
            fusion_out_dim = concat_out_dim if concat_out_dim is not None else dino_feat_dims[0]
            self.concat_fusion = MultiLayerConcatFusion(
                feat_dim=dino_feat_dims[0],
                num_layers=self.num_dino_layers,
                out_dim=fusion_out_dim,
                proj_type=concat_proj_type,
                pre_norm=concat_pre_norm,
            )
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=fusion_out_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
            ])
        else:
            # per_layer: per adapter (NOT shared across layers)
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=feat_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
                for feat_dim in dino_feat_dims
            ])

        # --- 5. Core Transformer Blocks ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # --- 6. Output Head ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # --- 7. Future Feature Prediction Decoder (optional) ---
        if self.use_future_feat:
            assert future_feat_num_queries is not None, \
                "use_future_feat=True need future_feat_num_queries (=future frame patch token num)"
            ff_out_dim = future_feat_out_dim if future_feat_out_dim is not None else dino_feat_dims[0]
            self.future_feat_decoder = FutureFeatureDecoder(
                num_queries=future_feat_num_queries,
                hidden_dim=hidden_dim,
                out_dim=ff_out_dim,
                num_heads=future_feat_heads,
                num_layers=future_feat_depth,
            )

    def forward(self,
                t,
                noisy_actions,
                qpos_history=None,
                dino_features_list=None,    # List[Tensor(B, N_l, feat_dim_l)]
                task_cond=None,             # (B, task_cond_dim)
                obs_tokens_evolved=None,
                ):
        """
        mode 1 (normal forward): obs_tokens_evolved=None
        mode 2 (second forward): obs_tokens_evolved not None
        """
        if obs_tokens_evolved is not None:
            return self._forward_with_evolved_obs(t, noisy_actions, obs_tokens_evolved)

        t_emb = self.time_mlp(t)

        # 1. Action / Proprio tokens
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action

        x_proprio_real = self.proprio_proj(qpos_history) + \
                         self.proprio_pos_emb[:, :qpos_history.shape[1], :] + \
                         self.type_emb_proprio

        # State
        x_proprio = torch.zeros_like(x_proprio_real)

        tokens_list = [x_action, x_proprio]
        proprio_start = x_action.shape[1]
        proprio_end = proprio_start + x_proprio.shape[1]

        # 2. Register Tokens
        if self.num_registers > 0:
            regs = self.register_tokens.expand(noisy_actions.shape[0], -1, -1)
            tokens_list.append(regs)

        # 2b. Task Condition Token (projected once, injected per config)
        #     task_cond_insert:
        #       "action_head" → token appended to the DiT input sequence (below)
        #       "adapter"     → token used as an extra adapter query (2c)
        #       "both"        → both of the above
        tc_proj = None
        if self.use_task_cond:
            assert task_cond is not None, "use_task_cond=True, forward without task_cond"
            tc_proj = self.task_cond_proj(task_cond).unsqueeze(1)     # (B, 1, hidden_dim)

        if self.inject_task_cond_action_head:
            tokens_list.append(tc_proj + self.type_emb_task_cond)

        # extra adapter query token (task-cond injected into adapter module)
        adapter_extra_query = tc_proj if self.inject_task_cond_adapter else None

        # 3. Multi-layer DINO Feature Tokens
        #    per_layer / concat
        adapter_cond_list = []
        if dino_features_list is not None:
            assert len(dino_features_list) == self.num_dino_layers, \
                f"Expected {self.num_dino_layers} DINO feature layers, got {len(dino_features_list)}"
            if self.fusion_mode == "concat":
                fused = self.concat_fusion(dino_features_list)       # (B, N, fusion_out_dim)
                cond = self.dino_adapters[0](fused, extra_query=adapter_extra_query)   # (B, Q[+1], hidden_dim)
                adapter_cond_list.append(cond)
                tokens_list.append(cond + self.type_emb_dino_layers[0])
            else:
                for layer_idx, feats in enumerate(dino_features_list):
                    cond = self.dino_adapters[layer_idx](feats, extra_query=adapter_extra_query)
                    adapter_cond_list.append(cond)
                    tokens_list.append(cond + self.type_emb_dino_layers[layer_idx])

        adapter_cond_tokens = torch.cat(adapter_cond_list, dim=1) if adapter_cond_list else None

        # 4. Concat all tokens
        x = torch.cat(tokens_list, dim=1)

        # 5. Transformer Blocks
        state_injected = False

        for i, block in enumerate(self.blocks):
            if i == self.state_inject_start and not state_injected:
                x[:, proprio_start:proprio_end, :] = x_proprio_real
                state_injected = True

            x = block(x, t_emb)

        # 6. Output Head
        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)

        # Post-DiT tokens after the action tokens (proprio / registers / task-cond /
        # adapter obs tokens). Used as conditioning input for the future-feature
        # decoder when future_feat_attach="cond_tokens".
        obs_tokens_out = x[:, self.action_len:, :]

        out = {
            "final_pred": final_pred,
            "obs_tokens_evolved": obs_tokens_out,
            "adapter_cond_tokens": adapter_cond_tokens,
        }

        return out

    def _forward_with_evolved_obs(self, t, noisy_actions, obs_tokens_evolved):
        """
        second forward: input evolved obs tokens as condition
        """
        t_emb = self.time_mlp(t)

        # new noisy action tokens
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action

        # concat: [new_action_tokens, evolved_obs_tokens]
        x = torch.cat([x_action, obs_tokens_evolved], dim=1)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)

        return {"final_pred": final_pred}


def calc_flow_matching_loss(
    model,
    x1,
    dino_features_list,
    qpos_history,
    task_cond=None,
    time_sampler="uniform",
    time_mu=0.0,
    time_sigma=1.0,
    # Next-Chunk Prediction
    x1_next=None,
    use_next_chunk_pred=False,
    lambda_next_chunk=0.5,
    # EMA params for the Next-Chunk second forward (None → use the current params)
    ema_params=None,
    # Future Feature Prediction
    future_feat_target=None,
    use_future_feat=False,
    future_feat_attach="cond_tokens",
    lambda_future_feat=0.5,
):
    """
    Flow Matching Loss (Next-Chunk Prediction + Task Condition + Future-Feature Prediction)

    ema_params: optional dict of EMA action-head params (e.g. EMAParams.get_params()).
                When given, the Next-Chunk second forward runs with these (detached)
                params instead of a detached copy of the current model params.
    """
    device = x1.device
    bs = x1.shape[0]

    # 1. noise sample x0
    x0 = torch.randn_like(x1)

    # 2. timestamp t
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")

    # 3. t
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1

    # 4. target v
    target_v = x1 - x0

    # 5. first forward
    preds = model(t,
                  noisy_actions=x_t,
                  dino_features_list=dino_features_list,
                  task_cond=task_cond,
                  qpos_history=qpos_history)

    pred_v_final = preds["final_pred"]
    obs_tokens_evolved = preds["obs_tokens_evolved"]

    # ==================== Flow Matching Loss ====================
    loss_mse = F.mse_loss(pred_v_final, target_v)

    # ==================== Next-Chunk Prediction Loss ====================
    loss_next_chunk = torch.tensor(0.0, device=device)

    if use_next_chunk_pred and x1_next is not None:
        x0_next = torch.randn_like(x1_next)

        if time_sampler == "uniform":
            t_next = torch.rand(bs, device=device)
        elif time_sampler == "logit_normal":
            normal_samples_next = torch.randn(bs, device=device)
            normal_samples_next = normal_samples_next * time_sigma + time_mu
            t_next = torch.sigmoid(normal_samples_next)
        else:
            t_next = torch.rand(bs, device=device)

        t_next_expand = t_next.view(bs, 1, 1)
        x_t_next = (1 - t_next_expand) * x0_next + t_next_expand * x1_next
        target_v_next = x1_next - x0_next

        # The second forward uses frozen params (EMA when provided) and receives
        # no direct gradients from loss_next_chunk. obs_tokens_evolved remains in
        # the computation graph, so gradients flow back to the first forward.
        if ema_params is not None:
            frozen_params = {k: v.detach() for k, v in ema_params.items()}
        else:
            frozen_params = {k: v.detach() for k, v in model.named_parameters()}

        preds_next = torch.func.functional_call(
            model,
            frozen_params,
            args=(t_next,),
            kwargs={
                'noisy_actions': x_t_next,
                'obs_tokens_evolved': obs_tokens_evolved,
            }
        )

        pred_v_next = preds_next["final_pred"]
        loss_next_chunk = F.mse_loss(pred_v_next, target_v_next)

    # ==================== Future Feature Prediction Loss ====================
    loss_future_feat = torch.tensor(0.0, device=device)

    if use_future_feat and future_feat_target is not None:
        if future_feat_attach == "adapter":
            cond_tokens = preds["adapter_cond_tokens"]
            assert cond_tokens is not None, \
                "future_feat_attach='adapter' forward without adapter_cond_tokens (dino_features_list null?)"
        else:  # "cond_tokens"
            cond_tokens = obs_tokens_evolved

        # gradient back to adapter / DiT; target from DINO
        pred_future = model.future_feat_decoder(cond_tokens)        # (B, M, D_dino)
        loss_future_feat = future_feature_cosine_loss(pred_future, future_feat_target)

    # ==================== Total Loss ====================
    loss = loss_mse + lambda_next_chunk * loss_next_chunk + lambda_future_feat * loss_future_feat

    return loss, {
        "pred_v": pred_v_final,
        "target_v": target_v,
        "loss_mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
        "loss_next_chunk": loss_next_chunk.item() if isinstance(loss_next_chunk, torch.Tensor) else loss_next_chunk,
        "loss_future_feat": loss_future_feat.item() if isinstance(loss_future_feat, torch.Tensor) else loss_future_feat,
    }
