import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from math import sqrt

from .util import FiLM, Attention, SinusoidalPosEmb, FlowMatchEmbedding


class FiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, n_heads, attn_dropout=0.0, ffn_dropout=0.0, norm_eps=1e-6, hidden_size=None):
        super(FiLMBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.attn = Attention(d_model, n_heads, dropout=attn_dropout)
        self.attn_norm = FiLM(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model)
        )
        self.ffn_norm = FiLM(d_model, d_t, eps=norm_eps)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x, t):
        x = x + self.attn(
            self.attn_norm(x, t)
        )

        return x + self.ffn_dropout(self.ffn(
            self.ffn_norm(x, t)
        ))


class FlowMatchTransformer(nn.Module):
    def __init__(
            self, length, pad_token, vocab_size, d_model, d_t, n_layers, n_heads,
            attn_dropout=0.0, ffn_dropout=0.0,
            sigma_min=1e-4
    ):
        super(FlowMatchTransformer, self).__init__()
        assert d_model % 2 == 0

        self.length = length
        self.pad_token = pad_token
        self.d_emb = d_model // 2

        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        pos_emb_scale = 1.0 / sqrt(d_model)
        self.pos_emb = nn.Parameter(
            torch.empty(1, length, d_model).uniform_(-pos_emb_scale, pos_emb_scale)
        )

        self.emb = FlowMatchEmbedding(vocab_size, self.d_emb, sigma_min=sigma_min)

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.layers = nn.ModuleList([
            FiLMBlock(d_model, d_t, n_heads, attn_dropout, ffn_dropout) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, vocab_size)

    @torch.inference_mode()
    def sample(self, tokens, num_steps=50, step="euler", guidance_scale=2.5):
        dt = 1 / num_steps
        L = tokens.shape[0]

        x_t = torch.randn(1, self.length, self.d_emb, device=tokens.device)

        clean = torch.zeros(1, self.length, self.d_emb)
        clean[:, :L] = self.emb(tokens.unsqueeze(0))

        if guidance_scale > 1.0:
            clean = torch.concatenate((
                torch.zeros_like(clean),
                clean
            ), dim=0)

        ts = torch.linspace(0, 1, num_steps, device=tokens.device).unsqueeze(1)
        for i in tqdm(range(num_steps)):
            pred_flow = self.pred_flow_guidance(x_t, ts[i], clean, guidance_scale)

            if step == "euler":
                x_t = x_t + dt * pred_flow
            elif step == "midpoint":
                x_t = x_t + dt * self.pred_flow_guidance(
                    x_t + 0.5 * dt * pred_flow, ts[i] + 0.5 * dt, clean, guidance_scale
                )
            elif step == "heun":
                if i == num_steps - 1:
                    x_t = x_t + dt * pred_flow
                else:
                    x_t = x_t + 0.5 * dt * (pred_flow + self.pred_flow_guidance(
                        x_t + dt * pred_flow, ts[i + 1], clean, guidance_scale
                    ))
            elif step == "stochastic":
                x_1 = x_t + (1 - ts[i]).view(1, 1, 1) * pred_flow
                if i == num_steps - 1:
                    x_t = x_1
                else:
                    next_t = ts[i + 1].view(1, 1, 1)
                    x_0 = torch.randn_like(x_t)
                    x_t = (1 - self.sigma_offset * next_t) * x_0 + next_t * x_1
            else:
                raise NotImplementedError

        pred_tokens = self.pred_logits(x_t, ts[-1], clean[-1].unsqueeze(0)).argmax(-1).squeeze()
        return torch.concatenate((
            tokens,
            pred_tokens[L:]
        ), dim=0)

    @torch.inference_mode()
    def pred_flow_guidance(self, x_t, t, clean, guidance_scale=2.5):
        if guidance_scale > 1.0:
            pred_uncond, pred_cond = self.pred_flow(
                torch.concat([x_t] * 2),
                torch.concat([t] * 2),
                clean
            ).chunk(2)
            pred_flow = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_flow = self.pred_flow(x_t, t, clean)

        return pred_flow

    @torch.inference_mode()
    def pred_flow(self, x_t, t, clean):
        logits = self.pred_logits(x_t, t, clean)

        return self.emb.interpolate_flow(x_t, t, logits)

    def pred_logits(self, x_t, t, clean=None):
        if clean is None:
            clean = torch.zeros_like(x_t)

        t_emb = self.t_model(t)
        x = torch.concatenate((
            x_t,
            clean
        ), dim=-1) + self.pos_emb

        for layer in self.layers:
            x = layer(x, t_emb)

        return self.head(x)

    def forward(self, tokens, clean_mask=None):
        B, L = tokens.shape
        if clean_mask is None:
            clean_mask = torch.zeros(B, L, device=tokens.device)

        x_1 = self.emb(tokens)
        clean = x_1 * clean_mask.view(B, L, 1)

        t = torch.rand(B, device=tokens.device).view(B, 1, 1)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - self.sigma_offset * t) * x_0 + t * x_1

        logits = self.pred_logits(x_t, t, clean)

        return F.cross_entropy(logits.transpose(1, 2), tokens)
