import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model, base=1e5):
        super(SinusoidalPosEmb, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.float().view(-1, 1) * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).view(B, -1)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        x = self.dropout(F.softmax(attn, dim=-1)) @ v

        return self.W_o(
            rearrange(x, "b n l d -> b l (n d)")
        )


class FiLM(nn.Module):
    def __init__(self, d_model, d_t, *norm_args, eps=1e-6, **norm_kwargs):
        super(FiLM, self).__init__()
        self.norm = nn.LayerNorm(d_model, *norm_args, eps=eps, **norm_kwargs)

        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        g = self.gamma(t).unsqueeze(1)
        b = self.beta(t).unsqueeze(1)

        return g * self.norm(x) + b


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_size=None):
        super(SwiGLU, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.gate = nn.Linear(d_model, hidden_size, bias=False)
        self.hidden = nn.Linear(d_model, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class FlowMatchEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_emb, *args, sigma_min=1e-4, **kwargs):
        super(FlowMatchEmbedding, self).__init__(vocab_size, d_emb, *args, **kwargs)
        self.sigma_offset = 1 - sigma_min
        self.scale = sqrt(d_emb)

    @torch.inference_mode()
    def interpolate_flow(self, x_t, t, logits):
        B = logits.shape[0]
        p = F.softmax(logits, dim=-1)
        x_1 = F.normalize(p @ self.weight, dim=-1, p=2) * self.scale

        return (x_1 - self.sigma_offset * x_t) / (1 - self.sigma_offset * t).view(B, 1, 1)

    def forward(self, tokens):
        return F.normalize(super().forward(tokens), dim=-1, p=2) * self.scale
