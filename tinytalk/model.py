"""A small decoder-only Transformer language model, initialized from scratch."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # PyTorch applies the causal mask and scaled dot product efficiently.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 block_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "d_model": d_model, "n_heads": n_heads,
                       "n_layers": n_layers, "block_size": block_size, "dropout": dropout}
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*(TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)))
        self.final_norm = nn.LayerNorm(d_model)
        self.language_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize)
        # Share token input and output weights (weight tying), a standard language-model design.
        self.language_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = token_ids.shape
        if length > self.config["block_size"]:
            raise ValueError(f"Input sequence length {length} exceeds model block_size {self.config['block_size']}")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        logits = self.language_head(self.final_norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 200, temperature: float = 0.7,
                 top_k: int = 8, eos_id: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config["block_size"]:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return token_ids
