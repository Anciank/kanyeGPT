"""Transformer model components for KanyeGPT."""

from typing import Union, Optional, Callable, Dict, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simpler and more efficient alternative to LayerNorm.
    It normalizes by the root mean square of the inputs without centering
    (subtracting the mean), which makes it computationally more efficient
    and often more stable for training large language models.

    Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # RMSNorm only has a learnable scale (weight), no bias
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm normalization.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor with same shape as input
        """
        # Store original dtype for restoration
        input_dtype = x.dtype

        # Compute RMS in float32 for numerical stability
        # RMS = sqrt(mean(x^2) + eps)
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalize and apply learnable scale
        return (x_float * rms).to(input_dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to queries and keys.

    Args:
        xq, xk: [B, T, n_head, head_dim]
        freqs_cis: [T, head_dim // 2] - frequencies for half the dims
    """
    # Split freqs into sin/cos
    cos = freqs_cis.cos()
    sin = freqs_cis.sin()

    # Expand freqs to match full head_dim by duplicating
    # [T, head_dim // 2] -> [T, head_dim]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Reshape for broadcasting: [1, T, 1, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Apply rotation
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # Precompute frequency tensor for half the dimensions
        # RoPE operates on pairs of dimensions, so we need dim//2 frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2)))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Cache the frequency tensor
        self.register_buffer('freqs', freqs)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get rotary frequencies for a given sequence length."""
        return self.freqs[:seq_len]


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking, RoPE, QK-Norm, and Flash Attention.

    This allows each token to attend to all previous tokens in the sequence,
    but not future tokens - essential for autoregressive generation.

    Features:
    - RoPE: Rotary positional embeddings for better position encoding
    - QK-Norm: RMSNorm normalization for queries and keys (training stability)
    - Flash Attention: Memory-efficient attention via PyTorch's SDPA
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # QKV projections: each outputs n_embd, then we split across heads
        # Using bias=False is standard for modern transformers
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        # QK-Norm: RMSNorm for queries and keys (more stable than LayerNorm)
        # Applied per-head to the head dimension
        self.q_norm = RMSNorm(self.head_size)
        self.k_norm = RMSNorm(self.head_size)

        # RoPE: Rotary positional embeddings
        self.rotary_emb = RotaryEmbedding(dim=self.head_size, max_seq_len=block_size)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)  # Output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute key, query, value projections
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Reshape for multi-head attention: [B, T, n_head, head_size]
        k = k.view(B, T, self.n_head, self.head_size)
        q = q.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)

        # QK-Norm: Normalize queries and keys BEFORE RoPE (standard practice)
        # This keeps Q and K on unit sphere, stabilizing attention logits
        k = self.k_norm(k)
        q = self.q_norm(q)

        # Apply RoPE to queries and keys
        freqs = self.rotary_emb(T)
        q, k = apply_rotary_emb(q, k, freqs)

        # Transpose for Flash Attention: [B, n_head, T, head_size]
        # PyTorch's scaled_dot_product_attention expects this format
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention via PyTorch's scaled_dot_product_attention
        # This automatically uses the most efficient implementation:
        # - Memory-efficient attention for forward
        # - Fused kernel for backward when available
        # - Falls back to standard implementation otherwise
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Use is_causal=True instead
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True  # Applies causal mask automatically
        )

        # Reshape back: [B, T, n_embd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class ReLUSquared(nn.Module):
    """ReLU² activation: squared ReLU for smoother gradients."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply ReLU then square the result
        return F.relu(x) ** 2


class FeedForward(nn.Module):
    """
    Feed-forward network layer with ReLU² activation.

    Projects embeddings to a higher dimension, applies non-linearity,
    then projects back. This allows the model to learn complex transformations.

    Uses ReLU² (squared ReLU) activation instead of GELU/SwiGLU for
    better efficiency and stability in the training regime.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand 4x (standard GPT practice)
            ReLUSquared(),                  # ReLU² activation
            nn.Linear(4 * n_embd, n_embd),  # Contract back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Complete Transformer block: Attention -> FeedForward,
    with residual connections and RMSNorm normalization.

    Uses Pre-Norm architecture (norm before sublayer) which is more stable
    for deep networks and modern training practices.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        # RMSNorm instead of LayerNorm for efficiency and stability
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture: norm -> sublayer -> residual
        x = x + self.sa(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Generative Pre-trained Transformer for text generation.

    Architecture:
    1. Token embeddings (no learned positional embeddings - using RoPE)
    2. Stack of Transformer blocks with RoPE, QK-Norm, and Flash Attention
    3. Final RMSNorm + linear projection to vocabulary

    Modern Features:
    - RMSNorm: Root mean square normalization (faster, more stable)
    - RoPE: Rotary positional embeddings for better position encoding
    - QK-Norm: Query-key normalization for training stability
    - ReLU²: Squared ReLU activation in the MLP
    - Flash Attention: Memory-efficient attention computation
    """

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.config = config

        # Embedding layers - only token embedding (RoPE handles positions)
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)

        # Transformer stack
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])

        # Output head with RMSNorm
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Initialize weights for better training dynamics
        self.apply(self._init_weights)

        # Tie embeddings: weight tying reduces parameters and can improve performance
        # The input embedding and output projection share the same weights
        self.token_embedding.weight = self.lm_head.weight

    def _init_weights(self, module):
        """
        Weight initialization following modern practices.

        - Linear/Embedding: Normal(0, 0.02)
        - RMSNorm: ones (weight is already initialized to ones in RMSNorm)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            # RMSNorm weight is already initialized to ones
            pass

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional, for computing loss)

        Returns:
            logits: Logits over vocabulary [B, T, V]
            loss: Cross-entropy loss (None if targets not provided)
        """
        B, T = idx.shape

        # Get token embeddings only (RoPE is applied inside attention)
        x = self.token_embedding(idx)

        # Transform through blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Context token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated token indices [B, T + max_new_tokens]
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def estimate_loss(self, get_batch_fn: Callable, eval_iters: int = 200) -> Dict[str, float]:
        """Estimate loss on train and validation sets."""
        self.eval()
        losses = {'train': torch.zeros(eval_iters), 'val': torch.zeros(eval_iters)}

        for split in ['train', 'val']:
            for k in range(eval_iters):
                X, Y = get_batch_fn(split)
                _, loss = self(X, Y)
                losses[split][k] = loss.item()

        self.train()
        return {k: v.mean().item() for k, v in losses.items()}
