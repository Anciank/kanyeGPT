"""Transformer model components for KanyeGPT."""

from typing import Union, Optional, Callable, Dict, Any
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    This allows each token to attend to all previous tokens in the sequence,
    but not future tokens - essential for autoregressive generation.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask - prevents attending to future tokens
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(head_size, n_embd)  # Output projection: head_size -> n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute key, query, value projections
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # Apply causal mask
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Apply to values
        out = wei @ v
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Feed-forward network layer.

    Projects embeddings to a higher dimension, applies non-linearity,
    then projects back. This allows the model to learn complex transformations.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand 4x (standard GPT practice)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Contract back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Complete Transformer block: Attention -> FeedForward,
    with residual connections and layer normalization.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture: norm -> sublayer -> residual
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Generative Pre-trained Transformer for text generation.

    Architecture:
    1. Token embeddings + positional embeddings
    2. Stack of Transformer blocks
    3. Final layer norm + linear projection to vocabulary
    """

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer stack
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # Initialize weights for better training dynamics
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Weight initialization following GPT-2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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

        # Get embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

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
