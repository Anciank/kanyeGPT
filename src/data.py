"""Data loading and preprocessing for KanyeGPT."""

import pickle
from pathlib import Path
from typing import Tuple, Callable

import torch


class Tokenizer:
    """Character-level tokenizer for the dataset."""

    def __init__(self, text: str):
        """Build vocabulary from text."""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Build mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Convert string to list of integers."""
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        """Convert list of integers to string."""
        return ''.join([self.itos[i] for i in l])

    def save(self, path: str):
        """Save tokenizer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'stoi': self.stoi,
                'itos': self.itos,
                'chars': self.chars,
                'vocab_size': self.vocab_size
            }, f)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.stoi = data['stoi']
        tokenizer.itos = data['itos']
        return tokenizer


class TextDataset:
    """
    Dataset for language modeling.

    Handles loading, tokenizing, and batching of text data.
    """

    def __init__(self, data_path: str, config):
        """Load and tokenize the dataset."""
        self.config = config
        self.device = config.device

        # Load raw text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Build tokenizer
        self.tokenizer = Tokenizer(text)

        # Tokenize entire dataset
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Train/val split (90/10)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of data.

        Args:
            split: 'train' or 'val'

        Returns:
            x: Input tokens [batch_size, block_size]
            y: Target tokens [batch_size, block_size]
        """
        data = self.train_data if split == 'train' else self.val_data

        # Random starting indices
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))

        # Gather batches
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.config.block_size + 1] for i in ix])

        return x.to(self.device), y.to(self.device)

    def save_tokenizer(self, path: str):
        """Save tokenizer for later use (e.g., fine-tuning)."""
        self.tokenizer.save(path)
        print(f"Tokenizers saved to {path}")

    @classmethod
    def load_tokenizer(cls, path: str):
        """Load a previously saved tokenizer."""
        return Tokenizer.load(path)
