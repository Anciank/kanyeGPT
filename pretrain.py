#!/usr/bin/env python3
"""Pre-training script for KanyeGPT.

This script trains a transformer model on raw Kanye West lyrics,
creating a "base model" that can later be fine-tuned on synthetic data.
"""

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PreTrainConfig
from model import GPTLanguageModel
from data import TextDataset


def print_separator(char: str = "=", length: int = 60):
    """Print a visual separator line."""
    print(char * length)


def print_section(title: str):
    """Print a formatted section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


class Trainer:
    """
    Training loop with logging and checkpointing.

    TODO: You can extend this class to add:
    - Learning rate scheduling (cosine decay, warmup)
    - Gradient clipping for training stability
    - Early stopping based on validation loss
    - TensorBoard/WandB logging
    """

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        # Optimizer with weight decay (AdamW)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )

        # Training state
        self.iterations = 0
        self.best_val_loss = float('inf')

    def train_step(self) -> float:
        """Single training step."""
        self.model.train()

        # Get batch
        xb, yb = self.dataset.get_batch('train')

        # Forward pass
        logits, loss = self.model(xb, yb)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self) -> dict:
        """Evaluate on train and validation sets."""
        return self.model.estimate_loss(
            self.dataset.get_batch,
            eval_iters=self.config.eval_iters
        )

    def train(self):
        """Main training loop."""
        print_section("Starting Pre-Training")
        print(f"Device: {self.config.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocabulary size: {self.dataset.tokenizer.vocab_size}")
        print_separator()

        for iter in range(self.config.max_iters):
            # Periodic evaluation
            if iter % self.config.eval_interval == 0:
                losses = self.evaluate()
                train_loss = losses['train']
                val_loss = losses['val']

                print(f"Step {iter:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(self.config.model_path)

            # Training step
            self.train_step()

        # Final evaluation
        print_section("Training Complete!")
        losses = self.evaluate()
        print(f"Final | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    def save_checkpoint(self, path: str):
        """Save model state."""
        # Ensure models directory exists
        Path(path).parent.mkdir(exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iterations': self.iterations,
            'best_val_loss': self.best_val_loss,
        }, path)

        print(f"Checkpoint saved to {path}")


def generate_sample(model, tokenizer, seed_text: str = "", length: int = 200):
    """Generate a sample from the model."""
    model.eval()

    # Encode seed text (or use newline as default start)
    if seed_text:
        context = torch.tensor([tokenizer.encode(seed_text)], dtype=torch.long)
    else:
        context = torch.tensor([[tokenizer.stoi['\n']]], dtype=torch.long)

    context = context.to(model.config.device)

    # Generate
    generated = model.generate(context, max_new_tokens=length, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].tolist())

    return generated_text


def main():
    """Main entry point."""
    print_section("KanyeGPT: Pre-Training Phase")

    # Load configuration
    config = PreTrainConfig()

    # Load dataset
    print_section("Loading Data")
    dataset = TextDataset(config.data_path, config)
    print(f"Dataset size: {len(dataset.train_data):,} training tokens")

    # Save tokenizer for fine-tuning phase
    dataset.save_tokenizer(config.meta_path)

    # Initialize model
    print_section("Initializing Model")
    model = GPTLanguageModel(dataset.tokenizer.vocab_size, config)
    model = model.to(config.device)

    # Create trainer and run training
    trainer = Trainer(model, dataset, config)
    trainer.train()

    # Generate sample
    print_section("Sample Generation")
    sample = generate_sample(model, dataset.tokenizer)
    print("-" * 60)
    print(sample)
    print("-" * 60)

    print_section("All Done!")
    print(f"Model saved to: {config.model_path}")
    print(f"Tokenizer saved to: {config.meta_path}")


if __name__ == "__main__":
    main()
