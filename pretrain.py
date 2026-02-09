#!/usr/bin/env python3
"""Pre-training script for KanyeGPT.

This script trains a transformer model on raw Kanye West lyrics,
creating a "base model" that can later be fine-tuned on synthetic data.
"""

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PreTrainConfig
from model import GPTLanguageModel
from data import TextDataset
from optimizer import Muon, get_optimizer


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

    Features:
    - Per-step loss tracking and logging
    - Loss data persistence to JSON for analysis
    - Best model saving with iteration-based naming
    - Periodic evaluation on train/validation sets

    TODO: You can extend this class to add:
    - Learning rate scheduling (cosine decay, warmup)
    - Gradient clipping for training stability
    - Early stopping based on validation loss
    - TensorBoard/WandB logging
    """

    def __init__(self, model, dataset, config, optimizer_name: str = "muon"):
        self.model = model
        self.dataset = dataset
        self.config = config

        # Use Muon optimizer for better training stability and faster convergence
        # Muon uses momentum SGD for 2D matrices and Adam for 1D/embedding params
        if optimizer_name.lower() == "muon":
            self.optimizer = Muon(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.95,
                nesterov=True,
                wd=0.1,  # Weight decay
                lr_1d=0.1,  # Lower LR for 1D/embedding params
            )
        else:
            # Fallback to AdamW if specified
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.1,
            )

        # Training state
        self.iterations = 0
        self.best_val_loss = float('inf')

        # Loss tracking
        self.train_losses = []  # Training loss at every step
        self.val_losses = []    # Validation loss at evaluation intervals
        self.val_iterations = []  # Iteration numbers for val losses

        # Create directories for saving
        Path(config.losses_dir).mkdir(parents=True, exist_ok=True)
        Path(config.best_model_dir).mkdir(parents=True, exist_ok=True)

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

    def log_step_loss(self, loss: float):
        """Log and save loss for every training step."""
        self.train_losses.append(loss)

        # Log to console (every step - may be verbose)
        print(f"Step {self.iterations:4d} | Train Loss: {loss:.4f}")

        # Persist losses to file after every step
        self.save_losses()

    def save_losses(self):
        """Save all accumulated losses to JSON file."""
        loss_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_iterations': self.val_iterations,
            'best_val_loss': self.best_val_loss,
            'current_iteration': self.iterations,
        }

        loss_file = Path(self.config.losses_dir) / 'losses.json'
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=2)

    def save_best_model(self, val_loss: float):
        """Save best model with iteration-based naming."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            # Save with iteration number in filename
            model_path = Path(self.config.best_model_dir) / f'best_iter_{self.iterations}.pth'

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'iterations': self.iterations,
                'best_val_loss': self.best_val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_iterations': self.val_iterations,
            }, model_path)

            print(f"  → Best model saved: {model_path} (val_loss: {val_loss:.4f})")

            # Also save a symlink/copy as the current best
            current_best = Path(self.config.best_model_dir) / 'best_current.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'iterations': self.iterations,
                'best_val_loss': self.best_val_loss,
            }, current_best)

    def train(self):
        """Main training loop."""
        print_section("Starting Pre-Training")
        print(f"Device: {self.config.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocabulary size: {self.dataset.tokenizer.vocab_size}")
        print(f"Losses will be saved to: {self.config.losses_dir}/losses.json")
        print(f"Best models will be saved to: {self.config.best_model_dir}/")
        print_separator()

        for iter in range(self.config.max_iters):
            self.iterations = iter

            # Training step with per-step logging
            loss = self.train_step()
            self.log_step_loss(loss)

            # Periodic evaluation
            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                losses = self.evaluate()
                train_loss = losses['train']
                val_loss = losses['val']

                print(f"  └─ Eval | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

                # Track validation losses
                self.val_losses.append(val_loss)
                self.val_iterations.append(iter)

                # Save best model
                self.save_best_model(val_loss)

        # Final evaluation
        print_section("Training Complete!")
        losses = self.evaluate()
        print(f"Final | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final losses
        self.save_losses()
        print(f"\nLoss data saved to: {self.config.losses_dir}/losses.json")

    def save_checkpoint(self, path: str):
        """Save model state."""
        # Ensure models directory exists
        Path(path).parent.mkdir(exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iterations': self.iterations,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_iterations': self.val_iterations,
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
    trainer = Trainer(model, dataset, config, optimizer_name=config.optimizer)
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
