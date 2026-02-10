#!/usr/bin/env python3
"""Pre-training script for KanyeGPT.

This script trains a transformer model on raw Kanye West lyrics,
creating a "base model" that can later be fine-tuned on synthetic data.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
from torch.nn import functional as F
import wandb

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PreTrainConfig
from model import GPTLanguageModel
from data import TextDataset
from optimizer import get_optimizer


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
    - Weights & Biases logging for visualization
    """

    def __init__(self, model, dataset, config, optimizer_name: str = "muon"):
        self.model = model
        self.dataset = dataset
        self.config = config

        # Optimizer (Muon or AdamW) wired to config hyperparameters
        self.optimizer = get_optimizer(
            model,
            optimizer_name=optimizer_name,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            lr_1d=config.lr_1d,
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

        # Initialize W&B (offline mode if not logged in)
        run_name = f"kanyegpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            wandb.init(
                project="kanyegpt",
                name=run_name,
                config={
                    'batch_size': config.batch_size,
                    'block_size': config.block_size,
                    'learning_rate': config.learning_rate,
                    'n_embd': config.n_embd,
                    'n_head': config.n_head,
                    'n_layer': config.n_layer,
                    'dropout': config.dropout,
                    'optimizer': config.optimizer,
                    'max_iters': config.max_iters,
                }
            )
            self.use_wandb = True
        except Exception as e:
            print(f"Warning: W&B not available ({e}). Training without W&B logging.")
            self.use_wandb = False

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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate(self) -> dict:
        """Evaluate on train and validation sets."""
        return self.model.estimate_loss(
            self.dataset.get_batch,
            eval_iters=self.config.eval_iters
        )

    def log_step_loss(self, loss: float):
        """Log loss for every training step (W&B handles persistence)."""
        self.train_losses.append(loss)

        # Log to W&B (if available)
        if self.use_wandb:
            wandb.log({'train_loss': loss}, step=self.iterations)

        # Log to console (every step - may be verbose)
        print(f"Step {self.iterations:4d} | Train Loss: {loss:.4f}")

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
        loss_file.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=2)

    def save_best_model(self, val_loss: float):
        """Save best model with iteration-based naming."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            # Save with iteration number in filename
            model_path = Path(self.config.best_model_dir) / f'best_iter_{self.iterations}.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)

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
        if self.use_wandb:
            print(f"W&B run: {wandb.run.url}")
        print_separator()

        no_improve_count = 0
        early_stopped = False

        for iter in range(self.config.max_iters):
            self.iterations = iter

            # Training step with per-step logging
            loss = self.train_step()
            self.log_step_loss(loss)

            # Periodic evaluation
            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                print(f"\n{'─'*60}")
                print(f"  📊 Evaluation at step {iter}")
                print(f"{'─'*60}")
                losses = self.evaluate()
                train_loss = losses['train']
                val_loss = losses['val']

                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"{'─'*60}\n")

                # Log to W&B (if available)
                if self.use_wandb:
                    wandb.log({
                        'eval/train_loss': train_loss,
                        'eval/val_loss': val_loss,
                    }, step=iter)

                # Track validation losses
                self.val_losses.append(val_loss)
                self.val_iterations.append(iter)

                # Save best model
                prev_best = self.best_val_loss
                self.save_best_model(val_loss)

                # Early stopping based on validation improvement
                if self.config.early_stop_patience > 0:
                    improved = val_loss < (prev_best - self.config.early_stop_min_delta)
                    if improved:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if no_improve_count >= self.config.early_stop_patience:
                        early_stopped = True
                        print(
                            f"Early stopping at step {iter}: "
                            f"no val improvement >= {self.config.early_stop_min_delta:.2f} "
                            f"for {self.config.early_stop_patience} evals."
                        )
                        break

        # Final evaluation
        print_section("Training Complete!")
        losses = self.evaluate()
        print(f"Final | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        if early_stopped:
            print("Training stopped early due to plateaued validation loss.")

        # Finish W&B run
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, path: str):
        """Save model state."""
        # Ensure models directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

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

    # Save final model
    trainer.save_checkpoint(config.model_path)
    print(f"Model saved to: {config.model_path}")
    print(f"Tokenizer saved to: {config.meta_path}")
    if trainer.use_wandb and wandb.run is not None:
        print(f"View W&B dashboard: {wandb.run.url}")


if __name__ == "__main__":
    main()
