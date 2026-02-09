"""Configuration and hyperparameters for KanyeGPT."""

from dataclasses import dataclass


def _get_device() -> str:
    """Get the best available device for training."""
    import torch
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@dataclass(frozen=True)
class PreTrainConfig:
    """Hyperparameters for pre-training phase."""

    # Training
    batch_size: int = 64
    block_size: int = 64
    max_iters: int = 3000
    learning_rate: float = 6e-4
    eval_interval: int = 500
    eval_iters: int = 200

    # Model architecture
    n_embd: int = 64
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0  # Reduced from 0.2 - modern LLMs use little/no dropout

    # Optimizer settings (Muon)
    optimizer: str = "adamw"  # Options: "muon" or "adamw"
    weight_decay: float = 0.1
    momentum: float = 0.95  # For Muon's momentum SGD
    lr_1d: float = 0.05  # Learning rate multiplier for 1D/embedding params

    # System (device determined at runtime)
    device: str = 'cpu'

    # Paths
    data_path: str = 'kanye_raw.txt'
    model_path: str = 'models/kanye_base.pth'
    meta_path: str = 'models/meta.pkl'
    losses_dir: str = 'models/losses'
    best_model_dir: str = 'models/best'

    def __post_init__(self):
        """Set device after initialization."""
        import torch
        object.__setattr__(self, 'device', 'mps' if torch.backends.mps.is_available() else 'cpu')


@dataclass(frozen=True)
class FineTuneConfig:
    """Hyperparameters for fine-tuning phase."""

    # Training (typically lower LR for fine-tuning)
    batch_size: int = 32
    max_iters: int = 1000
    learning_rate: float = 1e-4

    # TODO: You can experiment with these values to see how they affect
    # the model's ability to learn from synthetic data vs retaining original style
    # Try: learning_rate = 5e-5 for more conservative adaptation
    #       or learning_rate = 2e-4 for faster style transfer
