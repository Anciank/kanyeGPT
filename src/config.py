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
    block_size: int = 128
    max_iters: int = 3000
    learning_rate: float = 3e-4
    eval_interval: int = 500
    eval_iters: int = 200

    # Model architecture
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2

    # System (device determined at runtime)
    device: str = 'cpu'

    # Paths
    data_path: str = 'kanye_raw.txt'
    model_path: str = 'models/kanye_base.pth'
    meta_path: str = 'models/meta.pkl'

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
