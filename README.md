# KanyeGPT

A transformer-based language model trained on Kanye West lyrics, following a two-stage approach:
1. **Pre-training** on raw lyrics data
2. **Fine-tuning** on high-quality synthetic data

This mirrors how modern Large Language Models are built: pre-training on large, noisy corpora to learn general structure, then fine-tuning on curated data for quality and coherence.

## Project Structure

```
kanyeGPT/
├── src/
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Hyperparameter configurations
│   ├── data.py          # Data loading and tokenization
│   └── model.py         # Transformer model components
├── models/              # Saved model checkpoints (created after training)
├── pretrain.py          # Pre-training script
├── finetune.py          # Fine-tuning script (Phase 2)
├── kanye_raw.txt        # Raw lyrics data
└── kanye_synthetic.txt  # Synthetic data for fine-tuning (Phase 2)
```

## Phase 1: Pre-Training

### Setup

1. **Download the raw data:**
   ```bash
   curl -o kanye_raw.txt https://raw.githubusercontent.com/Ayodeleohh/Yesus-Lyric-Gen/master/kanye_verses.txt
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   # or
   pip install torch
   ```

3. **Run pre-training:**
   ```bash
   python pretrain.py
   ```

### What Pre-Training Does

- Loads and tokenizes the raw Kanye lyrics
- Trains a 4-layer Transformer model with ~300K parameters
- Saves the model weights to `models/kanye_base.pth`
- Saves the tokenizer to `models/meta.pkl`

### Training Output

The script will print progress every 500 steps:
```
Step    0 | Train Loss: 4.1234 | Val Loss: 4.1456
Step  500 | Train Loss: 2.3456 | Val Loss: 2.3890
...
```

After training, it generates a sample verse to demonstrate the model's capabilities.

## Architecture Details

`★ Insight ─────────────────────────────────────`
- **Causal Self-Attention**: Each token can only attend to previous tokens, enabling autoregressive generation
- **Position Embeddings**: Since attention has no inherent notion of order, we add learnable position embeddings
- **Layer Normalization**: Applied before each sub-layer (pre-norm) for more stable training
- **Feed-Forward Network**: Expands embeddings 4x, applies ReLU, then contracts back - this is where most of the model's "knowledge" is stored
`─────────────────────────────────────────────────`

### Model Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `n_embd` | 128 | Embedding dimension |
| `n_layer` | 4 | Number of Transformer blocks |
| `n_head` | 4 | Number of attention heads |
| `block_size` | 128 | Context window (tokens) |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `max_iters` | 3000 | Pre-training iterations |

## Phase 2: Fine-Tuning (Coming Next)

While pre-training runs, you can prepare synthetic data:

1. Use an LLM (ChatGPT, Claude, etc.) to generate 50-100 Kanye-style verses
2. Request specific themes: Chicago, fashion, self-reflection
3. Save outputs to `kanye_synthetic.txt`

The fine-tuning script will:
- Load the pre-trained base model
- Train on your synthetic data at a lower learning rate
- Create a refined model that captures Kanye's style more coherently

## Extending the Project

The `Trainer` class in `pretrain.py` includes TODOs for enhancements:

- **Learning rate scheduling**: Cosine decay or warmup for better convergence
- **Gradient clipping**: Prevent exploding gradients during training
- **Early stopping**: Halt training if validation loss plateaus
- **TensorBoard logging**: Visualize training metrics in real-time

## Requirements

- Python 3.12+
- PyTorch 2.10+
- (Optional) MPS support for Apple Silicon acceleration
