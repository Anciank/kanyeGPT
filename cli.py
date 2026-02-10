#!/usr/bin/env python3
"""Interactive CLI for KanyeGPT.

Run this script to generate lyrics using your trained model.
"""

import sys
import readline
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import PreTrainConfig
from model import GPTLanguageModel
from data import TextDataset


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @classmethod
    def header(cls, text):
        return cls.HEADER + text + cls.END

    @classmethod
    def blue(cls, text):
        return cls.BLUE + text + cls.END

    @classmethod
    def green(cls, text):
        return cls.GREEN + text + cls.END

    @classmethod
    def yellow(cls, text):
        return cls.YELLOW + text + cls.END

    @classmethod
    def bold(cls, text):
        return cls.BOLD + text + cls.END


def print_banner():
    """Print welcome banner."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║  KanyeGPT  -  Interactive Lyrics Generator             ║
╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Base Model: models/best/best_iter_1000.pth")
    print("Vocabulary: 96 characters | 636K parameters")
    print()


def print_help():
    """Print available commands."""
    print("\nAvailable Commands:")
    print("  g, generate     Generate lyrics")
    print("  t <value>, temp <value>    Set temperature (0.1-2.0, default 0.8)")
    print("  l <value>, len <value>     Set output length (50-2000, default 150)")
    print("  s <text>, seed <text>     Set seed text for guided generation")
    print("  c, clear         Clear seed text")
    print("  p, params        Show current parameters")
    print("  h, help          Show this help")
    print("  q, quit          Exit\n")


class KanyeCLI:
    """
    Interactive CLI for KanyeGPT.

    TODO: You can extend this with:
    - 'model' command to switch between base and fine-tuned models
    - 'save' command to save generated lyrics to file
    - 'batch' command to generate multiple samples at once
    - 'style' command to adjust generation style (pcreative vs focused)
    """

    def __init__(self, model_path, meta_path):
        """Initialize CLI with trained model."""
        self.model_path = model_path
        self.meta_path = meta_path
        self.config = PreTrainConfig()

        # Generation parameters
        self.temperature = 0.8
        self.length = 150
        self.seed_text = ""

        # Load model and tokenizer
        print("Loading model...")
        self._load_model()
        print("Model loaded!\n")

    def _load_model(self):
        """Load trained model and tokenizer."""
        # Load tokenizer
        self.tokenizer = TextDataset.load_tokenizer(self.meta_path)

        # Initialize model
        self.model = GPTLanguageModel(self.tokenizer.vocab_size, self.config)

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.config.device)
        self.model.eval()

    def generate(self):
        """Generate lyrics using current settings."""
        print("\nGenerating...\n")

        # Prepare context
        if self.seed_text:
            context = torch.tensor([self.tokenizer.encode(self.seed_text)], dtype=torch.long)
            print("Seed:", self.seed_text, "\n")
        else:
            context = torch.tensor([[self.tokenizer.stoi['\n']]], dtype=torch.long)

        context = context.to(self.config.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                context,
                max_new_tokens=self.length,
                temperature=self.temperature
            )

        # Decode and display
        output = self.tokenizer.decode(generated[0].tolist())

        print("─" * 60)
        print(output)
        print("─" * 60)
        print()

    def set_temperature(self, value):
        """Set generation temperature."""
        try:
            temp = float(value)
            if 0.1 <= temp <= 2.0:
                self.temperature = temp
                focus_type = 'More focused' if temp < 1.0 else 'More creative'
                print("Temperature set to", self.temperature)
                print("   (" + focus_type + ")\n")
            else:
                print("Error: Temperature must be between 0.1 and 2.0\n")
        except ValueError:
            print("Error: Invalid temperature value\n")

    def set_length(self, value):
        """Set output length."""
        try:
            length = int(value)
            if 50 <= length <= 2000:
                self.length = length
                print("Length set to", self.length, "tokens\n")
            else:
                print("Error: Length must be between 50 and 2000\n")
        except ValueError:
            print("Error: Invalid length value\n")

    def set_seed(self, text):
        """Set seed text for guided generation."""
        self.seed_text = text.strip()
        if self.seed_text:
            print("Seed text set:", repr(self.seed_text), "\n")
        else:
            print("Seed cleared.\n")

    def show_params(self):
        """Show current parameters."""
        print("\nCurrent Parameters:")
        focus_type = 'focused' if self.temperature < 1.0 else 'creative'
        print("  Temperature:", self.temperature, "(" + focus_type + ")")
        print("  Length:", self.length, "tokens")
        seed_display = '"' + self.seed_text + '"' if self.seed_text else 'none'
        print("  Seed:", seed_display)
        print()

    def run(self):
        """Run interactive CLI loop."""
        print_banner()
        print_help()
        print("Type a command or\n")

        while True:
            try:
                # Get input
                cmd_input = input("kanyeGPT> ").strip()

                if not cmd_input:
                    continue

                # Parse command
                parts = cmd_input.split(maxsplit=1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                # Handle commands
                if command in ['g', 'generate']:
                    self.generate()

                elif command in ['t', 'temp']:
                    if arg:
                        self.set_temperature(arg)
                    else:
                        print("Current temperature:", self.temperature, "\n")

                elif command in ['l', 'len']:
                    if arg:
                        self.set_length(arg)
                    else:
                        print("Current length:", self.length, "\n")

                elif command in ['s', 'seed']:
                    self.set_seed(arg)

                elif command in ['c', 'clear']:
                    self.seed_text = ""
                    print("Seed text cleared\n")

                elif command in ['p', 'params']:
                    self.show_params()

                elif command in ['h', 'help', '?']:
                    print_help()

                elif command in ['q', 'quit', 'exit']:
                    print("\nGoodbye!\n")
                    break

                else:
                    # Try to generate directly as seed text
                    self.set_seed(cmd_input)
                    self.generate()

            except KeyboardInterrupt:
                print("\n\nUse \"quit\" to exit.\n")
            except Exception as e:
                print("Error:", str(e), "\n")


def main():
    """Main entry point."""
    # Check if model exists
    model_path = 'models/best/best_iter_1000.pth'
    meta_path = 'models/meta.pkl'

    if not Path(model_path).exists():
        print("Error: Model not found at", model_path)
        print("Run `python pretrain.py` first to train the model.\n")
        sys.exit(1)

    # Run CLI
    cli = KanyeCLI(model_path, meta_path)
    cli.run()


if __name__ == "__main__":
    main()
