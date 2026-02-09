#!/usr/bin/env python3
"""Simple training monitor - shows live loss curves in terminal."""

import json
import time
import os
from pathlib import Path

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def show_progress():
    loss_file = Path("models/losses/losses.json")
    
    if not loss_file.exists():
        print("Waiting for training to start...")
        return
    
    with open(loss_file) as f:
        data = json.load(f)
    
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    val_iters = data.get('val_iterations', [])
    current_iter = data.get('current_iteration', 0)
    best_val = data.get('best_val_loss', float('inf'))
    
    clear_screen()
    print("=" * 60)
    print("           KanyeGPT Training Monitor")
    print("=" * 60)
    print(f"Step: {current_iter}/{3000} ({current_iter/3000*100:.1f}%)")
    print()
    
    if train_losses:
        print(f"Latest Train Loss: {train_losses[-1]:.4f}")
        if len(train_losses) > 1:
            trend = train_losses[-1] - train_losses[-100] if len(train_losses) >= 100 else train_losses[-1] - train_losses[0]
            print(f"Trend (last 100): {'↓' if trend < 0 else '↑'} {abs(trend):.4f}")
    
    if val_losses:
        print(f"Latest Val Loss:   {val_losses[-1]:.4f}")
        print(f"Best Val Loss:     {best_val:.4f}")
    
    print()
    
    # Simple ASCII chart of last 100 steps
    if len(train_losses) >= 10:
        recent = train_losses[-100:]
        min_loss, max_loss = min(recent), max(recent)
        height = 15
        
        print("Loss History (last 100 steps):")
        for i in range(height):
            line = "  "
            threshold = max_loss - (max_loss - min_loss) * (i / height)
            for val in recent:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            print(line)
        print(f"  {min_loss:.3f}" + " " * (96 - len(f"{min_loss:.3f}")) + f"{max_loss:.3f}")
    
    print()
    print("Press Ctrl+C to exit. Updates every 2 seconds.")

if __name__ == "__main__":
    try:
        while True:
            show_progress()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
