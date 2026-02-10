"""
The "Master Merge" - Combine all data sources

Combines lyrics, interviews, and synthetic data into one training file.
"""

import os
from typing import Optional


def merge_datasets(
    input_files: Optional[list[str]] = None,
    output_file: str = "dataset_large.txt",
) -> str:
    """
    Merge multiple data files into a single training dataset.

    Args:
        input_files: List of files to merge (default: clique_lyrics.txt, interviews, synthetic)
        output_file: Output filename

    Returns:
        Path to the output file
    """
    if input_files is None:
        input_files = [
            'clique_lyrics.txt',      # Hip hop grammar from collaborators
            'kanye_interviews.txt',   # Conversational flow & punctuation
            'kanye_synthetic.txt'     # AI-generated verses (if exists)
        ]

    print("=" * 60)
    print("🔗 THE MASTER MERGE")
    print("=" * 60)

    total_chars = 0
    files_merged = []

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in input_files:
            if os.path.exists(fname):
                print(f"   📄 Merging {fname}...")
                with open(fname, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n\n")  # Buffer between files
                    files_merged.append(fname)
                    total_chars += len(content)
            else:
                print(f"   ⚠️  Warning: {fname} not found. Skipping.")

    size_mb = total_chars / (1024 * 1024)

    print("\n" + "=" * 60)
    print("✨ MERGE COMPLETE!")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Files merged: {', '.join(files_merged)}")
    print(f"Total size: {size_mb:.2f} MB ({total_chars:,} characters)")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: If interviews were included, DELETE")
    print("   the old 'meta.pkl' file before training!")
    print("   The vocabulary needs to be regenerated.")
    print("=" * 60)

    return output_file


if __name__ == "__main__":
    merge_datasets()
