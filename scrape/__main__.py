"""
Yeezus Data Pipeline - CLI Entry Point

Run the full data pipeline: scrape lyrics, interviews, and merge.

Usage:
    python -m scrape                    # Run full pipeline
    python -m scrape --lyrics-only      # Skip interviews
    python -m scrape --interviews-only  # Skip lyrics
    python -m scrape --merge-only       # Just merge existing files
"""

import argparse
import os
import sys
from dotenv import load_dotenv

from scrape.clique import scrape_clique_lyrics
from scrape.interviews import scrape_interviews, YOUTUBE_AVAILABLE
from scrape.merge import merge_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Yeezus Data Pipeline - Scrape and merge training data"
    )
    parser.add_argument(
        "--lyrics-only", "-l",
        action="store_true",
        help="Only scrape lyrics, skip interviews"
    )
    parser.add_argument(
        "--interviews-only", "-i",
        action="store_true",
        help="Only scrape interviews, skip lyrics"
    )
    parser.add_argument(
        "--merge-only", "-m",
        action="store_true",
        help="Only merge existing files, skip scraping"
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset_large.txt",
        help="Output filename for merged dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    print("\n" + "=" * 60)
    print("🎤 YEEZUS DATA PIPELINE")
    print("=" * 60 + "\n")

    # Check for Genius token
    if not args.merge_only and not args.interviews_only:
        token = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
        if not token:
            print("❌ GENIUS_CLIENT_ACCESS_TOKEN not found in .env")
            print("   Please add it to your .env file to scrape lyrics.")
            sys.exit(1)

    # Run pipeline
    scrape_lyrics = not args.interviews_only and not args.merge_only
    scrape_interviews = not args.lyrics_only and not args.merge_only
    do_merge = True

    try:
        if scrape_lyrics:
            scrape_clique_lyrics(output_file="clique_lyrics.txt")

        if scrape_interviews:
            if not YOUTUBE_AVAILABLE:
                print("\n⚠️  youtube-transcript-api not installed.")
                print("   Skipping interviews. Install with:")
                print("   uv pip install youtube-transcript-api\n")
            else:
                scrape_interviews(output_file="kanye_interviews.txt")

        if do_merge:
            merge_datasets(output_file=args.output)

        print("\n✅ Pipeline complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
