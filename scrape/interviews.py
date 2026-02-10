"""
The "Rant" Scraper - Interview transcripts

Downloads transcripts from YouTube interviews to provide conversational
connective tissue (verbs, conjunctions, punctuation) that lyrics often miss.
"""

from typing import Optional

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
    _yt_api = YouTubeTranscriptApi()
except ImportError:
    YOUTUBE_AVAILABLE = False
    _yt_api = None


# Famous Kanye Interview IDs
DEFAULT_VIDEO_IDS = [
    "JbFhQy7B0w8",  # Joe Rogan Experience
    "f6oYMtCM67Q",  # BigboyTV
    "I4dGz9W3Fm4",  # Zane Lowe BBC
    "Lq1Q_s3G8aI",  # Charlemagne tha God
]


def scrape_interviews(
    video_ids: Optional[list[str]] = None,
    output_file: str = "kanye_interviews.txt",
) -> str:
    """
    Scrape transcripts from YouTube videos.

    Args:
        video_ids: List of YouTube video IDs
        output_file: Output filename

    Returns:
        Path to the output file

    Raises:
        ImportError: If youtube-transcript-api is not installed
    """
    if not YOUTUBE_AVAILABLE:
        raise ImportError(
            "youtube-transcript-api not installed. "
            "Install with: uv pip install youtube-transcript-api"
        )

    if video_ids is None:
        video_ids = DEFAULT_VIDEO_IDS

    print("=" * 60)
    print("📺 THE RANT SCRAPER")
    print("=" * 60)
    print(f"Target: {len(video_ids)} interview transcripts")
    print(f"Output: {output_file}")
    print("=" * 60)

    full_transcript = ""
    successful = 0

    for vid in video_ids:
        try:
            print(f"   📥 Fetching {vid}...")
            transcript_list = _yt_api.fetch(vid)
            text = " ".join([t.text for t in transcript_list])
            full_transcript += text + "\n\n"
            successful += 1
        except Exception as e:
            print(f"   ⚠️  Could not fetch {vid}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_transcript)

    print("\n" + "=" * 60)
    print(f"💾 Saved {successful}/{len(video_ids)} transcripts to {output_file}")
    print("=" * 60)

    return output_file


if __name__ == "__main__":
    scrape_interviews()
