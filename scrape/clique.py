"""
The "Clique" Scraper - Lyrics from collaborators

Downloads lyrics from Kanye's closest collaborators to teach the model
"Hip Hop Grammar" - proper flow, rhyme schemes, and vocabulary.
"""

import lyricsgenius
import time
import os
from typing import Optional


def scrape_clique_lyrics(
    token: Optional[str] = None,
    artists: Optional[list[str]] = None,
    max_songs: int = 50,
    output_file: str = "clique_lyrics.txt",
    sleep_time: float = 5.0,
    timeout: int = 30,
) -> str:
    """
    Scrape lyrics from Genius for specified artists.

    Args:
        token: Genius API access token (reads from env if None)
        artists: List of artist names to scrape
        max_songs: Maximum songs per artist
        output_file: Output filename
        sleep_time: Seconds to sleep between artists (rate limiting)
        timeout: API request timeout in seconds

    Returns:
        Path to the output file

    Raises:
        ValueError: If token not found or no artists specified
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Get token
    if token is None:
        token = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
    if not token:
        raise ValueError("GENIUS_CLIENT_ACCESS_TOKEN not found in .env file!")

    # Default artists - G.O.O.D. Music roster + influences
    if artists is None:
        artists = ["Kanye West", "Jay-Z", "Pusha T", "Kid Cudi", "Travis Scott"]
    if not artists:
        raise ValueError("No artists specified!")

    # Configure Genius API
    genius = lyricsgenius.Genius(token, timeout=timeout)
    genius.verbose = False
    genius.remove_section_headers = False  # Keep [Chorus] for structure
    genius.skip_non_songs = True
    genius.exclude_terms = ["Remix", "Live", "Mix", "Skit"]

    print("=" * 60)
    print("🎤 THE CLIQUE SCRAPER")
    print("=" * 60)
    print(f"Target: {len(artists)} artists x {max_songs} songs each")
    print(f"Output: {output_file}")
    print("=" * 60)

    total_songs = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for name in artists:
            print(f"\n🎤 Scraping {name}...")
            try:
                artist = genius.search_artist(name, max_songs=max_songs, sort="popularity")

                for song in artist.songs:
                    lyrics = song.lyrics
                    lines = lyrics.split('\n')
                    if len(lines) > 1:
                        clean_lyrics = "\n".join(lines[1:])
                        f.write(clean_lyrics + "\n\n\n\n")
                        total_songs += 1

                print(f"   ✅ {name}: {len(artist.songs)} songs saved.")
                time.sleep(sleep_time)  # Be nice to the API
            except Exception as e:
                print(f"   ❌ Failed on {name}: {e}")

    print("\n" + "=" * 60)
    print(f"💾 Saved {total_songs} songs to {output_file}")
    print("=" * 60)

    return output_file


if __name__ == "__main__":
    scrape_clique_lyrics()
