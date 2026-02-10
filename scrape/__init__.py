"""
Yeezus Data Pipeline - Scraper Package

A modular data collection system for training KanyeGPT.
Combines lyrics, interviews, and synthetic data into training datasets.
"""

from scrape.clique import scrape_clique_lyrics
from scrape.interviews import scrape_interviews
from scrape.merge import merge_datasets

__all__ = [
    "scrape_clique_lyrics",
    "scrape_interviews",
    "merge_datasets",
]
