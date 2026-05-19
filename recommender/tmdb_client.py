"""
TMDBClient  — No-API / Offline Mode
─────────────────────────────────────
Posters are served directly from TMDB's public image CDN using the
`poster_path` column that already exists in the Kaggle dataset CSV.
NO API KEY is required for this to work.

If a poster_path is missing, a clean SVG placeholder is used (no
external service needed).
"""

from __future__ import annotations

import os
from pathlib import Path

# Public TMDB image CDN — works without any API key
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# Inline SVG placeholder — no external service, always works
_SVG_PLACEHOLDER = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "width='300' height='450' viewBox='0 0 300 450'%3E"
    "%3Crect width='300' height='450' fill='%23161b22'/%3E"
    "%3Ctext x='150' y='210' font-family='Arial' font-size='48' "
    "fill='%2330363d' text-anchor='middle'%3E%F0%9F%8E%AC%3C/text%3E"
    "%3Ctext x='150' y='260' font-family='Arial' font-size='13' "
    "fill='%238b949e' text-anchor='middle'%3ENo Poster%3C/text%3E"
    "%3C/svg%3E"
)


class TMDBClient:
    """
    Offline-friendly TMDB client.

    poster_url() uses the poster_path column from the Kaggle dataset,
    which maps directly to TMDB's public image CDN — no API key needed.
    """

    def __init__(self):
        self.api_key = self._read_api_key()

    # ------------------------------------------------------------------ #
    def poster_url(self, row) -> str:
        """
        Return a poster image URL for a movie row.

        Priority:
          1. poster_path from dataset  →  TMDB public CDN (no key needed)
          2. Inline SVG placeholder    →  always works, no external call
        """
        path = (
            row.get("poster_path", "")
            if hasattr(row, "get")
            else getattr(row, "poster_path", "")
        )
        if path and str(path).strip() not in ("", "nan", "None", "none"):
            return f"{POSTER_BASE}{str(path).strip()}"
        return _SVG_PLACEHOLDER

    # ------------------------------------------------------------------ #
    def fetch_trailer_key(self, movie_id: int) -> str | None:
        """Trailer lookup — only works if API key is set, skips silently otherwise."""
        if not self.api_key:
            return None
        try:
            import requests
            import json
            cache = Path("data/cache/tmdb")
            cache.mkdir(parents=True, exist_ok=True)
            f = cache / f"trailer_{movie_id}.json"
            if f.exists():
                data = json.loads(f.read_text())
            else:
                r = requests.get(
                    f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                    params={"api_key": self.api_key},
                    timeout=5,
                )
                data = r.json() if r.ok else {}
                f.write_text(json.dumps(data))
            for v in data.get("results", []):
                if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                    return v["key"]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_api_key() -> str:
        """Read TMDB_API_KEY from .env or environment (fully optional)."""
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("TMDB_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and val != "your_tmdb_api_key_here":
                        return val
        return os.getenv("TMDB_API_KEY", "")
