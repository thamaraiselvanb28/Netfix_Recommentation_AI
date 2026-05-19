"""
DataLoader
──────────
Loads the TMDB Kaggle dataset (movies_metadata.csv + credits.csv + keywords.csv).

Priority order:
  1. data/raw/  — CSVs you placed manually
  2. kagglehub  — auto-download on first run (needs ~/.kaggle/kaggle.json)

After loading, the three DataFrames are merged and cleaned into one flat
movies_df that the RecommendationEngine consumes.
"""

from __future__ import annotations

import ast
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR   = Path("data/raw")
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "movies_clean.parquet"

KAGGLE_DATASET = "rounakbanik/the-movies-dataset"


class DataLoader:
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Return cleaned movies DataFrame (cached after first build)."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            return pd.read_parquet(CACHE_FILE)

        meta, credits, keywords = self._load_raw_csvs()
        df = self._merge_and_clean(meta, credits, keywords)
        df.to_parquet(CACHE_FILE, index=False)
        return df

    # ------------------------------------------------------------------ #
    def _load_raw_csvs(self):
        meta_path     = RAW_DIR / "movies_metadata.csv"
        credits_path  = RAW_DIR / "credits.csv"
        keywords_path = RAW_DIR / "keywords.csv"

        if not meta_path.exists():
            self._kaggle_download()

        meta     = pd.read_csv(meta_path,     low_memory=False)
        credits  = pd.read_csv(credits_path,  low_memory=False)
        keywords = pd.read_csv(keywords_path, low_memory=False)
        return meta, credits, keywords

    # ------------------------------------------------------------------ #
    @staticmethod
    def _kaggle_download():
        """Download via kagglehub (requires ~/.kaggle/kaggle.json)."""
        try:
            import kagglehub  # type: ignore
            path = kagglehub.dataset_download(KAGGLE_DATASET)
            src  = Path(path)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            for fname in ("movies_metadata.csv", "credits.csv", "keywords.csv"):
                src_file = src / fname
                if src_file.exists():
                    import shutil
                    shutil.copy(src_file, RAW_DIR / fname)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download dataset: {exc}\n"
                "Place movies_metadata.csv, credits.csv, keywords.csv in data/raw/"
            ) from exc

    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_parse(val):
        """Parse a stringified list/dict safely."""
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []

    @staticmethod
    def _extract_names(val, key="name", limit=None) -> list[str]:
        parsed = DataLoader._safe_parse(val)
        if not isinstance(parsed, list):
            return []
        names = [item[key] for item in parsed if isinstance(item, dict) and key in item]
        return names[:limit] if limit else names

    # ------------------------------------------------------------------ #
    def _merge_and_clean(
        self,
        meta: pd.DataFrame,
        credits: pd.DataFrame,
        keywords: pd.DataFrame,
    ) -> pd.DataFrame:
        # ── Basic meta cleanup ────────────────────────────────────────────
        meta = meta[meta["status"] == "Released"].copy()
        meta = meta[~meta["id"].astype(str).str.contains(r"\D", na=False)]  # drop bad IDs
        meta["id"] = meta["id"].astype(int)
        meta.drop_duplicates("id", keep="first", inplace=True)

        # ── Numeric cols ──────────────────────────────────────────────────
        for col in ("budget", "revenue", "popularity", "vote_average", "vote_count", "runtime"):
            meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0)

        # ── Year ──────────────────────────────────────────────────────────
        meta["release_date"] = pd.to_datetime(meta["release_date"], errors="coerce")
        meta["year"] = meta["release_date"].dt.year.fillna(0).astype(int)

        # ── Genres list ───────────────────────────────────────────────────
        meta["genres_list"] = meta["genres"].apply(
            lambda v: self._extract_names(v)
        )
        meta["genres_str"] = meta["genres_list"].apply(lambda g: " ".join(g))

        # ── Spoken languages ──────────────────────────────────────────────
        meta["languages_list"] = meta["spoken_languages"].apply(
            lambda v: self._extract_names(v)
        )

        # ── Production countries ──────────────────────────────────────────
        meta["countries_list"] = meta["production_countries"].apply(
            lambda v: self._extract_names(v)
        )

        # ── Merge credits ─────────────────────────────────────────────────
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce").dropna().astype(int)
        meta = meta.merge(credits, on="id", how="left")

        meta["cast_list"] = meta["cast"].apply(
            lambda v: self._extract_names(v, limit=5)
        )
        meta["cast_str"] = meta["cast_list"].apply(
            lambda c: " ".join(n.replace(" ", "") for n in c)
        )

        meta["director"] = meta["crew"].apply(self._extract_director)

        # ── Merge keywords ────────────────────────────────────────────────
        keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce").dropna().astype(int)
        meta = meta.merge(keywords, on="id", how="left")
        meta["keywords_list"] = meta["keywords"].apply(
            lambda v: self._extract_names(v, limit=10)
        )
        meta["keywords_str"] = meta["keywords_list"].apply(
            lambda k: " ".join(kw.replace(" ", "") for kw in k)
        )

        # ── Overview ──────────────────────────────────────────────────────
        meta["overview"] = meta["overview"].fillna("")

        # ── Weighted rating (IMDB formula) ────────────────────────────────
        C = meta["vote_average"].median()
        m = meta["vote_count"].quantile(0.70)
        q = meta[meta["vote_count"] >= m].copy()
        q["score"] = (
            q["vote_count"] / (q["vote_count"] + m) * q["vote_average"]
            + m / (q["vote_count"] + m) * C
        )
        meta = meta.merge(q[["id", "score"]], on="id", how="left")
        meta["score"] = meta["score"].fillna(0)

        # ── Drop junk ─────────────────────────────────────────────────────
        meta = meta[meta["title"].notna() & (meta["title"].str.strip() != "")]
        meta = meta[meta["vote_count"] >= 5].reset_index(drop=True)

        # ── Final column selection ────────────────────────────────────────
        keep = [
            "id", "title", "original_language", "overview",
            "genres_list", "genres_str", "cast_list", "cast_str",
            "director", "keywords_list", "keywords_str",
            "vote_average", "vote_count", "score",
            "budget", "revenue", "popularity", "runtime",
            "year", "release_date", "poster_path", "imdb_id",
            "tagline", "languages_list", "countries_list",
        ]
        existing = [c for c in keep if c in meta.columns]
        return meta[existing].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_director(crew_val) -> str:
        parsed = DataLoader._safe_parse(crew_val)
        if not isinstance(parsed, list):
            return ""
        for member in parsed:
            if isinstance(member, dict) and member.get("job") == "Director":
                return member.get("name", "").replace(" ", "")
        return ""
