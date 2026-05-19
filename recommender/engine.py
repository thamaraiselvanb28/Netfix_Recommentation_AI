"""
RecommendationEngine
────────────────────
Algorithms
  1. content_based()   — TF-IDF on soup + cosine / weighted-hybrid similarity
  2. discover()        — curated lists (top-rated, popular, hidden gems …)
  3. filter_movies()   — free-text + faceted search
  4. collab_score()    — lightweight item-item CF via vote signals
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler


class RecommendationEngine:
    def __init__(self, movies_df: pd.DataFrame):
        self.df = movies_df.reset_index(drop=True)
        self._title_to_idx: dict[str, int] = {}
        self._tfidf_matrix = None
        self._tfidf_vec: TfidfVectorizer | None = None
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────── #
    def fit(self):
        df = self.df

        # Build soup for TF-IDF
        df["soup"] = (
            df.get("overview",      pd.Series("", index=df.index)).fillna("") + " "
            + df.get("genres_str",  pd.Series("", index=df.index)).fillna("") + " "
            + df.get("cast_str",    pd.Series("", index=df.index)).fillna("") + " "
            + df.get("director",    pd.Series("", index=df.index)).fillna("") + " "
            + df.get("keywords_str",pd.Series("", index=df.index)).fillna("")
        )

        self._tfidf_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=25_000,
            stop_words="english",
        )
        self._tfidf_matrix = self._tfidf_vec.fit_transform(df["soup"])
        self._title_to_idx = pd.Series(df.index, index=df["title"]).to_dict()
        self._fitted = True

    # ──────────────────────────────────────────────────────────────────── #
    def content_based(
        self,
        title: str,
        n: int = 12,
        metric: Literal["Cosine", "Weighted Hybrid"] = "Cosine",
    ) -> pd.DataFrame:
        if title not in self._title_to_idx:
            return pd.DataFrame()

        idx = self._title_to_idx[title]
        vec = self._tfidf_matrix[idx]
        cos_scores = linear_kernel(vec, self._tfidf_matrix).flatten()

        if metric == "Weighted Hybrid":
            # Blend cosine with normalised popularity score
            scaler = MinMaxScaler()
            pop = scaler.fit_transform(
                self.df["score"].values.reshape(-1, 1)
            ).flatten()
            cos_norm = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min() + 1e-9)
            combined = 0.7 * cos_norm + 0.3 * pop
        else:
            combined = cos_scores

        sim_series = pd.Series(combined).drop(index=idx)
        top_idx = sim_series.nlargest(n).index
        result = self.df.iloc[top_idx].copy()
        result["similarity"] = combined[top_idx]
        return result.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────── #
    def discover(
        self,
        mode: str = "Top Rated",
        language: str = "All",
        n: int = 20,
    ) -> pd.DataFrame:
        df = self.df.copy()

        if language != "All":
            df = df[df["original_language"] == language]

        if mode == "Top Rated":
            df = df[df["vote_count"] >= df["vote_count"].quantile(0.6)]
            df = df.nlargest(n, "score")

        elif mode == "Most Popular":
            df = df.nlargest(n, "popularity")

        elif mode == "Latest Releases":
            df = df[df["year"] > 0].nlargest(n, "year")

        elif mode == "Hidden Gems":
            # High rating, low vote count (under the radar)
            df = df[
                (df["vote_average"] >= 7.0)
                & (df["vote_count"] < df["vote_count"].quantile(0.4))
            ].nlargest(n, "vote_average")

        elif mode == "Oscar Winners":
            kw_col = "keywords_str" if "keywords_str" in df.columns else "overview"
            mask = df[kw_col].str.contains(
                r"oscar|academy award|best picture", case=False, na=False
            )
            df = df[mask].nlargest(n, "score")
            if df.empty:
                df = self.df.nlargest(n, "score")

        else:
            df = df.nlargest(n, "score")

        return df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────── #
    def filter_movies(
        self,
        query: str = "",
        genres: list[str] | None = None,
        languages: list[str] | None = None,
        year_range: tuple[int, int] = (1900, 2024),
        min_rating: float = 0.0,
        sort_by: str = "Rating ↓",
    ) -> pd.DataFrame:
        df = self.df.copy()

        # Free-text search
        if query and query.strip():
            q = query.lower().strip()
            text_cols = ["title", "overview", "director", "cast_str", "keywords_str", "genres_str"]
            mask = pd.Series(False, index=df.index)
            for col in text_cols:
                if col in df.columns:
                    mask |= df[col].fillna("").str.lower().str.contains(
                        re.escape(q), na=False
                    )
            df = df[mask]

        # Genre filter
        if genres:
            genre_mask = df["genres_list"].apply(
                lambda gl: bool(set(gl) & set(genres)) if isinstance(gl, list) else False
            )
            df = df[genre_mask]

        # Language filter
        if languages:
            df = df[df["original_language"].isin(languages)]

        # Year range
        if "year" in df.columns:
            df = df[df["year"].between(year_range[0], year_range[1])]

        # Min rating
        df = df[df["vote_average"] >= min_rating]

        # Sort
        sort_map = {
            "Rating ↓":  ("vote_average", False),
            "Votes ↓":   ("vote_count",   False),
            "Year ↓":    ("year",         False),
            "Title A-Z": ("title",        True),
            "Revenue ↓": ("revenue",      False),
        }
        col, asc = sort_map.get(sort_by, ("score", False))
        if col in df.columns:
            df = df.sort_values(col, ascending=asc)

        return df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────── #
    def top_genres(self, n: int = 10) -> pd.DataFrame:
        from collections import Counter
        counts = Counter(
            g for genres in self.df["genres_list"].dropna() for g in genres
        )
        return pd.DataFrame(counts.most_common(n), columns=["Genre", "Count"])

    def top_languages(self, n: int = 10) -> pd.DataFrame:
        lang_counts = (
            self.df["original_language"]
            .value_counts()
            .head(n)
            .reset_index()
        )
        lang_counts.columns = ["Language", "Count"]
        return lang_counts

    def year_distribution(self) -> pd.DataFrame:
        df = self.df[self.df["year"] > 1900].copy()
        return df.groupby("year").size().reset_index(name="Count")
