"""
Sidebar component
─────────────────
Renders the global filter sidebar and returns a cfg dict consumed by app.py.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_sidebar(movies_df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">🎬 CineAI</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### 🎛 Global Filters")

        # ── Genre ─────────────────────────────────────────────────────────
        all_genres = sorted(
            {g for gl in movies_df["genres_list"].dropna() for g in gl}
        )
        genres = st.multiselect(
            "Genres",
            options=all_genres,
            default=[],
            placeholder="All genres",
        )

        # ── Language ──────────────────────────────────────────────────────
        all_langs = sorted(movies_df["original_language"].dropna().unique().tolist())
        languages = st.multiselect(
            "Languages",
            options=all_langs,
            default=[],
            placeholder="All languages",
        )

        # ── Year range ────────────────────────────────────────────────────
        min_year = int(movies_df["year"].min()) if "year" in movies_df.columns else 1900
        max_year = int(movies_df["year"].max()) if "year" in movies_df.columns else 2024
        min_year = max(min_year, 1900)
        year_range = st.slider(
            "Release Year",
            min_value=min_year,
            max_value=max_year,
            value=(1990, max_year),
        )

        # ── Min rating ────────────────────────────────────────────────────
        min_rating = st.slider(
            "Minimum Rating",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
        )

        st.markdown("---")

        # ── Watchlist quick-count ─────────────────────────────────────────
        wl_count = len(st.session_state.get("watchlist", []))
        st.markdown(
            f'<div class="sidebar-wl">❤️ Watchlist: <b>{wl_count}</b> movies</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<div class="sidebar-footer">'
            'Data: <a href="https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset" '
            'target="_blank">TMDB Kaggle</a><br>'
            'Posters: <a href="https://www.themoviedb.org" target="_blank">TMDB API</a>'
            '</div>',
            unsafe_allow_html=True,
        )

    return dict(
        genres=genres or None,
        languages=languages or None,
        year_range=year_range,
        min_rating=min_rating,
    )
