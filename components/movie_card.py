"""
Movie card components
─────────────────────
render_movie_grid  — responsive grid of movie cards
render_hero_card   — large featured card for the selected movie
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def _safe(val, default="N/A"):
    if val is None or (isinstance(val, float) and val != val):
        return default
    s = str(val).strip()
    return s if s and s != "nan" else default


def _stars(rating: float) -> str:
    full  = int(rating / 2)
    half  = 1 if (rating / 2 - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def render_hero_card(row, tmdb):
    """Large featured card shown above content-based results."""
    poster = tmdb.poster_url(row)
    title   = _safe(row.get("title", ""))
    year    = _safe(row.get("year", ""), "")
    rating  = float(row.get("vote_average", 0) or 0)
    votes   = int(row.get("vote_count", 0) or 0)
    genres  = row.get("genres_list", []) or []
    overview = _safe(row.get("overview", ""), "No overview available.")
    director = _safe(row.get("director", "").replace("None", "").strip() or "Unknown")
    runtime  = row.get("runtime", 0) or 0

    genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in genres[:4])
    runtime_str = f"{int(runtime)} min" if runtime else ""

    col_img, col_info = st.columns([1, 3])
    with col_img:
        st.image(poster, use_container_width=True)
    with col_info:
        st.markdown(f"""
<div class="hero-card-info">
  <h2 class="hc-title">{title} <span class="hc-year">({year})</span></h2>
  <div class="hc-meta">
    <span class="hc-rating">⭐ {rating:.1f}</span>
    <span class="hc-votes">{votes:,} votes</span>
    {f'<span class="hc-runtime">🕐 {runtime_str}</span>' if runtime_str else ''}
    <span class="hc-director">🎬 {director}</span>
  </div>
  <div class="hc-genres">{genres_html}</div>
  <p class="hc-overview">{overview}</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────── #
def render_movie_grid(
    df: pd.DataFrame,
    tmdb,
    key_prefix: str = "card",
    show_score: bool = False,
    score_label: str = "Score",
    watchlist_mode: bool = False,
    cols: int = 5,
):
    """Render a responsive grid of movie cards."""
    if df.empty:
        st.markdown('<div class="empty-state">No movies to display.</div>',
                    unsafe_allow_html=True)
        return

    # Deduplicate so the same movie never produces two identical keys
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    rows = [df.iloc[i : i + cols] for i in range(0, len(df), cols)]

    global_idx = 0
    for row_df in rows:
        grid_cols = st.columns(cols)
        for col, (_, movie) in zip(grid_cols, row_df.iterrows()):
            with col:
                _render_card(
                    movie, tmdb,
                    key_prefix=key_prefix,
                    card_idx=global_idx,
                    show_score=show_score,
                    score_label=score_label,
                    watchlist_mode=watchlist_mode,
                )
            global_idx += 1


def _render_card(movie, tmdb, key_prefix, card_idx, show_score, score_label, watchlist_mode):
    poster  = tmdb.poster_url(movie)
    title   = _safe(movie.get("title", "Unknown"))
    year    = _safe(movie.get("year", ""), "")
    rating  = float(movie.get("vote_average", 0) or 0)
    genres  = movie.get("genres_list", []) or []
    genre_str = ", ".join(genres[:2]) if genres else ""
    score_val = float(movie.get("similarity", movie.get("score", 0)) or 0)

    in_watchlist = title in st.session_state.get("watchlist", [])

    score_html = ""
    if show_score:
        score_html = f'<div class="card-score">{score_label}: {score_val:.2f}</div>'

    genres_html = "".join(
        f'<span class="genre-tag-sm">{g}</span>' for g in genres[:2]
    )

    st.markdown(f"""
<div class="movie-card">
  <div class="card-poster-wrap">
    <img src="{poster}" class="card-poster" loading="lazy"
         onerror="this.src='https://via.placeholder.com/300x450?text=No+Poster'"/>
    <div class="card-overlay">
      <div class="card-rating">⭐ {rating:.1f}</div>
    </div>
  </div>
  <div class="card-body">
    <div class="card-title" title="{title}">{title}</div>
    <div class="card-year">{year}</div>
    <div class="card-genres">{genres_html}</div>
    {score_html}
  </div>
</div>
""", unsafe_allow_html=True)

    if watchlist_mode:
        wl_label = "✓ Saved" if in_watchlist else "+ Watchlist"
        btn_key = f"{key_prefix}_wl_{card_idx}"
        if st.button(wl_label, key=btn_key, use_container_width=True):
            wl = st.session_state.get("watchlist", [])
            if title in wl:
                wl.remove(title)
            else:
                wl.append(title)
            st.session_state.watchlist = wl
            st.rerun()
