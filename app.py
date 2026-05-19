"""
╔══════════════════════════════════════════════════════════════════╗
║   CineAI v2 — TMDB-Powered Movie Recommendation Engine          ║
║   Dataset : TMDB Kaggle (movies_metadata + credits + keywords)  ║
║   Posters : TMDB API  (real movie posters)                      ║
║   Stack   : Streamlit · Scikit-learn · Pandas · TMDB API        ║
╚══════════════════════════════════════════════════════════════════╝

First-time setup
────────────────
  1. pip install -r requirements.txt
  2. Copy .env.example → .env  and fill in your TMDB API key
     Get a free key at: https://www.themoviedb.org/settings/api
  3. streamlit run app.py
     (dataset downloads automatically via kagglehub on first run,
      OR drop your CSVs into data/raw/ and it will use those)
"""

import os
import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CineAI · Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
with open("assets/styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Lazy imports ───────────────────────────────────────────────────────────────
from recommender.engine      import RecommendationEngine
from recommender.data_loader import DataLoader
from recommender.tmdb_client import TMDBClient
from components.movie_card   import render_movie_grid, render_hero_card
from components.sidebar      import render_sidebar
from components.metrics      import render_metrics_panel

# ── Session state ──────────────────────────────────────────────────────────────
defaults = dict(
    engine_ready=False, recommendations=[],
    selected_movie=None, user_ratings={},
    search_query="", watchlist=[],
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load everything (cached across reruns) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def boot():
    loader = DataLoader()
    movies_df = loader.load()
    engine    = RecommendationEngine(movies_df)
    engine.fit()
    tmdb      = TMDBClient()          # reads TMDB_API_KEY from .env / env var
    return engine, movies_df, tmdb

with st.spinner("🎬  Loading CineAI Engine…"):
    engine, movies_df, tmdb = boot()
    st.session_state.engine_ready = True

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrapper">
  <div class="hero-bg-grid"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">✦ AI-POWERED CINEMA DISCOVERY</div>
    <h1 class="hero-title">
      <span class="hero-red">Cine</span><span class="hero-white">AI</span>
    </h1>
    <p class="hero-subtitle">
      Hybrid recommendation engine &nbsp;·&nbsp;
      Content-Based + Collaborative Filtering &nbsp;·&nbsp;
      Real TMDB Posters
    </p>
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><span class="sn">{len(movies_df):,}</span><span class="sl">Movies</span></div>
    <div class="hero-stat"><span class="sn">{movies_df['original_language'].nunique()}</span><span class="sl">Languages</span></div>
    <div class="hero-stat"><span class="sn">{int(movies_df['vote_count'].sum()/1e6)}M+</span><span class="sl">Votes</span></div>
    <div class="hero-stat"><span class="sn">4</span><span class="sl">Algorithms</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
cfg = render_sidebar(movies_df)

# ── TABS ───────────────────────────────────────────────────────────────────────
t_discover, t_similar, t_search, t_analytics, t_watchlist = st.tabs([
    "🔥 Discover",
    "🎯 Similar Movies",
    "🔍 Search & Filter",
    "📊 Analytics",
    "❤️  Watchlist",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
with t_discover:
    st.markdown('<div class="section-title">Trending & Top Picks</div>', unsafe_allow_html=True)

    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        discover_mode = st.selectbox(
            "Show me",
            ["Top Rated", "Most Popular", "Latest Releases",
             "Hidden Gems", "Oscar Winners"],
            label_visibility="collapsed",
        )
    with sub2:
        disc_lang = st.selectbox(
            "Language",
            ["All"] + sorted(movies_df["original_language"].dropna().unique().tolist()),
            label_visibility="collapsed",
        )
    with sub3:
        disc_n = st.slider("Count", 10, 50, 20, key="disc_n")

    results = engine.discover(
        mode=discover_mode, language=disc_lang, n=disc_n
    )
    st.markdown(
        f'<div class="result-count">Showing <b>{len(results)}</b> movies</div>',
        unsafe_allow_html=True,
    )
    render_movie_grid(results, tmdb, key_prefix="disc",
                      show_score=False, watchlist_mode=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMILAR MOVIES (content-based)
# ══════════════════════════════════════════════════════════════════════════════
with t_similar:
    st.markdown('<div class="section-title">Content-Based Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="algo-badge content-badge">'
        '⚙️ TF-IDF · Cosine Similarity · Overview + Genres + Cast + Director + Keywords'
        '</div>',
        unsafe_allow_html=True,
    )

    col_pick, col_opts = st.columns([2, 1])
    with col_pick:
        titles = ["— choose a movie —"] + sorted(movies_df["title"].dropna().tolist())
        selected = st.selectbox("Pick a movie you love:", titles, key="cb_select")
    with col_opts:
        n_cb  = st.slider("Recommendations", 5, 30, 12, key="n_cb")
        metric = st.radio(
            "Similarity",
            ["Cosine", "Weighted Hybrid"],
            horizontal=True, key="cb_metric",
        )

    if selected != "— choose a movie —":
        with st.spinner("Computing similarities…"):
            recs = engine.content_based(selected, n=n_cb, metric=metric)

        row = movies_df[movies_df["title"] == selected].iloc[0]
        render_hero_card(row, tmdb)
        st.markdown("---")
        st.markdown(
            f'<div class="section-subtitle">Because you liked <b>{selected}</b></div>',
            unsafe_allow_html=True,
        )
        render_movie_grid(recs, tmdb, key_prefix="cb",
                          show_score=True, score_label="Similarity")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEARCH & FILTER
# ══════════════════════════════════════════════════════════════════════════════
with t_search:
    st.markdown('<div class="section-title">Search & Filter</div>', unsafe_allow_html=True)

    col_q, col_sort = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "q", placeholder="🔍  Title, director, cast, keyword…",
            label_visibility="collapsed",
        )
    with col_sort:
        sort_by = st.selectbox(
            "s", ["Rating ↓", "Votes ↓", "Year ↓", "Title A-Z", "Revenue ↓"],
            label_visibility="collapsed",
        )

    filtered = engine.filter_movies(
        query=query,
        genres=cfg["genres"],
        languages=cfg["languages"],
        year_range=cfg["year_range"],
        min_rating=cfg["min_rating"],
        sort_by=sort_by,
    )
    st.markdown(
        f'<div class="result-count">Found <b>{len(filtered)}</b> movies</div>',
        unsafe_allow_html=True,
    )
    if filtered.empty:
        st.markdown('<div class="empty-state">🎬 No results. Try different filters.</div>',
                    unsafe_allow_html=True)
    else:
        render_movie_grid(filtered.head(40), tmdb, key_prefix="search",
                          show_score=False, watchlist_mode=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with t_analytics:
    st.markdown('<div class="section-title">Dataset Analytics</div>', unsafe_allow_html=True)
    render_metrics_panel(engine, movies_df)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
with t_watchlist:
    st.markdown('<div class="section-title">❤️ My Watchlist</div>', unsafe_allow_html=True)

    wl = st.session_state.watchlist
    if not wl:
        st.markdown(
            '<div class="empty-state">Your watchlist is empty.<br>'
            'Click <b>+ Watchlist</b> on any movie card to add it here!</div>',
            unsafe_allow_html=True,
        )
    else:
        wl_df = movies_df[movies_df["title"].isin(wl)].copy()
        st.markdown(
            f'<div class="result-count"><b>{len(wl_df)}</b> movies saved</div>',
            unsafe_allow_html=True,
        )
        render_movie_grid(wl_df, tmdb, key_prefix="wl",
                          show_score=False, watchlist_mode=True)

        if st.button("🗑  Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
