# 🎬 CineAI v2 — TMDB-Powered Movie Recommendation Engine

A full-stack Streamlit application that combines content-based filtering, collaborative signals, and real TMDB movie posters into a polished cinema discovery experience.

---

## Features

| Tab | Description |
|---|---|
| 🔥 Discover | Top Rated, Most Popular, Hidden Gems, Latest Releases, Oscar Winners |
| 🎯 Similar Movies | TF-IDF cosine similarity + weighted hybrid over overview, genres, cast, director, keywords |
| 🔍 Search & Filter | Full-text search + genre / language / year / rating facets |
| 📊 Analytics | Genre distribution, language pie, rating histogram, budget vs revenue scatter, top 10 table |
| ❤️ Watchlist | Persistent in-session watchlist with add/remove |

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/your-username/cineai.git
cd cineai
```

### 2. Virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your TMDB API key
# Get a free key at: https://www.themoviedb.org/settings/api
```

### 5. Add the dataset

**Option A — Manual (recommended for first run):**
Download from [Kaggle TMDB Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and place these three files in `data/raw/`:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

**Option B — Auto-download via kagglehub:**
Place your `kaggle.json` at `~/.kaggle/kaggle.json` and the app downloads on first run.

### 6. Run

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`. Dataset is cached to `data/cache/movies_clean.parquet` after the first load (~30s).

---

## Project Structure

```
cineai/
├── app.py                        # Main Streamlit entrypoint
├── requirements.txt
├── .env.example                  # Copy to .env and add your TMDB key
├── .gitignore
├── README.md
│
├── assets/
│   └── styles.css                # Dark cinema UI theme
│
├── recommender/
│   ├── __init__.py
│   ├── data_loader.py            # CSV loading, merging, cleaning → parquet cache
│   ├── engine.py                 # TF-IDF, discover, filter, collab signals
│   └── tmdb_client.py            # TMDB API: poster URLs, trailers, caching
│
├── components/
│   ├── __init__.py
│   ├── movie_card.py             # render_movie_grid, render_hero_card
│   ├── sidebar.py                # Global filter sidebar
│   └── metrics.py                # Analytics tab (Plotly charts + KPIs)
│
└── data/
    ├── raw/                      # Place CSVs here (gitignored)
    └── cache/                    # Auto-generated parquet + TMDB JSON cache
```

---

## Requirements

- Python 3.9 – 3.12
- Free TMDB API key (for real posters — app works without it, shows placeholders)
- TMDB Kaggle dataset CSVs

---

## Suggested Future Upgrades

- Replace TF-IDF with transformer embeddings (sentence-transformers)
- Add MLflow experiment tracking
- Add user login + persistent watchlist (SQLite / Supabase)
- Deploy with Docker + CI/CD on Render or Railway
- Add threshold tuning and prediction calibration
- Real collaborative filtering with implicit ratings

---

## License

MIT
