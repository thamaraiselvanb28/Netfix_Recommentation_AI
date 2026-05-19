"""
Metrics / Analytics panel
──────────────────────────
Renders the full Analytics tab with Plotly charts.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd


_DARK = "#0d1117"
_PAPER = "#161b22"
_TEXT  = "#c9d1d9"
_ACCENT = "#58a6ff"


def _layout(title=""):
    return dict(
        template="plotly_dark",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_DARK,
        font=dict(family="Inter", color=_TEXT, size=12),
        title=dict(text=title, font=dict(size=15, color=_TEXT)),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def render_metrics_panel(engine, movies_df: pd.DataFrame):
    df = movies_df.copy()

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🎬 Total Movies",    f"{len(df):,}")
    k2.metric("🌍 Languages",       df["original_language"].nunique())
    k3.metric("⭐ Avg Rating",      f"{df['vote_average'].mean():.2f}")
    k4.metric("🗳 Total Votes",     f"{int(df['vote_count'].sum()/1e6)}M+")
    k5.metric("📅 Year Span",
              f"{int(df['year'].min())}–{int(df['year'].max())}"
              if "year" in df.columns else "N/A")

    st.markdown("---")

    # ── Row 1: Genre bar + Language pie ────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        genre_df = engine.top_genres(15)
        fig = px.bar(
            genre_df, x="Count", y="Genre", orientation="h",
            color="Count", color_continuous_scale="Blues",
        )
        fig.update_layout(**_layout("Top Genres"))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        lang_df = engine.top_languages(10)
        fig = px.pie(
            lang_df, names="Language", values="Count",
            hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_layout(**_layout("Top Languages"))
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Movies per year + Rating distribution ───────────────────────
    c3, c4 = st.columns(2)

    with c3:
        year_df = engine.year_distribution()
        year_df = year_df[year_df["year"] >= 1950]
        fig = px.area(
            year_df, x="year", y="Count",
            color_discrete_sequence=[_ACCENT],
        )
        fig.update_layout(**_layout("Movies Released per Year"))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(
            df[df["vote_average"] > 0],
            x="vote_average", nbins=30,
            color_discrete_sequence=[_ACCENT],
        )
        fig.update_layout(**_layout("Rating Distribution"))
        fig.update_xaxes(title="Rating")
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Runtime box + Budget vs Revenue scatter ─────────────────────
    c5, c6 = st.columns(2)

    with c5:
        rt = df[(df["runtime"] > 30) & (df["runtime"] < 300)].copy()
        rt["genre_1"] = rt["genres_list"].apply(
            lambda g: g[0] if isinstance(g, list) and g else "Other"
        )
        top_genres = rt["genre_1"].value_counts().head(8).index.tolist()
        rt = rt[rt["genre_1"].isin(top_genres)]
        fig = px.box(
            rt, x="genre_1", y="runtime",
            color="genre_1",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(**_layout("Runtime by Genre (min)"))
        fig.update_xaxes(title="")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        bvr = df[(df["budget"] > 1e6) & (df["revenue"] > 1e6)].copy()
        bvr["budget_m"]  = bvr["budget"]  / 1e6
        bvr["revenue_m"] = bvr["revenue"] / 1e6
        bvr = bvr.nlargest(500, "vote_count")
        fig = px.scatter(
            bvr, x="budget_m", y="revenue_m",
            color="vote_average", size="popularity",
            hover_name="title",
            color_continuous_scale="RdYlGn",
            labels={"budget_m": "Budget ($M)", "revenue_m": "Revenue ($M)"},
            opacity=0.7,
        )
        fig.update_layout(**_layout("Budget vs Revenue (Top 500 by votes)"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Top 10 table ───────────────────────────────────────────────────────
    st.markdown("#### 🏆 Top 10 Movies by Weighted Score")
    top10 = (
        df[df["vote_count"] >= 1000]
        .nlargest(10, "score")[
            ["title", "year", "vote_average", "vote_count", "score", "genres_list"]
        ]
        .copy()
    )
    top10["Genres"] = top10["genres_list"].apply(
        lambda g: ", ".join(g[:3]) if isinstance(g, list) else ""
    )
    top10 = top10.rename(columns={
        "title": "Title", "year": "Year",
        "vote_average": "Rating", "vote_count": "Votes", "score": "Score"
    })[["Title", "Year", "Rating", "Votes", "Score", "Genres"]]
    top10["Votes"] = top10["Votes"].apply(lambda x: f"{int(x):,}")
    top10["Score"] = top10["Score"].round(3)
    st.dataframe(top10, use_container_width=True, hide_index=True)
