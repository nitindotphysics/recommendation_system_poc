#!/usr/bin/env python
import os
os.environ.setdefault("RUN_EVAL", "0")

import sys
import importlib
from typing import List, Any, Callable
import pandas as pd
import gradio as gr
import re  # <-- added for robust genre parsing

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

DATA_CANDIDATES = [APP_DIR, os.path.join(APP_DIR, "data")]

def _find_file(fname: str) -> str:
    for d in DATA_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return fname

MOVIES_CSV = _find_file("movies_cleaned.csv")
USERS_CSV  = _find_file("users_cleaned.csv")
RATINGS_CSV = _find_file("ratings_cleaned.csv")

# Load dataframes for UI population and fallbacks
_movies_df = pd.read_csv(MOVIES_CSV)
_ratings_df = pd.read_csv(RATINGS_CSV)

# Heuristic columns
_title_col = next((c for c in ["title","movie_title","name"] if c in _movies_df.columns), _movies_df.columns[0])
_movie_id_col = next((c for c in ["movie_id","item_id","movieId","id"] if c in _movies_df.columns), None)
if _movie_id_col is None:
    _movies_df["movie_id"] = range(1, len(_movies_df)+1)
    _movie_id_col = "movie_id"
_user_col = next((c for c in ["user_id","userId","uid","user"] if c in _ratings_df.columns), _ratings_df.columns[0])
_item_col = next((c for c in ["item_id","movie_id","movieId","iid","id"] if c in _ratings_df.columns), _ratings_df.columns[1])
_rating_col = next((c for c in ["rating","ratings","score"] if c in _ratings_df.columns), None) or "rating"

_titles = _movies_df[_title_col].dropna().astype(str).tolist()
_user_ids = sorted(pd.unique(_ratings_df[_user_col]).tolist())

# Common genre names for one-hot detection (if your CSV uses columns per-genre)
_KNOWN_GENRES = [
    "Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama",
    "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
    "War","Western","Family","Biography","History","Sport","Music"
]

# Lazy import Rec_Final (heavy work lives there)
_rec_mod = None
def _rec():
    global _rec_mod
    if _rec_mod is None:
        _rec_mod = importlib.import_module("recommender")
    return _rec_mod

def _to_titles(out: Any) -> List[str]:
    if out is None:
        return []
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            return []
        first = out[0]
        if isinstance(first, (list, tuple)) and len(first) >= 1:
            return [str(x[0]) for x in out]
        else:
            return [str(x) for x in out]
    return [str(out)]

def _user_unrated_items(user_id: int):
    user_rated = set(_ratings_df.loc[_ratings_df[_user_col]==user_id, _item_col].tolist())
    all_items = set(_movies_df[_movie_id_col].tolist())
    return [iid for iid in all_items if iid not in user_rated]

def _popular_fallback(user_id: int, top_n: int = 10):
    unrated = set(_user_unrated_items(user_id))
    pop = _ratings_df.groupby(_item_col)[_rating_col].count().sort_values(ascending=False).index.tolist()
    filtered = [iid for iid in pop if iid in unrated][:top_n]
    if not filtered:
        filtered = pop[:top_n]
    return _movies_df.set_index(_movie_id_col).loc[filtered, _title_col].dropna().astype(str).tolist()

def _safe_generate_with_predictor(mod, predictor: Callable, user_id: int, top_n: int) -> List[str]:
    # Try module's generate_recommendations; if user is missing, brute-score unrated items instead.
    gen = getattr(mod, "generate_recommendations", None)
    if callable(gen):
        try:
            out = gen(int(user_id), predictor, n=int(top_n))
            titles = _to_titles(out)
            if titles:
                return titles
        except Exception:
            pass
    # Brute-score unrated items with predictor
    titles_scores = []
    unrated = _user_unrated_items(int(user_id))
    for iid in unrated:
        try:
            score = predictor(int(user_id), int(iid))
        except Exception:
            score = 0
        if score and score > 0:
            titles_scores.append((int(iid), float(score)))
    if not titles_scores:
        return _popular_fallback(int(user_id), int(top_n))
    titles_scores.sort(key=lambda x: x[1], reverse=True)
    top = [iid for iid,_ in titles_scores[:int(top_n)]]
    return _movies_df.set_index(_movie_id_col).loc[top, _title_col].dropna().astype(str).tolist()

# ---------- Content-Based ----------
def content_based_recommend(user_id: int, top_n: int = 10):
    mod = _rec()
    # Prefer explicit content-based predictor if present
    predictor = getattr(mod, "predict_rating_fast", None)
    if callable(predictor):
        try:
            titles = _safe_generate_with_predictor(mod, predictor, int(user_id), int(top_n))
            if not titles:
                titles = _popular_fallback(int(user_id), int(top_n))
            return pd.DataFrame({"Movie": titles})
        except Exception as e:
            return pd.DataFrame({"Movie": [f"ERROR: {e}"]})
    # If a function named content-based exists, try it (rare)
    cb_fn = getattr(mod, "content_based_recommendations", None)
    if callable(cb_fn):
        try:
            titles = _to_titles(cb_fn(int(user_id), int(top_n)))
            if not titles:
                titles = _popular_fallback(int(user_id), int(top_n))
            return pd.DataFrame({"Movie": titles})
        except Exception as e:
            return pd.DataFrame({"Movie": [f"ERROR: {e}"]})
    # Last resort: popularity
    return pd.DataFrame({"Movie": _popular_fallback(int(user_id), int(top_n))})

# ---------- User-User CF ----------
def user_user_recommend(user_id: int, top_n: int = 10):
    mod = _rec()
    predictor = getattr(mod, "user_based_predict", None)
    if callable(predictor):
        titles = _safe_generate_with_predictor(mod, predictor, int(user_id), int(top_n))
        return pd.DataFrame({"Movie": titles})
    return pd.DataFrame({"Movie": _popular_fallback(int(user_id), int(top_n))})

# ---------- Item-Item CF ----------
def item_item_recommend(user_id: int, top_n: int = 10):
    mod = _rec()
    predictor = getattr(mod, "item_based_predict", None)
    if callable(predictor):
        titles = _safe_generate_with_predictor(mod, predictor, int(user_id), int(top_n))
        return pd.DataFrame({"Movie": titles})
    return pd.DataFrame({"Movie": _popular_fallback(int(user_id), int(top_n))})

# ---------- SVD (Surprise) ----------
def svd_recommend(user_id: int, top_n: int = 10):
    mod = _rec()
    # 1) If there's an explicit recommender, use it
    for name in ["svd_recommend", "svd_recommendations", "svd_topn"]:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                titles = _to_titles(fn(int(user_id), int(top_n)))
                if not titles:
                    titles = _popular_fallback(int(user_id), int(top_n))
                return pd.DataFrame({"Movie": titles})
            except Exception as e:
                return pd.DataFrame({"Movie": [f"ERROR: {e}"]})
    # 2) If a predictor exists, use generator path
    svd_pred = getattr(mod, "svd_predict", None)
    if callable(svd_pred):
        titles = _safe_generate_with_predictor(mod, svd_pred, int(user_id), int(top_n))
        return pd.DataFrame({"Movie": titles})
    # 3) If Surprise algo is trained (mod.algo), score unrated directly
    algo = getattr(mod, "algo", None)
    if algo is not None and hasattr(algo, "predict"):
        try:
            unrated = _user_unrated_items(int(user_id))
            scored = []
            for iid in unrated:
                try:
                    est = algo.predict(int(user_id), int(iid)).est
                except Exception:
                    est = 0
                scored.append((iid, est))
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [iid for iid,_ in scored[:int(top_n)]]
            titles = _movies_df.set_index(_movie_id_col).loc[top, _title_col].dropna().astype(str).tolist()
            if not titles:
                titles = _popular_fallback(int(user_id), int(top_n))
            return pd.DataFrame({"Movie": titles})
        except Exception as e:
            return pd.DataFrame({"Movie": [f"ERROR: {e}"]})
    # 4) Fallback
    return pd.DataFrame({"Movie": _popular_fallback(int(user_id), int(top_n))})

# --- Movie details as a two-column table with all genres in ONE row ---
def _extract_genres_for_row(row: pd.Series) -> str:
    """Return a single string with all genres for the movie."""
    # 1) If a 'genres' column exists
    if "genres" in row.index and pd.notna(row["genres"]):
        g = row["genres"]
        if isinstance(g, str):
            parts = [x.strip() for x in re.split(r"[|,;/]", g) if x.strip()]
            return ", ".join(parts) if parts else g.strip()
        if isinstance(g, (list, tuple, set)):
            return ", ".join([str(x).strip() for x in g if str(x).strip()])
    # 2) Try one-hot genre columns
    present_cols = [c for c in _KNOWN_GENRES if c in _movies_df.columns]
    picked = []
    for c in present_cols:
        try:
            val = row[c]
            if pd.notna(val) and (val == 1 or val is True or str(val).strip() == "1"):
                picked.append(c)
        except Exception:
            pass
    if picked:
        return ", ".join(picked)
    return "(none)"

def movie_details_table(title: str):
    if not title:
        return pd.DataFrame({"Field": [], "Value": []})
    rowdf = _movies_df[_movies_df[_title_col] == title]
    if rowdf.empty:
        return pd.DataFrame({"Field": ["info"], "Value": ["Title not found."]})
    row = rowdf.iloc[0]  # Series
    genres_str = _extract_genres_for_row(row)

    # Build fields (ensure 'genres' always present and in one row)
    fields = []
    values = []
    # Title first
    fields.append("title"); values.append(row.get(_title_col))
    # Genres in one row
    fields.append("genres"); values.append(genres_str)
    # A few other useful fields if present
    for k in ["release_date", "year", "imdb_url", "movie_id", "movieId", "id"]:
        if k in row.index:
            fields.append(k); values.append(row.get(k))
    return pd.DataFrame({"Field": fields, "Value": values})

with gr.Blocks(title="Movie Recommender Demo App") as demo:
    gr.Markdown("# Movie Recommender\nSelect an approach and get movie suggestions (titles only).")

    with gr.Tabs():
        with gr.Tab("Content-Based"):
            with gr.Row():
                user_in_cb = gr.Dropdown(choices=_user_ids, value=_user_ids[0] if _user_ids else None, label="User ID", interactive=True)
                topn_in = gr.Slider(1, 20, value=10, step=1, label="Top N")
            cb_out = gr.Dataframe(headers=["Movie"], label="Recommendations", interactive=False)
            gr.Button("Recommend").click(content_based_recommend, inputs=[user_in_cb, topn_in], outputs=[cb_out])

        with gr.Tab("User-User CF"):
            with gr.Row():
                user_in_uu = gr.Dropdown(choices=_user_ids, value=_user_ids[0] if _user_ids else None, label="User ID", interactive=True)
                topn_uu = gr.Slider(1, 20, value=10, step=1, label="Top N")
            uu_out = gr.Dataframe(headers=["Movie"], label="Recommendations", interactive=False)
            gr.Button("Recommend").click(user_user_recommend, inputs=[user_in_uu, topn_uu], outputs=[uu_out])

        with gr.Tab("Item-Item CF"):
            with gr.Row():
                user_in_ii = gr.Dropdown(choices=_user_ids, value=_user_ids[0] if _user_ids else None, label="User ID", interactive=True)
                topn_ii = gr.Slider(1, 20, value=10, step=1, label="Top N")
            ii_out = gr.Dataframe(headers=["Movie"], label="Recommendations", interactive=False)
            gr.Button("Recommend").click(item_item_recommend, inputs=[user_in_ii, topn_ii], outputs=[ii_out])

        with gr.Tab("SVD"):
            with gr.Row():
                user_in_svd = gr.Dropdown(choices=_user_ids, value=_user_ids[0] if _user_ids else None, label="User ID", interactive=True)
                topn_svd = gr.Slider(1, 20, value=10, step=1, label="Top N")
            svd_out = gr.Dataframe(headers=["Movie"], label="Recommendations", interactive=False)
            gr.Button("Recommend").click(svd_recommend, inputs=[user_in_svd, topn_svd], outputs=[svd_out])

        with gr.Tab("Movie Details"):
            title_in = gr.Dropdown(choices=_titles, label="Pick a movie", interactive=True)
            details = gr.Dataframe(headers=["Field", "Value"], label="Details", interactive=False)
            gr.Button("Show Details").click(movie_details_table, inputs=[title_in], outputs=[details])

    gr.Markdown("Your personalized movie recommender app.\n\n")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, show_error=True)
