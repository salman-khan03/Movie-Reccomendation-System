"""
MovieLens Recommendation Demo
==============================
Command-line script that loads movies.csv + ratings.csv and runs:
  - Content-based recommender  (TF-IDF genre vectors + cosine similarity)
  - Collaborative filtering     (TruncatedSVD on user-item matrix)
  - Hybrid recommender          (blends both score types)
  - Decision Tree classifier    (predicts liked / not-liked)

Usage
-----
  python main.py --mode all --title "Toy Story" --top 10
  python main.py --mode content --title "The Matrix" --top 10
  python main.py --mode hybrid  --title "Inception"  --alpha 0.7
  python main.py --genre Action --top 10
  python main.py --list-titles
"""

from __future__ import annotations

import argparse
import difflib
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(project_dir: str, ratings_sample: int = 500_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    movies_path  = os.path.join(project_dir, "movies.csv")
    ratings_path = os.path.join(project_dir, "ratings.csv")

    for path in (movies_path, ratings_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path, nrows=ratings_sample)

    # Genre string for TF-IDF (replace | with space)
    movies["genre_str"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "", regex=False)
        .str.strip()
        .fillna("")
    )
    return movies, ratings


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

@dataclass
class MovieRecommender:
    movies:  pd.DataFrame
    ratings: pd.DataFrame

    # populated during build()
    movie_stats:     pd.DataFrame          = field(default=None, repr=False)
    tfidf:           TfidfVectorizer       = field(default=None, repr=False)
    tfidf_matrix:    object                = field(default=None, repr=False)  # sparse
    movie_factors:   np.ndarray | None     = field(default=None, repr=False)
    cf_movie_to_idx: dict                  = field(default_factory=dict, repr=False)
    cf_idx_to_movie: dict                  = field(default_factory=dict, repr=False)
    movie_title_to_id: dict               = field(default_factory=dict, repr=False)
    movie_id_to_idx:   dict               = field(default_factory=dict, repr=False)

    def build(self, top_users: int = 2_000, top_movies: int = 3_000, svd_components: int = 50) -> None:
        movies  = self.movies
        ratings = self.ratings

        # Lookup dicts
        self.movie_title_to_id = pd.Series(movies["movieId"].values, index=movies["title"]).to_dict()
        self.movie_id_to_idx   = pd.Series(movies.index, index=movies["movieId"]).to_dict()

        # Per-movie stats from ratings sample
        valid_ids = set(movies["movieId"])
        rf = ratings[ratings["movieId"].isin(valid_ids)]
        self.movie_stats = (
            rf.groupby("movieId")
            .agg(num_ratings=("rating", "count"), avg_rating=("rating", "mean"))
            .reset_index()
        )

        # TF-IDF content matrix
        self.tfidf        = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(movies["genre_str"])

        # Collaborative: top-K users × top-M movies
        top_u = rf["userId"].value_counts().head(top_users).index
        top_m = rf["movieId"].value_counts().head(top_movies).index
        cf_data = rf[rf["userId"].isin(top_u) & rf["movieId"].isin(top_m)]

        cf_users  = sorted(cf_data["userId"].unique())
        cf_movies = sorted(cf_data["movieId"].unique())
        u_idx = {u: i for i, u in enumerate(cf_users)}
        self.cf_movie_to_idx = {m: i for i, m in enumerate(cf_movies)}
        self.cf_idx_to_movie = {i: m for m, i in self.cf_movie_to_idx.items()}

        rows = cf_data["userId"].map(u_idx).values
        cols = cf_data["movieId"].map(self.cf_movie_to_idx).values
        vals = cf_data["rating"].astype(np.float32).values
        ui   = csr_matrix((vals, (rows, cols)), shape=(len(cf_users), len(cf_movies)))

        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self.movie_factors = svd.fit_transform(ui.T)
        print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    def _resolve(self, query: str) -> tuple[str, int, int]:
        """Return (title, movieId, movies_df_index)."""
        if query not in self.movie_title_to_id:
            hits = difflib.get_close_matches(query, list(self.movie_title_to_id), n=1, cutoff=0.35)
            if not hits:
                raise ValueError(f"No match for '{query}'.")
            print(f"  -> matched to '{hits[0]}'")
            query = hits[0]
        mid = self.movie_title_to_id[query]
        return query, mid, self.movie_id_to_idx[mid]

    def recommend_content(self, title: str, n: int = 10, min_ratings: int = 50) -> pd.DataFrame:
        _, mid, idx = self._resolve(title)
        sims       = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx]  = -1.0
        top_idx    = np.argsort(-sims)
        result = (
            self.movies.iloc[top_idx][["movieId", "title", "genres"]].copy()
            .merge(self.movie_stats[["movieId", "num_ratings", "avg_rating"]], on="movieId", how="left")
        )
        result["content_score"] = np.round(sims[top_idx], 4)
        result = result[result["num_ratings"].fillna(0) >= min_ratings].head(n)
        result["avg_rating"] = result["avg_rating"].round(2)
        return result.reset_index(drop=True)

    def recommend_collab(self, title: str, n: int = 10) -> pd.DataFrame:
        _, mid, _ = self._resolve(title)
        if mid not in self.cf_movie_to_idx:
            raise ValueError(f"'{title}' not in CF subset. Use content-based instead.")
        cf_i  = self.cf_movie_to_idx[mid]
        sims  = cosine_similarity(self.movie_factors[cf_i:cf_i + 1], self.movie_factors).flatten()
        sims[cf_i] = -1.0
        top_cf  = np.argsort(-sims)[:n]
        top_ids = [self.cf_idx_to_movie[i] for i in top_cf]
        score_map = {top_ids[k]: round(float(sims[top_cf[k]]), 4) for k in range(len(top_ids))}
        result = (
            self.movies[self.movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]].copy()
            .merge(self.movie_stats[["movieId", "num_ratings", "avg_rating"]], on="movieId", how="left")
        )
        result["collab_score"] = result["movieId"].map(score_map)
        result["avg_rating"]   = result["avg_rating"].round(2)
        return result.sort_values("collab_score", ascending=False).reset_index(drop=True)

    def recommend_hybrid(self, title: str, n: int = 10, alpha: float = 0.6) -> pd.DataFrame:
        _, mid, cb_idx = self._resolve(title)
        cb_sim         = cosine_similarity(self.tfidf_matrix[cb_idx], self.tfidf_matrix).flatten()
        cb_sim[cb_idx] = 0.0

        cf_sim = np.zeros(len(self.movies), dtype=np.float32)
        if mid in self.cf_movie_to_idx:
            cf_i   = self.cf_movie_to_idx[mid]
            cf_raw = cosine_similarity(self.movie_factors[cf_i:cf_i + 1], self.movie_factors).flatten()
            cf_raw[cf_i] = 0.0
            for j, score in enumerate(cf_raw):
                all_idx = self.movie_id_to_idx.get(self.cf_idx_to_movie[j])
                if all_idx is not None:
                    cf_sim[all_idx] = score

        scaler  = MinMaxScaler()
        cb_norm = scaler.fit_transform(cb_sim.reshape(-1, 1)).flatten()
        cf_norm = scaler.fit_transform(cf_sim.reshape(-1, 1)).flatten()

        hybrid         = alpha * cb_norm + (1.0 - alpha) * cf_norm
        hybrid[cb_idx] = -1.0
        top_idx        = np.argsort(-hybrid)

        result = (
            self.movies.iloc[top_idx][["movieId", "title", "genres"]].copy()
            .merge(self.movie_stats[["movieId", "num_ratings", "avg_rating"]], on="movieId", how="left")
        )
        result["hybrid_score"] = np.round(hybrid[top_idx], 4)
        result = result[result["num_ratings"].fillna(0) >= 50].head(n)
        result["avg_rating"] = result["avg_rating"].round(2)
        return result.reset_index(drop=True)

    def recommend_by_genre(self, genre: str, n: int = 10, min_ratings: int = 50) -> pd.DataFrame:
        g_lower = genre.lower()
        mask    = self.movies["genres"].str.lower().str.contains(g_lower, regex=False)
        result  = (
            self.movies[mask][["movieId", "title", "genres"]].copy()
            .merge(self.movie_stats[["movieId", "num_ratings", "avg_rating"]], on="movieId", how="left")
        )
        result = (
            result[result["num_ratings"].fillna(0) >= min_ratings]
            .sort_values("avg_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        result["avg_rating"] = result["avg_rating"].round(2)
        return result


# ---------------------------------------------------------------------------
# Decision Tree helper
# ---------------------------------------------------------------------------

def run_decision_tree(movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rf = ratings[ratings["movieId"].isin(set(movies["movieId"]))].copy()
    rf["liked"] = (rf["rating"] >= 4.0).astype(int)

    m_agg = rf.groupby("movieId").agg(
        avg_movie_rating=("rating", "mean"),
        num_movie_ratings=("rating", "count"),
    ).reset_index()
    u_agg = rf.groupby("userId").agg(
        avg_user_rating=("rating", "mean"),
        num_user_ratings=("rating", "count"),
    ).reset_index()

    genre_dummies = movies["genres"].str.get_dummies(sep="|")
    genre_dummies.columns = (
        genre_dummies.columns
        .str.replace("(no genres listed)", "no_genre", regex=False)
        .str.replace(" ", "_").str.replace("-", "_")
    )
    genre_dummies["movieId"] = movies["movieId"].values

    dt_full = (
        rf.merge(m_agg, on="movieId", how="left")
        .merge(u_agg,   on="userId",  how="left")
        .merge(genre_dummies, on="movieId", how="left")
    )

    genre_cols   = [c for c in genre_dummies.columns if c != "movieId"]
    feature_cols = ["avg_movie_rating", "num_movie_ratings",
                    "avg_user_rating",  "num_user_ratings"] + genre_cols
    clean = dt_full[feature_cols + ["liked"]].dropna()

    X = clean[feature_cols]
    y = clean["liked"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Liked", "Liked"], digits=4)
    cm     = confusion_matrix(y_test, y_pred)
    cm_df  = pd.DataFrame(
        cm,
        index=["True: Not Liked", "True: Liked"],
        columns=["Pred: Not Liked", "Pred: Liked"],
    )
    rules = export_text(clf, feature_names=feature_cols, max_depth=3)

    summary = {
        "accuracy"      : round(acc, 4),
        "n_train"       : len(X_train),
        "n_test"        : len(X_test),
        "report"        : report,
        "confusion_matrix": cm_df,
        "rules"         : rules,
    }
    return cm_df, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cosine_similarity(A, B):  # thin re-export so the dataclass methods above work
    from sklearn.metrics.pairwise import cosine_similarity as _cs
    return _cs(A, B)


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens recommendation demo")
    parser.add_argument("--title",       type=str,   default=None)
    parser.add_argument("--genre",       type=str,   default=None)
    parser.add_argument("--mode",        type=str,   choices=["content", "collab", "hybrid", "all"], default="all")
    parser.add_argument("--top",         type=int,   default=10)
    parser.add_argument("--alpha",       type=float, default=0.6)
    parser.add_argument("--min-votes",   type=int,   default=50)
    parser.add_argument("--list-titles", action="store_true")
    parser.add_argument("--titles-limit",type=int,   default=40)
    parser.add_argument("--train-tree",  action="store_true")
    parser.add_argument("--excel-out",   type=str,   default="my_results.xlsx")
    parser.add_argument("--sample",      type=int,   default=500_000, help="Max ratings rows to load")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading data...")
    movies, ratings = load_data(project_dir, ratings_sample=args.sample)
    print(f"  movies : {len(movies):,}   ratings sample : {len(ratings):,}")

    rec = MovieRecommender(movies=movies, ratings=ratings)
    print("Building recommender (TF-IDF + SVD)...")
    rec.build()
    print(f"  Content matrix : {rec.tfidf_matrix.shape}")
    print(f"  CF movie factors: {rec.movie_factors.shape}")

    if args.list_titles:
        sample = movies[["movieId", "title"]].head(args.titles_limit)
        print(sample.to_string(index=False))
        return

    sheets: dict[str, pd.DataFrame] = {}

    if args.genre:
        print(f"\nTop picks for genre '{args.genre}':")
        res = rec.recommend_by_genre(args.genre, n=args.top, min_ratings=args.min_votes)
        print(res.to_string(index=True))
        sheets["genre_based"] = res

    title = args.title
    if title is None:
        best  = rec.movie_stats.sort_values("num_ratings", ascending=False).iloc[0]
        title = rec.movies.loc[rec.movie_id_to_idx[best["movieId"]], "title"]
        print(f"\nNo title given — using most-rated: '{title}'")

    if args.mode in ("content", "all"):
        print(f"\nContent-based recommendations for '{title}':")
        res = rec.recommend_content(title, n=args.top, min_ratings=args.min_votes)
        print(res.to_string(index=True))
        sheets["content_based"] = res

    if args.mode in ("collab", "all"):
        print(f"\nCollaborative recommendations for '{title}':")
        try:
            res = rec.recommend_collab(title, n=args.top)
            print(res.to_string(index=True))
            sheets["collaborative"] = res
        except ValueError as e:
            print(f"  Skipped: {e}")

    if args.mode in ("hybrid", "all"):
        print(f"\nHybrid recommendations for '{title}' (alpha={args.alpha}):")
        res = rec.recommend_hybrid(title, n=args.top, alpha=args.alpha)
        print(res.to_string(index=True))
        sheets["hybrid"] = res

    if args.train_tree or args.mode == "all":
        print("\nTraining Decision Tree classifier...")
        cm_df, summary = run_decision_tree(movies, ratings)
        print(f"Accuracy : {summary['accuracy']:.4f}")
        print(summary["report"])
        print("Top-level rules:\n" + summary["rules"])
        sheets["decision_tree_cm"]    = summary["confusion_matrix"]
        sheets["decision_tree_rules"] = pd.DataFrame({"rules": summary["rules"].splitlines()})
        sheets["dt_summary"] = pd.DataFrame([{
            "accuracy": summary["accuracy"],
            "n_train" : summary["n_train"],
            "n_test"  : summary["n_test"],
        }])

    if sheets:
        excel_path = args.excel_out
        if not os.path.isabs(excel_path):
            excel_path = os.path.join(project_dir, excel_path)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=True)
        print(f"\nSaved to: {excel_path}")


if __name__ == "__main__":
    main()
