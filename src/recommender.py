from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_similarity


class SteamRecommender:
    """
    TF‑IDF + SVD (100 dims) + K‑Medoids.
    Pode treinar, salvar, carregar e gerar recomendações Top‑N.
    """

    def __init__(self, n_components: int = 100, n_clusters: int = 160):
        self.n_components = n_components
        self.n_clusters = n_clusters

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.norm = Normalizer(copy=False)
        self.kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric="cosine",
            init="k-medoids++",
            random_state=42,
        )

        self.df = None
        self.X_reduced = None

    # ---------- treino ----------
    def fit(self, df):
        X = self.vectorizer.fit_transform(df["cbf_text"])
        X_reduced = self.svd.fit_transform(X).astype(np.float32)
        X_reduced = self.norm.fit_transform(X_reduced)

        self.kmedoids.fit(X_reduced)

        df["cluster"] = self.kmedoids.labels_
        df["is_medoid"] = False
        df.loc[self.kmedoids.medoid_indices_, "is_medoid"] = True

        self.df = df.reset_index(drop=True)
        self.X_reduced = X_reduced

    # ---------- persistência ----------
    def save(self, folder: Path):
        folder.mkdir(exist_ok=True)
        joblib.dump(self.vectorizer, folder / "tfidf.joblib")
        joblib.dump(self.svd, folder / "svd.joblib")
        joblib.dump(self.kmedoids, folder / "kmedoids.joblib")

    def load(self, folder: Path, df):
        self.vectorizer = joblib.load(folder / "tfidf.joblib")
        self.svd = joblib.load(folder / "svd.joblib")
        self.kmedoids = joblib.load(folder / "kmedoids.joblib")

        X = self.vectorizer.transform(df["cbf_text"])
        X_reduced = self.svd.transform(X).astype(np.float32)
        X_reduced = Normalizer(copy=False).fit_transform(X_reduced)

        df["cluster"] = self.kmedoids.predict(X_reduced)
        df["is_medoid"] = False
        df.loc[self.kmedoids.medoid_indices_, "is_medoid"] = True

        self.df = df.reset_index(drop=True)
        self.X_reduced = X_reduced

    # ---------- recomendação ----------
    def recommend(self, title: str, top_n: int = 10):
        if self.df is None:
            raise RuntimeError("Modelo não carregado/treinado.")

        idx_map = (
            self.df.reset_index().set_index("name")["index"].drop_duplicates()
        )
        if title not in idx_map:
            return None

        idx = idx_map[title]
        sims = cosine_similarity(
            self.X_reduced[idx : idx + 1], self.X_reduced
        ).ravel()
        top_idx = sims.argsort()[::-1][1 : top_n + 1]

        return self.df.loc[
            top_idx, ["appid", "name", "genres", "price", "cluster"]
        ]
