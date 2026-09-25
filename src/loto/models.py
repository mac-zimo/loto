"""
Modèles prédictifs et détection de patterns avancés.

Fonctionnalités:
- Markov chains pour prédire la probabilité du prochain tirage
- Classification des tirages par similarité (KNN)
- Régression pour estimer les jackpot odds
- Détection d'anomalies dans les tirages
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from loto.model_significance import correlation_test_significance
from loto.config import NUM_BALLS, MAX_BALL, NUM_CHANCE_MAX


class MarkovChainAnalyzer:
    """
    Analyze les transitions entre états numériques.

    Modélise la probabilité qu'un numéro soit tiré sachant
   哪些 numéros ont été tirés précédemment.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.transition_matrix = None
        self.cond_prob = None
        self._build_matrices()

    def _build_matrices(self):
        """Construit la matrice de transition et les probabilités conditionnelles."""
        # Matrice de transition: P(next ball | current ball) approximée par co-occurrence glissante
        n = MAX_BALL + 1  # 1-indexed, index 0 unused

        # Co-occurrence sliding window (consecutive draws)
        self.cooccurrence = np.zeros((n, n), dtype=float)
        self.marginal = np.zeros(n, dtype=float)

        prev_balls = None
        for _, row in self.df.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            if not balls:
                continue

            # Marginal count
            for b in balls:
                self.marginal[b] += 1

            # Co-occurrence with previous draw
            if prev_balls is not None:
                for pb in prev_balls:
                    for b in balls:
                        self.cooccurrence[pb][b] += 1

            prev_balls = balls

        # Normaliser en probabilités conditionnelles
        row_sums = self.marginal.copy()
        row_sums[row_sums == 0] = 1  # éviter div by zero
        self.cond_prob = self.cooccurrence / row_sums[:, np.newaxis]

    def predict_next(self, last_draw_balls: list[int]) -> list[tuple[int, float]]:
        """
        Prédit les numéros les plus probables pour le prochain tirage,
        basé sur les transitions depuis les derniers tirages.

        Args:
            last_draw_balls: Les 5 numéros du dernier tirage connu

        Returns:
            Liste de (numéro, score_probabilité) triée par score décroissant
        """
        scores = np.zeros(MAX_BALL + 1)

        for ball in last_draw_balls:
            if 1 <= ball <= MAX_BALL:
                scores += self.cond_prob[ball]

        # Normaliser et exclure les numéros déjà dans le dernier tirage
        scores[last_draw_balls] = 0
        scores = scores / np.sum(scores) if np.sum(scores) > 0 else scores

        # Retourner les Top 10
        top_indices = np.argsort(scores)[::-1][:10]
        return [(int(idx), float(scores[idx])) for idx in top_indices if idx > 0]

    def get_transition_probability(self, ball_a: int, ball_b: int) -> float:
        """P(ball_b | ball_a a été tiré il y a N draws)."""
        if 1 <= ball_a <= MAX_BALL and 1 <= ball_b <= MAX_BALL:
            return float(self.cond_prob[ball_a][ball_b])
        return 0.0


class ClusterAnalyzer:
    """
    Regroupe les tirages en clusters pour identifier des profils de jeu.

    Permet de détecter si certains types de tirages (par leur composition)
    se produisent plus fréquemment que d'autres.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = None
        self.kmeans = None
        self.labels_ = None

    def _build_features(self) -> pd.DataFrame:
        """Extrait des features de chaque tirage."""
        features_list = []

        for _, row in self.df.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            if len(balls) != NUM_BALLS:
                continue

            feature_vec = [
                sum(balls),  # somme totale
                sum(1 for b in balls if b % 2 == 0) / NUM_BALLS,  # ratio pairs
                min(balls),  # minimum
                max(balls),  # maximum
                np.std(balls),  # écart-type
                (sum(balls) / NUM_BALLS),  # moyenne
                max(balls) - min(balls),  # étendue
                sum(1 for b in balls if b <= 10) / NUM_BALLS,  # part du premier range
                sum(1 for b in balls if 21 <= b <= 30) / NUM_BALLS,  # part du middle range
            ]
            features_list.append(feature_vec)

        return pd.DataFrame(features_list, columns=[
            "sum", "even_ratio", "min", "max", "std", "mean", "range", "low_fraction", "mid_fraction"
        ])

    def fit_clusters(self, n_clusters: int = 5):
        """Entraîne le clustering."""
        self.features = self._build_features()
        if len(self.features) < n_clusters * 3:
            raise ValueError(f"Pas assez de données pour {n_clusters} clusters")

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_ = self.kmeans.fit_predict(self.features.values)

    def get_cluster_profiles(self) -> pd.DataFrame:
        """Retourne le profil de chaque cluster."""
        if self.kmeans is None:
            raise RuntimeError("Call fit_clusters() first")

        profiles = []
        for label in range(len(self.kmeans.cluster_centers_)):
            mask = self.labels_ == label
            profile = {
                "cluster": label,
                "size": int(mask.sum()),
                "pct": f"{mask.mean()*100:.1f}%",
            }
            cluster_features = self.features[mask] if mask.any() else None
            if cluster_features is not None:
                profile["sum_mean"] = float(cluster_features["sum"].mean())
                profile["even_ratio_mean"] = float(cluster_features["even_ratio"].mean())
                profile["std_mean"] = float(cluster_features["std"].mean())

            profiles.append(profile)

        return pd.DataFrame(profiles).sort_values("size", ascending=False)


class AnomalyDetector:
    """
    Détecte les tirages atypiques selon divers critères.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.baseline_stats = None

    def compute_baseline(self):
        """Calcule les statistiques de référence."""
        sums = []
        even_ratios = []

        for _, row in self.df.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            if len(balls) != NUM_BALLS:
                continue

            sums.append(sum(balls))
            even_ratios.append(sum(1 for b in balls if b % 2 == 0) / NUM_BALLS)

        self.baseline_stats = {
            "sum_mean": np.mean(sums),
            "sum_std": np.std(sums),
            "even_ratio_mean": np.mean(even_ratios),
            "even_ratio_std": np.std(even_ratios),
        }

    def find_anomalies(self, threshold_sigma: float = 3.0) -> pd.DataFrame:
        """Retourne les tirages considérés comme anomales."""
        if self.baseline_stats is None:
            self.compute_baseline()

        anomalies = []

        for idx, row in self.df.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            if len(balls) != NUM_BALLS:
                continue

            draw_sum = sum(balls)
            even_ratio = sum(1 for b in balls if b % 2 == 0) / NUM_BALLS

            sum_z = abs(draw_sum - self.baseline_stats["sum_mean"]) / self.baseline_stats["sum_std"]
            even_z = abs(even_ratio - self.baseline_stats["even_ratio_mean"]) / max(self.baseline_stats["even_ratio_std"], 1e-9)

            if sum_z > threshold_sigma or even_z > threshold_sigma:
                anomalies.append({
                    "date": row.get("date_tirage", ""),
                    "sum": draw_sum,
                    "sum_z_score": round(sum_z, 2),
                    "even_ratio": round(even_ratio, 2),
                    "even_z_score": round(even_z, 2),
                })

        return pd.DataFrame(anomalies)


def run_modeling(df: pd.DataFrame) -> dict:
    """Run all modeling and returns structured results."""
    result = {}

    # Markov chain analysis
    markov = MarkovChainAnalyzer(df)
    last_balls = []
    for _, row in df.tail(1).iterrows():
        last_balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]

    result["markov_predictions"] = markov.predict_next(last_balls) if last_balls else []

    # Cluster analysis
    cluster = ClusterAnalyzer(df)
    try:
        cluster.fit_clusters(n_clusters=5)
        result["cluster_profiles"] = cluster.get_cluster_profiles().to_dict(orient="records")
    except ValueError as e:
        result["cluster_profiles"] = str(e)

    # Anomaly detection
    anomaly = AnomalyDetector(df)
    anomaly.compute_baseline()
    anomalies_df = anomaly.find_anomalies(threshold_sigma=2.5)
    result["anomalies_count"] = len(anomalies_df)
    if not anomalies_df.empty:
        result["anomalies"] = anomalies_df.head(10).to_dict(orient="records")

    # Significance test on top pairs
    result["significant_pairs"] = correlation_test_significance(df, top_n=50)

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from loto.analysis import get_dataframe
    df = get_dataframe()
    results = run_modeling(df)
    print(f"Markov predictions: {results.get('markov_predictions', [])[:5]}")
    print(f"Anomalies detected: {results.get('anomalies_count', 0)}")
