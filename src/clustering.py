"""Customer segmentation helpers: K-Means scan, PCA embedding and cluster profiles."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from src.data import RANDOM_STATE


def scan_k(Z: np.ndarray,ks=range(2, 9),random_state: int = RANDOM_STATE,n_init: int = 10,) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Z)
        other = KMeans(n_clusters=k, n_init=n_init, random_state=random_state + 1).fit(Z)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(Z, km.labels_)),
                "stability_ari": float(adjusted_rand_score(km.labels_, other.labels_)),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def fit_pca(Z: np.ndarray, n_components: int | None = None, random_state: int = RANDOM_STATE) -> PCA:
    return PCA(n_components=n_components, random_state=random_state).fit(Z)


def cluster_profile(X: pd.DataFrame,y: pd.Series,labels,num_cols: list[str],cat_cols: list[str],) -> pd.DataFrame:
    labels = np.asarray(labels)
    out = {}
    for c in sorted(np.unique(labels)):
        mask = labels == c
        row = {
            "size": int(mask.sum()),
            "share_%": 100 * mask.mean(),
            "churn_rate_%": 100 * float(np.mean(y.to_numpy()[mask])),
        }
        for col in num_cols:
            row[f"{col}_mean"] = float(X.loc[mask, col].mean())
        for col in cat_cols:
            counts = X.loc[mask, col].astype(str).value_counts(normalize=True)
            row[f"{col}_top"] = f"{counts.index[0]} ({100 * counts.iloc[0]:.0f}%)"
        out[c] = row
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("cluster")


def relabel_by_churn(labels, y) -> np.ndarray:
    labels = np.asarray(labels)
    y = np.asarray(y)
    rates = {c: float(y[labels == c].mean()) for c in np.unique(labels)}
    mapping = {old: new for new, old in enumerate(sorted(rates, key=rates.get))}
    return np.array([mapping[c] for c in labels])
