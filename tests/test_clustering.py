import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from src.clustering import cluster_profile, fit_pca, relabel_by_churn, scan_k


def test_scan_k_finds_three_well_separated_blobs():
    Z, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    table = scan_k(Z, ks=range(2, 6), n_init=5)
    assert table["silhouette"].idxmax() == 3
    assert table.loc[3, "stability_ari"] > 0.95
    assert table["inertia"].is_monotonic_decreasing


def test_fit_pca_explained_variance():
    Z, _ = make_blobs(n_samples=200, n_features=6, centers=3, random_state=1)
    pca = fit_pca(Z, n_components=2)
    assert pca.transform(Z).shape == (200, 2)
    ratio = pca.explained_variance_ratio_
    assert 0 < ratio.sum() <= 1 and ratio[0] >= ratio[1]


def _small():
    X = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 40, 50, 60],
            "Contract": ["m2m", "m2m", "m2m", "two", "two", "m2m"],
        }
    )
    y = pd.Series([1, 1, 0, 0, 0, 0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    return X, y, labels


def test_cluster_profile_sizes_rates_and_top_category():
    X, y, labels = _small()
    prof = cluster_profile(X, y, labels, ["tenure"], ["Contract"])
    assert prof["size"].tolist() == [3, 3]
    assert prof["share_%"].sum() == pytest.approx(100)
    assert prof.loc[0, "churn_rate_%"] == pytest.approx(100 * 2 / 3)
    assert prof.loc[1, "churn_rate_%"] == pytest.approx(0)
    assert prof.loc[0, "tenure_mean"] == pytest.approx(2)
    assert prof.loc[0, "Contract_top"].startswith("m2m (100%)")
    assert prof.loc[1, "Contract_top"].startswith("two (67%)")


def test_relabel_by_churn_orders_clusters_by_rate():
    _, y, labels = _small()
    out = relabel_by_churn(labels, y)
    # cluster 1 (0% churn) becomes 0, cluster 0 (67% churn) becomes 1
    assert out.tolist() == [1, 1, 1, 0, 0, 0]
    out2 = relabel_by_churn(np.array([5, 5, 7, 7, 9, 9]), np.array([1, 1, 0, 0, 1, 0]))
    assert out2.tolist() == [2, 2, 0, 0, 1, 1]
