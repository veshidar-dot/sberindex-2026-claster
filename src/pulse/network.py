"""Три сети на одних и тех же МО:

  comove - «движутся вместе»: корреляция месячных изменений собственных отклонений z (все 6 категорий подряд,
           каждая стандартизована), ребро только если корреляция выше квантиля q нулевой модели (месяцы
           перемешаны внутри ряда), затем k самых сильных соседей; вес - корреляция;
  road   - «рядом»: k ближайших по дорогам, вес exp(-d / d0);
  attr   - «похожи»: k ближайших по стандартизованным атрибутам, вес - гауссово ядро.
Все сети симметризуются (максимум из двух направлений) и без петель.
"""
import numpy as np
import scipy.sparse as sp


def knn_graph(W, k):
    """Оставить в каждой строке k наибольших положительных весов, симметризовать максимумом."""
    n = W.shape[0]
    W = np.where(np.isfinite(W), W, 0.0)
    np.fill_diagonal(W, 0.0)
    idx = np.argpartition(-W, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = W[rows, idx.ravel()]
    keep = vals > 0
    A = sp.csr_matrix((vals[keep], (rows[keep], idx.ravel()[keep])), shape=(n, n))
    return A.maximum(A.T).tocsr()


def comove_corr(Z, rng, n_null=1):
    """Корреляции и порог нулевой модели. Z - (n, категории, месяцы) собственных отклонений."""
    dz = np.diff(Z, axis=2)
    dz = (dz - dz.mean(2, keepdims=True)) / dz.std(2, keepdims=True)
    X = dz.reshape(dz.shape[0], -1)
    C = np.corrcoef(X)
    np.fill_diagonal(C, np.nan)
    T = dz.shape[2]
    nulls = []
    for _ in range(n_null):
        Xs = np.stack([np.stack([r[rng.permutation(T)] for r in x]) for x in dz]).reshape(dz.shape[0], -1)
        Cs = np.corrcoef(Xs)
        np.fill_diagonal(Cs, np.nan)
        nulls.append(Cs[np.triu_indices_from(Cs, 1)])
    return C, np.concatenate(nulls)


def comove_graph(C, null, q, k):
    thr = np.nanpercentile(null, q)
    W = np.where(C > thr, C, 0.0)
    return knn_graph(W, k), float(thr)


def road_graph(D, k, d0):
    W = np.exp(-np.where(np.isfinite(D), D, np.inf) / d0)
    return knn_graph(W, k)


def attr_graph(X, k):
    from sklearn.metrics import pairwise_distances
    Dx = pairwise_distances(X)
    s = np.median(np.sort(Dx, 1)[:, 1:k + 1])
    return knn_graph(np.exp(-(Dx / s) ** 2), k)
