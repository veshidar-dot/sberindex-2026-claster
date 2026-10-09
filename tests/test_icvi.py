"""Проверки индексов качества и сетей на случаях с известным ответом."""
import numpy as np
import scipy.sparse as sp

from mo_common.icvi import graph_indices, icvi, s_dbw, selftest
from pulse.cluster import merge_small
from pulse.network import knn_graph


def test_analytic_cliques():
    assert selftest()


def test_relabel_invariance():
    rng = np.random.default_rng(0)
    A = sp.random(60, 60, density=0.1, random_state=1)
    A = (A + A.T).tocsr()
    lab = rng.integers(0, 4, 60)
    perm = np.array([2, 0, 3, 1])
    a, b = graph_indices(A, lab), graph_indices(A, perm[lab])
    assert all(abs(a[k] - b[k]) < 1e-12 for k in a)


def test_good_partition_beats_random():
    """Три плотных блока: истинное разбиение лучше случайного по SW, CH, S_Dbw, AVI и MQ. AVU при равномерном
    фоне связей между блоками не зависит от разбиения (он меряет, нет ли пары групп, которые стоит слить), поэтому
    для него отдельная проверка - на пересечённом разбиении."""
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (40, 3)), rng.normal(6, 1, (40, 3)), rng.normal(-6, 1, (40, 3))]
    true = np.repeat([0, 1, 2], 40)
    B = (rng.random((120, 120)) < 0.02).astype(float)
    B += (true[:, None] == true[None, :]) * (rng.random((120, 120)) < 0.3)
    A = sp.csr_matrix(np.triu(B, 1) + np.triu(B, 1).T)
    rnd = rng.permutation(true)
    t, r = icvi(X, A, true), icvi(X, A, rnd)
    assert t["SW"] > r["SW"] and t["CH"] > r["CH"] and t["S_Dbw"] < r["S_Dbw"]
    assert t["AVI"] > r["AVI"] and t["MQ"] > r["MQ"]
    # AVU - «склеенность»: при одном и том же K = 4 разрезанный пополам блок (половины сильно связаны и их стоит
    # слить) даёт больший AVU, чем четвёртая группа, набранная поровну из всех блоков
    over = true.copy()
    over[:20] = 3
    mixed = true.copy()
    mixed[np.r_[0:7, 40:47, 80:86]] = 3
    assert graph_indices(A, over)["AVU"] > graph_indices(A, mixed)["AVU"]


def test_avu_two_clusters_constant():
    A = sp.csr_matrix(np.random.default_rng(1).random((20, 20)))
    A = A + A.T
    lab = np.random.default_rng(2).integers(0, 2, 20)
    assert abs(graph_indices(A, lab)["AVU"] - 1) < 1e-12


def test_s_dbw_separated():
    X = np.r_[np.zeros((10, 2)), np.ones((10, 2)) * 10] + np.random.default_rng(0).normal(0, .1, (20, 2))
    assert s_dbw(X, np.r_[np.zeros(10, int), np.ones(10, int)]) < 0.1


def test_knn_symmetric():
    W = np.random.default_rng(0).random((30, 30))
    A = knn_graph(W, 5)
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0


def test_merge_small():
    A = sp.csr_matrix(np.ones((6, 6)) - np.eye(6))
    lab = merge_small(np.array([0, 0, 0, 0, 1, 2]), A, 2)
    assert len(np.unique(lab)) == 2 or len(np.unique(lab)) == 1
