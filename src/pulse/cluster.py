"""Методы кластеризации, которые сравниваются на одних и тех же МО и одних индексах:

  kmeans_attr     - k-средних только по атрибутам (сеть не используется) - база «без сети»;
  leiden_comove   - Leiden на сети «движутся вместе» (атрибуты не используются) - база «только сеть»;
  spectral_joint  - атрибутированная: спектральная кластеризация на сумме нормированных матриц
                    alpha * attr + (1 - alpha) * comove (одновременно похожи и движутся вместе);
  leiden_joint    - Leiden на той же совместной сети (число групп задаёт разрешение, а не K);
  ward_road       - Уорд по атрибутам с ограничением связности по сети дорог: группы - сплошные территории
                    (контраст: «похожий пульс рядом» против «похожий пульс где угодно»).
"""
import numpy as np
import scipy.sparse as sp


def _norm(A):
    d = np.asarray(A.sum(1)).ravel()
    d = np.where(d > 0, 1 / np.sqrt(d), 0.0)
    D = sp.diags(d)
    return D @ A @ D


def kmeans_attr(X, K, seed):
    from sklearn.cluster import KMeans
    return KMeans(K, n_init=10, random_state=seed).fit_predict(X)


def spectral_joint(A_attr, A_net, K, alpha, seed):
    from sklearn.cluster import SpectralClustering
    W = alpha * _norm(A_attr) + (1 - alpha) * _norm(A_net)
    W = sp.csr_matrix(W)
    return SpectralClustering(K, affinity="precomputed", random_state=seed, assign_labels="cluster_qr").fit_predict(W)


def leiden(A, resolution, seed):
    import igraph as ig
    import leidenalg as la
    A = sp.triu(sp.csr_matrix(A), 1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())), directed=False)
    g.es["weight"] = A.data.tolist()
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed)
    return np.array(p.membership)


def joint_graph(A_attr, A_net, alpha):
    return sp.csr_matrix(alpha * _norm(A_attr) + (1 - alpha) * _norm(A_net))


def ward_road(X, A_road, K):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(K, linkage="ward", connectivity=A_road).fit_predict(X)


def merge_small(lab, A, min_size):
    """Мелкие группы (меньше min_size МО) присоединяются к группе, с которой у них больше всего веса рёбер."""
    lab = lab.copy()
    A = sp.csr_matrix(A)
    while True:
        u, cnt = np.unique(lab, return_counts=True)
        small = u[cnt < min_size]
        if not len(small) or len(u) <= 2:
            break
        g = small[np.argmin(cnt[cnt < min_size])]
        idx = np.where(lab == g)[0]
        w = np.asarray(A[idx].sum(0)).ravel()
        tot = {k: w[lab == k].sum() for k in u if k != g}
        tgt = max(tot, key=tot.get) if max(tot.values()) > 0 else u[u != g][0]
        lab[idx] = tgt
    _, lab = np.unique(lab, return_inverse=True)
    return lab
