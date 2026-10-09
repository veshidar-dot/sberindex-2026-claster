"""Внутренние индексы качества кластеризации (ICVI) для атрибутированных сетей.

Признаковое пространство: SW (силуэт), CH (Калински-Харабаш; рядом CH/N - по рекомендации Shalileh, Antonov,
Tsyplakova 2025: CH растёт с объёмом выборки), S_Dbw (Halkidi, Vazirgiannis 2001).
Сетевое пространство (Shalileh, Antonov, Tsyplakova. Doklady Mathematics 112(3), 2025, формулы 18-22;
AVI и AVU - Biswas, Biswas, Expert Systems with Applications, 2017), для неотрицательной матрицы весов A:
  S_kl = сумма весов рёбер из кластера k в кластер l; O_k = sum_{l!=k} S_kl; In_l = sum_{k!=l} S_kl;
  AVI = mean_k S_kk / (S_kk + O_k)                          - изолированность кластеров (больше - лучше);
  AVU = (1/K) sum_k sum_{l!=k} S_kl / (O_k + In_l - S_kl)   - «склеенность» с другими кластерами (меньше - лучше);
  ANUI = AVI / (1 + AVI * AVU);
  MQ - модулярность Ньюмана (для направленного графа - направленная модулярность Лейхта-Ньюмана).
У каждого индекса есть нулевая модель: те же размеры кластеров, метки перемешаны случайно.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import calinski_harabasz_score, silhouette_score

DIRECTION = {"SW": 1, "CH": 1, "CH_per_N": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "ANUI": 1, "MQ": 1}


def s_dbw(X, lab):
    ks = np.unique(lab)
    K = len(ks)
    mem = [X[lab == k] for k in ks]
    cen = np.array([m.mean(0) for m in mem])
    vn = np.array([np.linalg.norm(m.var(0)) for m in mem])
    scat = vn.mean() / max(np.linalg.norm(X.var(0)), 1e-12)
    std = np.sqrt(vn.sum()) / K
    dens = np.array([(np.linalg.norm(m - c, axis=1) <= std).sum() for m, c in zip(mem, cen)])
    tot = 0.0
    for a in range(K):
        for b in range(K):
            if a != b:
                pts = np.vstack([mem[a], mem[b]])
                u = (cen[a] + cen[b]) / 2
                den = max(dens[a], dens[b])
                if den > 0:
                    tot += (np.linalg.norm(pts - u, axis=1) <= std).sum() / den
    return float(scat + tot / (K * (K - 1)))


def graph_indices(A, lab, directed=False):
    _, lab = np.unique(lab, return_inverse=True)
    K = lab.max() + 1
    n = len(lab)
    M = sp.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, K))
    A = sp.csr_matrix(A)
    S = (M.T @ A @ M).toarray()
    d = np.diag(S).copy()
    out = S.sum(1) - d
    inn = S.sum(0) - d
    avi = float(np.mean(np.divide(d, d + out, out=np.zeros(K), where=(d + out) > 0)))
    den = out[:, None] + inn[None, :] - S
    U = np.divide(S, den, out=np.zeros_like(S), where=den > 0)
    np.fill_diagonal(U, 0)
    avu = float(U.sum() / K)
    m = S.sum()
    if directed:                      # Лейхт-Ньюман: sum_k [S_kk/m - (out_k)(in_k)/m^2]
        q = float((d / m - S.sum(1) * S.sum(0) / m ** 2).sum())
    else:
        q = float((d / m - (S.sum(1) / m) ** 2).sum())
    return {"AVI": avi, "AVU": avu, "ANUI": avi / (1 + avi * avu), "MQ": q}


def icvi(X, A, lab, directed=False, sample=None, seed=0):
    """Все индексы. sample - подвыборка узлов для силуэта и S_Dbw (квадратичные по n)."""
    lab = np.asarray(lab)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(lab)) if sample is None or sample >= len(lab) else np.sort(rng.choice(len(lab), sample, replace=False))
    out = {"K": int(len(np.unique(lab))),
           "SW": float(silhouette_score(X[idx], lab[idx])),
           "CH": float(calinski_harabasz_score(X, lab)),
           "S_Dbw": s_dbw(X[idx], lab[idx])}
    out["CH_per_N"] = out["CH"] / len(lab)
    out.update(graph_indices(A, lab, directed))
    return out


def null_icvi(X, A, lab, reps=20, directed=False, sample=None, seed=0):
    """Среднее индексов при случайной перестановке меток (та же структура размеров кластеров)."""
    rng = np.random.default_rng(seed)
    rows = [icvi(X, A, rng.permutation(lab), directed, sample, seed) for _ in range(reps)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def selftest():
    """Две несвязные клики: AVI = 1, AVU = 0, MQ = 0.5 (аналитический случай статьи)."""
    B = np.ones((4, 4)) - np.eye(4)
    A = np.block([[B, np.zeros((4, 4))], [np.zeros((4, 4)), B]])
    r = graph_indices(A, np.array([0] * 4 + [1] * 4))
    assert abs(r["AVI"] - 1) < 1e-12 and abs(r["AVU"]) < 1e-12 and abs(r["MQ"] - 0.5) < 1e-12, r
    r2 = graph_indices(A, np.array([0] * 4 + [1] * 4), directed=True)
    assert abs(r2["MQ"] - 0.5) < 1e-12, r2
    return True


if __name__ == "__main__":
    print("selftest", selftest())
