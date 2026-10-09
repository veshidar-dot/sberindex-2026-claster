"""Сравнение методов: сетка параметров, все индексы качества против нулевой модели, устойчивость, выбор.

    python -m pulse.compare

Индексы считаются в одном и том же пространстве: признаки - стандартизованные атрибуты, сеть - «движутся вместе».
Для каждого разбиения - превышение над нулевой моделью (те же размеры групп, метки перемешаны): индексы зависят
от K, поэтому сравнивать сырые значения разных K некорректно (Shalileh, Antonov, Tsyplakova 2025).
Выбор - сведение ранжирований по семи индексам (Борда и Коупленд) среди лучших конфигураций каждого метода.
"""
import json

import numpy as np
import pandas as pd
from .config import load_config
from sklearn.metrics import adjusted_rand_score

from mo_common.data import load_dict, load_panel, road_matrix
from mo_common.icvi import DIRECTION, icvi, null_icvi
from .cluster import joint_graph, kmeans_attr, leiden, merge_small, spectral_joint, ward_road
from .features import decompose, window_features
from .network import attr_graph, comove_corr, comove_graph, road_graph

INDICES = ["SW", "CH_per_N", "S_Dbw", "AVI", "AVU", "ANUI", "MQ"]


def robust_z(F, clip):
    med = F.median()
    iqr = (F.quantile(0.75) - F.quantile(0.25)).replace(0, 1)
    return ((F - med) / (iqr / 1.349)).clip(-clip, clip)


def prepare(cfg):
    root = cfg["data_root"]
    P = load_panel(root)
    Z, N = decompose(P)
    full = np.isfinite(P.V).all((1, 2))
    F = window_features(P, Z, N, list(range(len(P.months)))).loc[P.ids[full]]
    X = robust_z(F[cfg["features"]], cfg["clip"]).to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    C, null = comove_corr(Z[full], rng)
    g = cfg["graphs"]
    A_co, thr = comove_graph(C, null, g["comove"]["q"], g["comove"]["k"])
    D = road_matrix(root, P.ids[full])
    A_road = road_graph(D, g["road"]["k"], g["road"]["d0_km"])
    A_attr = attr_graph(X, g["attr"]["k"])
    return dict(P=P, Z=Z, N=N, full=full, F=F, X=X, C=C, thr=thr, A_co=A_co, A_road=A_road, A_attr=A_attr, D=D)


def partitions(cfg, d):
    gr, seed, ms = cfg["grid"], cfg["seed"], cfg["grid"]["min_size"]
    X, A_co, A_attr, A_road = d["X"], d["A_co"], d["A_attr"], d["A_road"]
    for K in gr["K"]:
        yield "kmeans_attr", {"K": K}, kmeans_attr(X, K, seed)
        yield "ward_road", {"K": K}, ward_road(X, A_road, K)
        for a in gr["alpha"]:
            yield "spectral_joint", {"K": K, "alpha": a}, spectral_joint(A_attr, A_co, K, a, seed)
    for r in gr["leiden_resolution"]:
        yield "leiden_comove", {"res": r}, merge_small(leiden(A_co, r, seed), A_co, ms)
        for a in gr["alpha"]:
            J = joint_graph(A_attr, A_co, a)
            yield "leiden_joint", {"res": r, "alpha": a}, merge_small(leiden(J, r, seed), J, ms)


def stability(fn, n, reps, frac, seed):
    rng = np.random.default_rng(seed)
    full = fn(np.arange(n))
    s = []
    for _ in range(reps):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        s.append(adjusted_rand_score(full[idx], fn(idx)))
    return float(np.mean(s))


def main():
    cfg = load_config()
    from pathlib import Path
    out = Path(cfg["out"]); out.mkdir(exist_ok=True)
    d = prepare(cfg)
    ic = cfg["icvi"]
    print(f"МО: {len(d['X'])}, порог корреляции {d['thr']:.3f}, рёбер comove {d['A_co'].nnz // 2}, road {d['A_road'].nnz // 2}", flush=True)
    rows, labels = [], {}
    nulls = {}
    for method, par, lab in partitions(cfg, d):
        K = len(np.unique(lab))
        r = icvi(d["X"], d["A_co"], lab, sample=ic["sample"], seed=cfg["seed"])
        key = (K, tuple(np.sort(np.bincount(lab))))
        if K not in nulls:
            nulls[K] = null_icvi(d["X"], d["A_co"], lab, ic["null_reps"], sample=ic["sample"], seed=cfg["seed"])
        nl = nulls[K]
        road = icvi(d["X"], d["A_road"], lab, sample=200, seed=cfg["seed"])
        row = {"method": method, **par, **r, "AVI_road": road["AVI"], "MQ_road": road["MQ"]}
        for k in INDICES:
            row[f"{k}_lift"] = DIRECTION[k] * (r[k] - nl[k])
        rows.append(row)
        labels[f"{method}|{json.dumps(par)}"] = lab.tolist()
        print(f"  {method} {par}: K={K} SW={r['SW']:.3f} AVI={r['AVI']:.3f} MQ={r['MQ']:.3f}", flush=True)
    t = pd.DataFrame(rows)
    # ранги по превышению над нулём внутри всей сетки, затем лучшая конфигурация каждого метода
    lift = t[[f"{k}_lift" for k in INDICES]]
    t["borda"] = lift.rank(ascending=False).sum(1)
    t.to_csv(out / "compare_grid.csv", index=False, encoding="utf-8")
    best = t.loc[t.groupby("method").borda.idxmin()].copy()
    # Коупленд между лучшими конфигурациями методов по семи индексам
    L = best[[f"{k}_lift" for k in INDICES]].to_numpy()
    wins = (L[:, None, :] > L[None, :, :]).sum(2)
    best["copeland"] = (wins > wins.T).sum(1) - (wins < wins.T).sum(1)
    # устойчивость лучших конфигураций
    X, A_co, A_attr, A_road, seed = d["X"], d["A_co"], d["A_attr"], d["A_road"], cfg["seed"]
    stab = []
    for _, b in best.iterrows():
        m = b.method
        def fn(idx, b=b, m=m):
            if m == "kmeans_attr":
                return kmeans_attr(X[idx], int(b.K), seed)
            if m == "ward_road":
                return ward_road(X[idx], A_road[idx][:, idx], int(b.K))
            if m == "spectral_joint":
                return spectral_joint(A_attr[idx][:, idx], A_co[idx][:, idx], int(b.K), b.alpha, seed)
            if m == "leiden_comove":
                return leiden(A_co[idx][:, idx], b.res, seed)
            return leiden(joint_graph(A_attr[idx][:, idx], A_co[idx][:, idx], b.alpha), b.res, seed)
        stab.append(stability(fn, len(X), ic["stability_reps"], ic["stability_frac"], seed))
    best["stability_ARI"] = stab
    best = best.sort_values(["copeland", "borda"], ascending=[False, True])
    best.to_csv(out / "compare_best.csv", index=False, encoding="utf-8")
    json.dump(labels, open(out / "labels.json", "w"), ensure_ascii=False)
    pd.set_option("display.width", 250)
    cols = ["method", "K", "alpha", "res", "SW", "CH_per_N", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "AVI_road", "copeland", "borda", "stability_ARI"]
    print(best[[c for c in cols if c in best]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
