"""Итоговое разбиение, динамическая сеть и проверка смысла типов.

    python -m pulse.final

1. Итог - лучшая по сведению ранжирований конфигурация (out/compare_best.csv, первая строка).
2. Уверенность в типе каждого МО: 30 перезапусков на 90 % подвыборках с разными seed; метки запуска
   сопоставляются с итоговыми венгерским алгоритмом; уверенность - доля запусков, где МО попал в свой тип.
3. Портрет типа: средние атрибутов и отклонение от общего среднего в долях стандарта (правило интерпретации
   Миркина), состав по субъектам, AMI типов с делением на субъекты.
4. Связь с другими задачами: частота подтверждённых сдвигов (сам ряд и ряд относительно соседей) и, если в
   config.local.yaml указан forecasting_repo, ошибка прогноза ансамбля из работы по прогнозированию.
5. Динамическая атрибутированная сеть: в каждом скользящем 12-месячном окне заново считаются атрибуты МО и сеть
   «движутся вместе» (порог - та же квантиль нулевой модели окна), тот же метод кластеризации. Каждая группа окна
   получает итоговый тип большинства своих МО; сплочённость типа в окне - доля его МО в одной группе окна (не зависит
   от нумерации групп). Матрицы переходов между соседними окнами, доля МО, сохранивших тип, «устойчивое ядро»
   (МО во всех окнах в той же группе, что и большинство МО его типа).
6. Кластеры изменений: k-средних по разности атрибутов 2024 и 2023 гг. - не «какой МО», а «как он изменился».
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

from mo_common.data import load_dict, road_matrix
from .cluster import joint_graph, kmeans_attr, leiden, merge_small, spectral_joint
from .compare import prepare, robust_z
from .config import load_config
from .features import shift_history, window_features
from .network import attr_graph, comove_corr, comove_graph


def match(ref, lab):
    """Перенумеровать lab так, чтобы совпадения с ref были максимальны (лишние группы получают новые номера)."""
    K1, K2 = ref.max() + 1, lab.max() + 1
    M = np.zeros((K1, K2))
    for a, b in zip(ref, lab):
        M[a, b] += 1
    r, c = linear_sum_assignment(-M)
    mp = {cc: rr for rr, cc in zip(r, c)}
    extra = iter(range(K1, K1 + K2))
    for b in range(K2):
        if b not in mp:
            mp[b] = next(extra)
    return np.array([mp[b] for b in lab])


def run_method(best, X, A_attr, A_co, seed, min_size):
    m = best["method"]
    if m == "leiden_joint":
        J = joint_graph(A_attr, A_co, best["alpha"])
        return merge_small(leiden(J, best["res"], seed), J, min_size)
    if m == "spectral_joint":
        return spectral_joint(A_attr, A_co, int(best["K"]), best["alpha"], seed)
    if m == "leiden_comove":
        return merge_small(leiden(A_co, best["res"], seed), A_co, min_size)
    return kmeans_attr(X, int(best["K"]), seed)


def forecast_errors(repo, ids):
    """Ошибка прогноза ансамбля в % от факта для «Всех категорий» (репозиторий veshidar-dot/sberindex-2026)."""
    sys.path.insert(0, str(Path(repo)))
    from sbi.config import load_config as sbi_config
    from sbi.data import load_panel as sbi_panel
    from sbi.report import load_forecasts
    cfg = sbi_config()
    P = sbi_panel(cfg)
    F, O = load_forecasts(cfg, ["ensemble"])["ensemble"]
    rows = np.where(P.cat == 0)[0]
    tid = P.keys.territory_id.to_numpy()[rows]
    out = {}
    for h in (1, 6):
        e = [np.abs(F[k, rows, h - 1] / P.V[rows, o + h] - 1) for k, o in enumerate(O) if o + h < P.T]
        out[f"mape_h{h}"] = pd.Series(np.nanmean(np.array(e), 0) * 100, index=tid)
    return pd.DataFrame(out).reindex(ids)


def main():
    cfg = load_config()
    out = Path(cfg["out"])
    d = prepare(cfg)
    P, X, F = d["P"], d["X"], d["F"]
    ids = P.ids[d["full"]]
    best = pd.read_csv(out / "compare_best.csv").iloc[0].to_dict()
    seed, ms = cfg["seed"], cfg["grid"]["min_size"]
    lab = run_method(best, X, d["A_attr"], d["A_co"], seed, ms)
    K = lab.max() + 1
    print("итог:", best["method"], {k: best[k] for k in ("K", "alpha", "res") if k in best and best[k] == best[k]}, "K =", K, flush=True)

    # 2. уверенность
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(ids)); seen = np.zeros(len(ids))
    for r in range(30):
        idx = np.sort(rng.choice(len(ids), int(0.9 * len(ids)), replace=False))
        l2 = run_method(best, X[idx], d["A_attr"][idx][:, idx], d["A_co"][idx][:, idx], seed + r + 1, ms)
        l2 = match(lab[idx], l2)
        hits[idx] += (l2 == lab[idx]); seen[idx] += 1
    conf = hits / np.maximum(seen, 1)

    # 3-4. портрет и связь с другими задачами
    dic = load_dict(cfg["data_root"]).set_index("territory_id").reindex(ids)
    S = shift_history(P, d["Z"], road_matrix(cfg["data_root"], P.ids), ref_months=6).loc[ids]
    T = F.copy()
    T["type"] = lab
    T["confidence"] = conf
    T = T.join(S)
    if cfg.get("forecasting_repo"):
        T = T.join(forecast_errors(cfg["forecasting_repo"], ids))
    T["region"] = dic.region_name.to_numpy()
    T["name"] = dic.municipal_district_name.to_numpy()
    T["lat"] = dic.municipal_district_center_lat.to_numpy()
    T["lon"] = dic.municipal_district_center_lon.to_numpy()
    allf = cfg["features"] + ["sync", "food_share", "mkt_share"]
    prof = T.groupby("type")[allf].mean()
    mirkin = (prof - T[allf].mean()) / T[allf].std()
    sizes = T.groupby("type").size()
    regions = {k: T[T.type == k].region.value_counts().head(6).to_dict() for k in range(K)}
    vcols = [c for c in ["shifts_own", "shifts_rel", "mape_h1", "mape_h6", "confidence"] if c in T]
    val = T.groupby("type")[vcols].mean()
    ami_reg = adjusted_mutual_info_score(T.region, T.type)

    # 5. динамическая атрибутированная сеть по окнам
    feats = cfg["features"]
    med = F[feats].median()
    scale = (F[feats].quantile(0.75) - F[feats].quantile(0.25)).replace(0, 1) / 1.349
    g = cfg["graphs"]
    wlab, wami, cohesion, with_type = {}, {}, {}, {}
    for name, (a, b) in cfg["windows"].items():
        mm = list(range(P.months.index(a), P.months.index(b) + 1))
        Fw = window_features(P, d["Z"], d["N"], mm).loc[ids]
        Xw = ((Fw[feats] - med) / scale).clip(-cfg["clip"], cfg["clip"]).to_numpy()
        Cw, nullw = comove_corr(d["Z"][d["full"]][:, :, mm], np.random.default_rng(seed))
        A_co_w, _ = comove_graph(Cw, nullw, g["comove"]["q"], g["comove"]["k"])
        A_at_w = attr_graph(Xw, g["attr"]["k"])
        lw = run_method(best, Xw, A_at_w, A_co_w, seed, ms)
        # группа окна получает тот итоговый тип, из которого в неё попало больше всего МО (не зависит от номеров
        # групп; две группы окна могут соответствовать одному типу, если тип в этом окне распался)
        ct = pd.crosstab(lw, lab)
        wlab[name] = ct.idxmax(axis=1).reindex(lw).to_numpy()
        cohesion[name] = {int(k): float(pd.Series(lw[lab == k]).value_counts(normalize=True).iloc[0]) for k in range(K)}
        modal = {k: pd.Series(lw[lab == k]).mode().iloc[0] for k in range(K)}
        with_type[name] = lw == np.array([modal[k] for k in lab])     # МО в той же группе окна, что большинство его типа
        wami[name] = float(adjusted_mutual_info_score(lab, lw))
        print(f"  окно {name}: групп {lw.max() + 1}, AMI с итоговыми типами {wami[name]:.3f}", flush=True)
    names = list(cfg["windows"])
    Kw = K
    trans = {f"{names[i]} -> {names[i + 1]}": pd.crosstab(wlab[names[i]], wlab[names[i + 1]]).reindex(
        index=range(Kw), columns=range(Kw), fill_value=0).values.tolist() for i in range(len(names) - 1)}
    stay = {k: float(np.mean(wlab[names[i]] == wlab[names[i + 1]])) for i, k in enumerate(trans)}
    W = np.stack([wlab[n] for n in names], 1)
    core = np.stack([with_type[n] for n in names], 1).all(1)
    changes = (W[:, 1:] != W[:, :-1]).sum(1)

    # 6. кластеры изменений
    F23 = window_features(P, d["Z"], d["N"], list(range(0, 12))).loc[ids]
    F24 = window_features(P, d["Z"], d["N"], list(range(12, 24))).loc[ids]
    dF = F24[feats] - F23[feats]
    dlab = kmeans_attr(robust_z(dF, cfg["clip"]).to_numpy(), 4, seed)
    dprof = dF.groupby(dlab).mean()

    T["delta_type"] = dlab
    T["window_changes"] = changes
    T["core"] = core
    for n_ in names:
        T[f"type_{n_}"] = wlab[n_]
    T.to_csv(out / "types.csv", encoding="utf-8")
    res = {"method": {k: (None if v != v else v) for k, v in best.items()}, "K": int(K), "n": int(len(ids)),
           "sizes": {int(k): int(v) for k, v in sizes.items()},
           "profile": prof.round(4).to_dict(orient="index"), "mirkin": mirkin.round(3).to_dict(orient="index"),
           "regions": regions, "ami_regions": float(ami_reg), "validation": val.round(4).to_dict(orient="index"),
           "windows": names, "window_ami": wami, "cohesion": cohesion, "transitions": trans, "stay_share": stay,
           "core_share": float(core.mean()), "core_by_type": T.groupby("type").core.mean().round(3).to_dict(),
           "delta_profile": dprof.round(3).to_dict(orient="index"), "delta_sizes": np.bincount(dlab).tolist(),
           "delta_by_type": pd.crosstab(T.type, T.delta_type).values.tolist(),
           "confidence_mean": float(conf.mean()), "confidence_low_share": float((conf < 0.6).mean())}
    json.dump(res, open(out / "final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    pd.set_option("display.width", 220)
    print("размеры:", sizes.to_dict(), "AMI с субъектами:", round(ami_reg, 3))
    print("уверенность: средняя", round(conf.mean(), 3), "доля МО < 0.6:", round((conf < 0.6).mean(), 3))
    print(mirkin.round(2).to_string())
    print(val.round(3).to_string())
    print("сохранили тип между соседними окнами:", {k: round(v, 3) for k, v in stay.items()})
    print("устойчивое ядро:", round(core.mean(), 3), res["core_by_type"])
    print("кластеры изменений 2023 -> 2024:", np.bincount(dlab).tolist())
    print(dprof.round(2).to_string())


if __name__ == "__main__":
    main()
