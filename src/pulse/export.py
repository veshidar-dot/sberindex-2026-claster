"""Данные для лендинга: всё в один JSON внутри landing/index.html (страница открывается и с диска, и с GitHub Pages).

    python -m pulse.export
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import prepare
from .config import load_config

NAMES = {   # имена типов по их портретам (out/final.json) и внешней проверке (out/external.json)
    0: "Бюджетные районы Сибири и Севера",
    1: "Поволжье и Приуралье",
    2: "Промышленные города Урала и Сибири",
    3: "Растущий пояс Подмосковья и юга",
    4: "Москва",
    5: "Петербург",
}


def main():
    cfg = load_config()
    out = Path(cfg["out"])
    T = pd.read_csv(out / "types.csv", index_col=0)
    r = json.load(open(out / "final.json", encoding="utf-8"))
    grid = pd.read_csv(out / "compare_grid.csv")
    best = pd.read_csv(out / "compare_best.csv")
    d = prepare(cfg)
    P = d["P"]
    ids = P.ids[d["full"]]
    Z = d["Z"][d["full"]][:, 0, :]                       # собственное отклонение «Всех категорий», лог
    T = T.reindex(ids)
    order = np.lexsort((-T.confidence.to_numpy(), T.type.to_numpy()))
    seis = np.round((Z[order] - np.nanmean(Z[order], 1, keepdims=True)) * 1000).astype(int)
    type_curves = {int(k): np.round(np.nanmean(Z[T.type.to_numpy() == k], 0) * 100, 2).tolist() for k in range(r["K"])}
    wcols = [c for c in T.columns if c.startswith("type_")]
    mo = pd.DataFrame({"id": ids, "n": T.name, "r": T.region, "t": T.type, "c": T.confidence.round(2),
                       "lat": T.lat.round(3), "lon": T.lon.round(3), "dt": T.delta_type, "core": T.core.astype(int),
                       "w": T[wcols].astype(int).values.tolist()})
    for c in ("beta", "season", "drift", "mkt_speed", "noise"):
        mo[c] = T[c].round(3)
    data = {
        "names": NAMES, "months": P.months, "final": r,
        "seismo": {"order_types": T.type.to_numpy()[order].tolist(), "z": seis.tolist()},
        "type_curves": type_curves,
        "mo": mo.where(mo.notna(), None).to_dict(orient="records"),
        "grid": grid[["method", "K", "alpha", "res", "SW", "CH_per_N", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "AVI_road", "borda"]]
            .round(4).where(grid.notna(), None).to_dict(orient="records"),
        "best": best.round(4).where(best.notna(), None).to_dict(orient="records"),
        "threshold": d["thr"], "edges": {"comove": int(d["A_co"].nnz // 2), "road": int(d["A_road"].nnz // 2), "attr": int(d["A_attr"].nnz // 2)},
    }
    ext = out / "external.json"
    if ext.exists():
        data["external"] = json.loads(ext.read_text(encoding="utf-8"))
    tpl = Path("landing/template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=float).replace("</", "<\\/"))
    Path("landing/index.html").write_text(html, encoding="utf-8")
    print("landing/index.html", Path("landing/index.html").stat().st_size // 1024, "КБ")


if __name__ == "__main__":
    main()
