"""Внешняя проверка типов данными, которых не было в кластеризации: Росстат (БД ПМО в обработке tochno.st, CC BY 4.0)
и индекс мобильности СберИндекса по СЗФО.

    python -m pulse.external

Росстат, 2023 г., МО верхнего уровня, январь-декабрь: среднемесячная зарплата (Y48423007, «Всего») и численность
работников по разделам ОКВЭД2 (Y48423005). ОКТМО сопоставляется с territory_id по истории справочника МО СберИндекса.
Меры: η² - доля дисперсии лога зарплаты, объяснённая типами, против той же доли для субъектов и для случайного
разбиения с теми же размерами групп; отраслевой профиль типов (доля работников в сельском хозяйстве, добыче,
обрабатывающих производствах, торговле, госуправлении, образовании и здравоохранении; раздел, которого нет
в таблице МО, считается нулём - доли являются оценкой снизу).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

SECTORS = {"A": "сельское хозяйство", "B": "добыча", "C": "обработка", "G": "торговля", "O": "госуправление",
           "P": "образование", "Q": "здравоохранение"}


def read(path, code):
    cols = ["okved2", "mun_level", "oktmo", "year", "indicator_value", "indicator_period"]
    d = pd.read_csv(path, sep=";", usecols=cols, dtype=str)
    d = d[d.mun_level.str.contains("верхнего", na=False) & (d.indicator_period == "Январь-декабрь") & (d.year == "2023")]
    d["v"] = pd.to_numeric(d.indicator_value.str.replace(",", "."), errors="coerce")
    return d


def rub(v):
    return f"{v:,.0f}".replace(",", " ") + " ₽"


def pct(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def eta2(v, g):
    ok = np.isfinite(v)
    v, g = v[ok], np.asarray(g)[ok]
    m = v.mean()
    s = pd.Series(v).groupby(g)
    return float((s.count() * (s.mean() - m) ** 2).sum() / ((v - m) ** 2).sum())


def main():
    cfg = load_config()
    out = Path(cfg["out"])
    T = pd.read_csv(out / "types.csv", index_col=0)
    hist = pd.read_csv(Path(cfg["data_root"]) / "data/ref/mo_dict_history.csv", dtype={"oktmo": str})
    hist["ok8"] = hist.oktmo.str.replace("-", "").str[:8]
    o2t = hist.drop_duplicates("ok8", keep="last").set_index("ok8").territory_id
    raw = Path(cfg["data_root"]) / "data/raw/rosstat"
    w = read(raw / "data_Y48423007_year2023_112_v20250918.csv", "Y48423007")
    w = w[w.okved2.str.startswith("Всего", na=False)]
    wage = w.assign(tid=w.oktmo.map(o2t)).dropna(subset=["tid"]).groupby("tid").v.max()
    e = read(raw / "data_Y48423005_year2023_112_v20250918.csv", "Y48423005")
    e["tid"] = e.oktmo.map(o2t)
    e = e.dropna(subset=["tid"])
    tot = e[e.okved2.str.startswith("Всего", na=False)].groupby("tid").v.max()
    sec = e[e.okved2.str.startswith("Раздел", na=False)].copy()
    sec["s"] = sec.okved2.str.split().str[1].str.translate(str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX"))   # кириллица в кодах разделов
    shares = sec[sec.s.isin(list(SECTORS))].pivot_table(index="tid", columns="s", values="v", aggfunc="max")
    # раздела нет в таблице МО (нет работников или значение скрыто как конфиденциальное) - считаем нулём: доли - оценка снизу
    shares = shares.reindex(tot.index).fillna(0.0).div(tot, axis=0)
    T = T.join(np.log(wage).rename("log_wage")).join(shares)
    m = T.log_wage.notna()
    rng = np.random.default_rng(cfg["seed"])
    rand = [eta2(T.log_wage[m].to_numpy(), rng.permutation(T.type[m].to_numpy())) for _ in range(200)]
    res = {"n_wage": int(m.sum()),
           "eta2_types": eta2(T.log_wage[m].to_numpy(), T.type[m]),
           "eta2_regions": eta2(T.log_wage[m].to_numpy(), T.region[m]),
           "eta2_random": float(np.mean(rand)),
           "wage_by_type": np.exp(T[m].groupby("type").log_wage.mean()).round(0).to_dict(),
           "sectors_by_type": (T.groupby("type")[[s for s in SECTORS if s in T]].mean() * 100).round(1).to_dict(orient="index"),
           "sector_names": SECTORS}
    # мобильность СЗФО
    mob = pd.read_csv(Path(cfg["data_root"]) / "data/raw/mobility_szfo.csv")
    hd = hist.drop_duplicates("municipal_district_name", keep="last").set_index("municipal_district_name").territory_id
    mob["tid"] = mob.mo_name.map(hd)
    mob = mob[mob.period == "2024-12"].dropna(subset=["tid"]).set_index("tid").km
    T = T.join(mob.rename("mobility_km"))
    mm = T.mobility_km.notna()
    res["mobility_by_type"] = T[mm].groupby("type").mobility_km.agg(["count", "median"]).round(2).to_dict(orient="index")
    names = {0: "бюджетных районов Сибири и Севера", 1: "Поволжья и Приуралья", 2: "промышленных городов Урала", 3: "пояса Подмосковья", 4: "Москвы", 5: "Петербурга"}
    wb = res["wage_by_type"]
    sb = res["sectors_by_type"]
    res["lines"] = [
        f"Зарплата Росстата (2023 г., {rub(res['n_wage'])[:-2]} МО), которой не было среди признаков: типы объясняют {res['eta2_types'] * 100:.0f} % дисперсии её логарифма "
        f"(случайное разбиение с теми же размерами - {pct(res['eta2_random'] * 100)} %, субъекты - {res['eta2_regions'] * 100:.0f} %). "
        f"Средняя зарплата от {rub(wb[min(wb, key=wb.get)])} у {names[int(min(wb, key=wb.get))]} до {rub(wb[max(wb, key=wb.get)])} у {names[int(max(wb, key=wb.get))]}.",
        "Отраслевой профиль Росстата (доля работников организаций, оценка снизу): у бюджетных районов госуправление, образование и "
        f"здравоохранение - {sb[0]['O'] + sb[0]['P'] + sb[0]['Q']:.0f} %, обработка - {sb[0]['C']:.0f} %; у промышленных городов Урала обработка "
        f"{sb[2]['C']:.0f} % и добыча {sb[2]['B']:.0f} %; у пояса Подмосковья обработка {sb[3]['C']:.0f} %, торговля {sb[3]['G']:.0f} %.",
    ]
    (out / "external.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "lines"}, ensure_ascii=False, indent=1, default=float))
    print("\n".join(res["lines"]))


if __name__ == "__main__":
    main()
