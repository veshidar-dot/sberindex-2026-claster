"""Общие загрузчики данных для обеих работ по кластеризации.

Все пути - относительно корня clustering/ (data/raw, data/ref); каждая работа передаёт свой корень данных
из config.yaml. Данные:
  consumption.parquet, connection.parquet, market_access.parquet - пакет хакатона СберИндекса (CC BY-SA 4.0);
  mo_dict_full.csv - справочник МО СберИндекса (территория, ОКТМО, центр, субъект, координаты; CC BY-SA 4.0);
  mobility_szfo.csv - индекс мобильности СберИндекса по СЗФО (средняя дальность покупок, км; CC BY-SA 4.0);
  mo_national.geojson - упрощённые границы МО из набора СберИндекса о границах МО (CC BY-SA 4.0);
  rosstat/*.zip - БД ПМО Росстата в обработке tochno.st (CC BY 4.0).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["Продовольствие", "Маркетплейсы", "Общественное питание", "Здоровье", "Транспорт"]
TOTAL = "Все категории"


@dataclass
class Panel:
    ids: np.ndarray            # territory_id, по возрастанию
    months: list               # ['2023-01', ...]
    V: np.ndarray              # (МО, категория [Все + 5], месяц), руб. на жителя; NaN - нет оценки
    cats: list


def load_panel(root):
    c = pd.read_parquet(Path(root) / "data/raw/consumption.parquet")
    c["month"] = pd.to_datetime(c.date).dt.strftime("%Y-%m")
    cats = [TOTAL] + CATEGORIES
    w = c.pivot_table(index="territory_id", columns=["category", "month"], values="value", aggfunc="first")
    ids = np.array(sorted(w.index))
    months = sorted(c.month.unique())
    V = np.full((len(ids), len(cats), len(months)), np.nan)
    for k, cat in enumerate(cats):
        if cat in w.columns.get_level_values(0):
            V[:, k, :] = w[cat].reindex(index=ids, columns=months).to_numpy(float)
    V[V <= 0] = np.nan
    return Panel(ids=ids, months=months, V=V, cats=cats)


def load_dict(root):
    return pd.read_csv(Path(root) / "data/ref/mo_dict_full.csv")


def road_matrix(root, ids, kind="highway"):
    """Плотная матрица расстояний по дорогам (км) между МО ids; NaN - нет связи. Таблица хакатона хранит
    пару в одну сторону, расстояние считается симметричным."""
    c = pd.read_parquet(Path(root) / "data/raw/connection.parquet")
    c = c[c["type"] == kind]
    pos = {t: i for i, t in enumerate(ids)}
    D = np.full((len(ids), len(ids)), np.nan)
    a = c.territory_id_x.map(pos)
    b = c.territory_id_y.map(pos)
    ok = a.notna() & b.notna()
    a, b, v = a[ok].astype(int).to_numpy(), b[ok].astype(int).to_numpy(), c.distance[ok].to_numpy(float)
    D[a, b] = v
    D[b, a] = np.where(np.isnan(D[b, a]), v, np.fmin(D[b, a], v))
    np.fill_diagonal(D, 0.0)
    return D


def market_access(root, ids):
    m = pd.read_parquet(Path(root) / "data/raw/market_access.parquet").set_index("territory_id").market_access
    return m.reindex(ids).to_numpy(float)


def load_mobility(root):
    """Индекс мобильности СЗФО с territory_id (сопоставление по полному названию МО справочника)."""
    m = pd.read_csv(Path(root) / "data/raw/mobility_szfo.csv")
    d = load_dict(root)
    name = d.drop_duplicates("municipal_district_name", keep="last").set_index("municipal_district_name").territory_id
    m["territory_id"] = m.mo_name.map(name)
    return m


def load_geo(root):
    import geopandas as gpd
    return gpd.read_file(Path(root) / "data/raw/mo_national.geojson")
