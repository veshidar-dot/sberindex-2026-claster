"""Скачивает все исходные данные в data/ (в git данные не хранятся).

    python scripts/download_data.py            # всё, включая Росстат (около 1,3 ГБ архива, из него берутся 2 файла)
    python scripts/download_data.py --no-rosstat

Источники и лицензии - в README («Данные»). Если сетевой адрес изменится, файлы можно положить вручную
по путям ниже.
"""
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, REF = ROOT / "data/raw", ROOT / "data/ref"

HACKATHON = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
# справочник МО СберИндекса (центры, субъекты, ОКТМО, история) - копия в открытом репозитории участника (MIT)
MO_DICT = "https://raw.githubusercontent.com/stasvinokur/konkurs_sberindex_2026/06741e1317056a1ffe01fae681b87c659794506d/data/reference/sberindex_municipal_districts.csv"
# упрощённые границы МО (адаптация набора СберИндекса о границах МО, CC BY-SA 4.0)
GEO = "https://raw.githubusercontent.com/elasesinodetupaz/sberindex-mo-clustering/9e5614b8376371bd507d68da0f083e92f12f5007/data/geo/mo_national.geojson"
ROSSTAT = "https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/by_indicator/data_section32_112_v20250918.zip"
ROSSTAT_FILES = ["data_Y48423005_parts/data_Y48423005_year2023_112_v20250918.csv",
                 "data_Y48423007_parts/data_Y48423007_year2023_112_v20250918.csv"]


def get(url, dest=None):
    print("скачиваю", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def mobility():
    """Индекс мобильности СберИндекса по СЗФО: через публичный API сайта (POST /api/sowa). Если API недоступно,
    файл data/raw/mobility_szfo.csv (столбцы mo_name, period, km) можно выгрузить с дашборда вручную."""
    import base64
    import uuid
    from datetime import datetime, timedelta, timezone
    route = "/dataset/v1/indeks-mobilnosti?filter=%7B%22SOURCE%22%3A%22SI%22%7D&limit=1000&offset=0"
    body = {"SOWA": {"method": "GET", "route": base64.b64encode(route.encode()).decode(), "data": {"type": "object", "value": []}}}
    req = urllib.request.Request("https://sberindex.ru/api/sowa", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "RqUID": uuid.uuid4().hex, "X-Language": "ru"})
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.loads(r.read())

    def dec(e, t=None):
        if isinstance(e, list):
            return [dec(x) for x in e]
        if isinstance(e, dict) and e.get("type") == "object":
            return {x["key"]: dec(x["value"], x["type"]) for x in e["value"]}
        s = str(e)
        if s.startswith("__"):
            ty, v = s[2:].split("__", 1)
            ty = t or ty
            return float(v) if ty == "number" else base64.b64decode(v).decode() if ty == "string" else v
        return e
    d = dec(j["SOWA"]["data"])
    f = d["fields"]
    rows = []
    for r in d["data"]:
        rec = dict(zip(f, r))
        t = datetime.fromisoformat(rec["period"].replace("Z", "+00:00")) + timedelta(hours=3)
        rows.append((rec["ref_area"], t.strftime("%Y-%m"), rec["value"]))
    out = RAW / "mobility_szfo.csv"
    out.write_text("mo_name,period,km\n" + "".join(f'"{a}",{b},{c}\n' for a, b, c in rows), encoding="utf-8")
    print(out, len(rows), "строк")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rosstat", action="store_true")
    a = ap.parse_args()
    z = zipfile.ZipFile(io.BytesIO(get(HACKATHON)))
    for name in ("consumption.parquet", "connection.parquet", "market_access.parquet"):
        member = next(n for n in z.namelist() if n.endswith(name))
        (RAW / name).parent.mkdir(parents=True, exist_ok=True)
        (RAW / name).write_bytes(z.read(member))
    import pandas as pd
    hist = pd.read_csv(io.BytesIO(get(MO_DICT)))
    REF.mkdir(parents=True, exist_ok=True)
    hist.to_csv(REF / "mo_dict_history.csv", index=False, encoding="utf-8")
    last = hist.sort_values(["territory_id", "year_to", "year_from"]).groupby("territory_id").tail(1)
    last[["territory_id", "oktmo", "municipal_district_name", "municipal_district_name_short", "municipal_district_type",
          "municipal_district_center", "region_code", "region_name", "municipal_district_center_lat",
          "municipal_district_center_lon", "year_from", "year_to"]].to_csv(REF / "mo_dict_full.csv", index=False, encoding="utf-8")
    get(GEO, RAW / "mo_national.geojson")
    try:
        mobility()
    except Exception as e:                       # noqa: BLE001 - сайт может закрыть API; это вспомогательный источник
        print("индекс мобильности не скачан:", e)
    if not a.no_rosstat:
        dest = RAW / "rosstat"
        dest.mkdir(parents=True, exist_ok=True)
        zr = zipfile.ZipFile(io.BytesIO(get(ROSSTAT)))
        for m in ROSSTAT_FILES:
            (dest / Path(m).name).write_bytes(zr.read(m))
    print("готово")


if __name__ == "__main__":
    main()
