"""config.yaml с наложением config.local.yaml (локальные пути, не в git)."""
from pathlib import Path

import yaml


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path="config.yaml"):
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    local = p.with_name("config.local.yaml")
    if local.exists():
        _merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    Path(cfg["out"]).mkdir(parents=True, exist_ok=True)
    return cfg
