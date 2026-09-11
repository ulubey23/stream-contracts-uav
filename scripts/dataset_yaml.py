"""Resolve YOLO data.yaml paths (Ultralytics layout)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_data_yaml(yaml_path: Path) -> dict[str, Any]:
    with Path(yaml_path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_root(yaml_path: Path) -> Path:
    y = load_data_yaml(yaml_path)
    return Path(y["path"]).resolve()


def split_image_dir(yaml_path: Path, split_key: str) -> Path:
    y = load_data_yaml(yaml_path)
    root = Path(y["path"]).resolve()
    rel = y.get(split_key) or y.get("val")
    if not rel:
        raise KeyError(split_key)
    return (root / rel).resolve()


def write_minimal_yaml(
    out: Path,
    *,
    path: Path,
    train_glob: str = "train/images",
    val_glob: str = "valid/images",
    test_glob: str | None = "test/images",
    nc: int = 1,
    names: list[str] | None = None,
) -> None:
    names = names or ["drone"]
    d: dict[str, Any] = {
        "path": str(path.resolve()).replace("\\", "/"),
        "train": train_glob,
        "val": val_glob,
        "nc": nc,
        "names": names,
    }
    if test_glob:
        d["test"] = test_glob
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
