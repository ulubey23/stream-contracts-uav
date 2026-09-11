"""
Write Ultralytics YOLO `data.yaml` files with absolute `path:` for Windows + Unicode folders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import depth_dataset_root, rgb_dataset_root


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_yaml(out_path: Path, dataset_root: Path, names: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics expects path + train/val/test relative to path
    payload = {
        "path": str(dataset_root).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "generated",
    )
    args = ap.parse_args()

    rgb_root = rgb_dataset_root()
    dep_root = depth_dataset_root()
    paired_root = _repo_root() / "data" / "derived" / "paired_rgb_strict"

    _write_yaml(args.out_dir / "dbev_rgb.yaml", rgb_root, names=["drone"])
    _write_yaml(args.out_dir / "dbev_depth.yaml", dep_root, names=["Drone"])
    if paired_root.is_dir() and (paired_root / "train" / "images").is_dir():
        _write_yaml(args.out_dir / "dbev_rgb_paired_strict.yaml", paired_root.resolve(), names=["drone"])
        wrote_paired = True
    else:
        wrote_paired = False

    print("Wrote:")
    print(" ", args.out_dir / "dbev_rgb.yaml")
    print(" ", args.out_dir / "dbev_depth.yaml")
    if wrote_paired:
        print(" ", args.out_dir / "dbev_rgb_paired_strict.yaml")
    else:
        print(" (skip) dbev_rgb_paired_strict.yaml — run scripts/build_paired_rgb_subset.py first")
    print("RGB root:", rgb_root)
    print("Depth root:", dep_root)
    if wrote_paired:
        print("Paired RGB (strict) root:", paired_root.resolve())


if __name__ == "__main__":
    main()
