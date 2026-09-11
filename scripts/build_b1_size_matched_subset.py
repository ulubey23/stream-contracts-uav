"""
Build a random RGB training subset with the SAME cardinality as strict paired P1 (2070 images)
to disentangle "subset size / sampling" from "strict pairing protocol".

Uses DBEV RGB train images and labels; val/test stay the official full RGB splits.

  python scripts/build_b1_size_matched_subset.py --n 2070 --seed 42

Writes:
  data/derived/b1_rgb_train2070_seed42/train/{images,labels}
  configs/generated/dbev_rgb_train2070_control.yaml

Then train (example):
  python scripts/run_yolo.py train --data configs/generated/dbev_rgb_train2070_control.yaml \\
      --name train_rgb_train2070_control --epochs 100 --batch 8 --seed 42 --paper-snapshot
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import rgb_dataset_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2070, help="Must match P1 train count for fair control.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "data" / "derived" / "b1_rgb_train2070_seed42",
    )
    args = ap.parse_args()

    rgb_root = rgb_dataset_root()
    img_dir = rgb_root / "train" / "images"
    lb_dir = rgb_root / "train" / "labels"
    imgs = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(imgs) < args.n:
        raise SystemExit(f"Need at least {args.n} train images, found {len(imgs)}")

    rng = random.Random(args.seed)
    chosen = rng.sample(imgs, args.n)

    out_train_img = args.out / "train" / "images"
    out_train_lb = args.out / "train" / "labels"
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lb.mkdir(parents=True, exist_ok=True)

    for src in chosen:
        shutil.copy2(src, out_train_img / src.name)
        src_lb = lb_dir / f"{src.stem}.txt"
        if not src_lb.is_file():
            raise FileNotFoundError(src_lb)
        shutil.copy2(src_lb, out_train_lb / f"{src.stem}.txt")

    val_imgs = rgb_root / "valid" / "images"
    test_imgs = rgb_root / "test" / "images"
    cfg_dir = _ROOT / "configs" / "generated"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfg_dir / "dbev_rgb_train2070_control.yaml"
    payload = {
        "train": str((out_train_img).resolve()).replace("\\", "/"),
        "val": str(val_imgs.resolve()).replace("\\", "/"),
        "test": str(test_imgs.resolve()).replace("\\", "/"),
        "nc": 1,
        "names": ["drone"],
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Copied {len(chosen)} train pairs -> {args.out / 'train'}")
    print("Wrote", yaml_path)
    print("Val/test:", val_imgs)
    print("Test:", test_imgs)


if __name__ == "__main__":
    main()
