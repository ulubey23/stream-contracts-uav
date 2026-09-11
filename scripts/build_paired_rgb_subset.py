"""
Materialize a YOLO layout for RGB images that are strictly paired with depth (same numeric id)
and have identical (width, height) as the depth frame.

Labels MUST come from the RGB split (same stem as the chosen RGB image), not from depth.
Depth and RGB BEV frames can share an id and size while bounding boxes differ in normalized
coordinates; using depth labels on RGB images mis-supervises training (we observed ~0 mAP).

Run after: python scripts/pairing_stats.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import rgb_dataset_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_ROOT / "data" / "derived" / "paired_rgb_strict",
        help="YOLO root: <out-root>/{train,valid,test}/{images,labels}/",
    )
    ap.add_argument(
        "--pairs-dir",
        type=Path,
        default=_ROOT / "results",
        help="Directory containing pairs_{split}_strict.json",
    )
    args = ap.parse_args()

    rgb_root = rgb_dataset_root()
    for split in ("train", "valid", "test"):
        pj = args.pairs_dir / f"pairs_{split}_strict.json"
        if not pj.is_file():
            raise SystemExit(f"Missing {pj}; run scripts/pairing_stats.py first.")
        payload = json.loads(pj.read_text(encoding="utf-8"))
        pairs = payload["pairs"]
        img_out = args.out_root / split / "images"
        lb_out = args.out_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lb_out.mkdir(parents=True, exist_ok=True)
        for p in pairs:
            did = p["id"]
            src_rgb = Path(p["rgb"])
            dst_img = img_out / f"{did}.jpg"
            shutil.copy2(src_rgb, dst_img)
            src_lb = rgb_root / split / "labels" / f"{src_rgb.stem}.txt"
            dst_lb = lb_out / f"{did}.txt"
            if not src_lb.is_file():
                raise FileNotFoundError(src_lb)
            shutil.copy2(src_lb, dst_lb)
        print(f"{split}: wrote {len(pairs)} image/label pairs -> {args.out_root / split}")

    print("Done. Regenerate configs: python scripts/generate_yolo_configs.py")


if __name__ == "__main__":
    main()
