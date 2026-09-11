"""
Compute how many depth images in each split have at least one RGB image whose Roboflow-style
filename starts with the same numeric id (e.g. depth `1050.jpg` <-> `1050_jpg.rf.*.jpg`).

This is a reproducible pairing rule for DBEV-UAV exports; it is not guaranteed to be 100% by the
dataset authors — always report the match rates alongside fusion results.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import depth_dataset_root, rgb_dataset_root


def _rgb_index(rgb_images_dir: Path) -> dict[int, list[str]]:
    by_id: dict[int, list[str]] = defaultdict(list)
    for p in rgb_images_dir.glob("*.jpg"):
        stem = p.stem
        if "_" not in stem:
            continue
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            by_id[int(prefix)].append(str(p.resolve()))
    for k in by_id:
        by_id[k].sort()
    return by_id


def _same_size(rgb_path: str, depth_path: str) -> bool:
    with Image.open(rgb_path) as r, Image.open(depth_path) as d:
        return r.size == d.size


def stats_for_split(rgb_root: Path, depth_root: Path, split: str) -> dict:
    rgb_dir = rgb_root / split / "images"
    depth_dir = depth_root / split / "images"
    by_id = _rgb_index(rgb_dir)
    depth_files = sorted(depth_dir.glob("*.jpg"))
    matched = 0
    multi = 0
    pairs: list[dict] = []
    for dp in depth_files:
        did = int(dp.stem)
        cands = by_id.get(did)
        if not cands:
            continue
        matched += 1
        if len(cands) > 1:
            multi += 1
        rgb_path = cands[0]
        pairs.append({"id": did, "depth": str(dp.resolve()), "rgb": rgb_path, "rgb_candidates": len(cands)})
    pairs_strict = [p for p in pairs if _same_size(p["rgb"], p["depth"])]
    return {
        "split": split,
        "depth_images": len(depth_files),
        "matched": matched,
        "match_rate": (matched / len(depth_files)) if depth_files else 0.0,
        "multi_candidate_pairs": multi,
        "matched_same_size": len(pairs_strict),
        "same_size_rate": (len(pairs_strict) / len(depth_files)) if depth_files else 0.0,
        "pairs": pairs,
        "pairs_strict": pairs_strict,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_ROOT / "results" / "pairing_stats.json")
    args = ap.parse_args()

    rgb_root = rgb_dataset_root()
    depth_root = depth_dataset_root()
    report = {"rgb_root": str(rgb_root), "depth_root": str(depth_root), "splits": {}}
    report_strict = {"rgb_root": str(rgb_root), "depth_root": str(depth_root), "splits": {}}
    for split in ("train", "valid", "test"):
        full = stats_for_split(rgb_root, depth_root, split)
        summary = {k: v for k, v in full.items() if k not in ("pairs", "pairs_strict")}
        report["splits"][split] = summary
        pair_path = args.out.with_name(f"pairs_{split}.json")
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        pair_path.write_text(
            json.dumps({k: v for k, v in full.items() if k != "pairs_strict"}, indent=2),
            encoding="utf-8",
        )
        strict_payload = {
            "split": full["split"],
            "depth_images": full["depth_images"],
            "matched": full["matched_same_size"],
            "match_rate": full["same_size_rate"],
            "pairs": full["pairs_strict"],
        }
        pair_path.with_name(f"pairs_{split}_strict.json").write_text(json.dumps(strict_payload, indent=2), encoding="utf-8")
        report_strict["splits"][split] = {k: v for k, v in strict_payload.items() if k != "pairs"}
        print(
            f"{split}: depth={full['depth_images']} matched={full['matched']} ({full['match_rate']:.3f}) "
            f"same_size={full['matched_same_size']} ({full['same_size_rate']:.3f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    strict_out = args.out.with_name("pairing_stats_strict.json")
    strict_out.write_text(json.dumps(report_strict, indent=2), encoding="utf-8")
    print("Wrote summary:", args.out)
    print("Wrote strict (same WxH) summary:", strict_out)


if __name__ == "__main__":
    main()
