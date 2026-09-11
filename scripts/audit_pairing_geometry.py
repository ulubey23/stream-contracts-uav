"""
Pairing audit: identity collisions, multi-candidate matches, cross-split ID overlap,
augmentation-suffix structure, and optional pixel similarity on a sample of strict pairs
(cheap proxy for scene correspondence, not full registration).

  python scripts/audit_pairing_geometry.py
  python scripts/audit_pairing_geometry.py --pixel-sample 50

Writes:
  results/exports/pairing_audit.json
  results/exports/pairing_audit_summary.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import depth_dataset_root, rgb_dataset_root  # noqa: E402

_ID_RE = re.compile(r"^(\d+)")
_AUG_RE = re.compile(r"_(aug|jpg\.rf\.|[a-f0-9]{8,})", re.I)


def _stem_id(stem: str) -> int | None:
    m = _ID_RE.match(stem)
    return int(m.group(1)) if m else None


def _list_images(split_dir: Path) -> list[Path]:
    img = split_dir / "images"
    if not img.is_dir():
        return []
    return sorted(
        p for p in img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # W, H


def _audit_split(rgb_root: Path, depth_root: Path, split: str) -> dict:
    rgb_files = _list_images(rgb_root / split)
    dep_files = _list_images(depth_root / split)

    rgb_by_id: dict[int, list[Path]] = defaultdict(list)
    for p in rgb_files:
        i = _stem_id(p.stem)
        if i is not None:
            rgb_by_id[i].append(p)

    dep_ids = []
    multi = 0
    matched = 0
    size_fail = 0
    collisions = 0  # >1 same-size RGB candidates
    chosen_lex = 0
    pairs = []
    aug_suffix_count = 0

    for dp in dep_files:
        did = _stem_id(dp.stem)
        if did is None:
            continue
        dep_ids.append(did)
        cands = rgb_by_id.get(did, [])
        if not cands:
            continue
        if len(cands) > 1:
            multi += 1
        try:
            dw, dh = _size(dp)
        except Exception:
            continue
        same = []
        for rp in cands:
            try:
                rw, rh = _size(rp)
            except Exception:
                continue
            if (rw, rh) == (dw, dh):
                same.append(rp)
        if not same:
            size_fail += 1
            continue
        matched += 1
        if len(same) > 1:
            collisions += 1
        same_sorted = sorted(same, key=lambda p: p.name)
        pick = same_sorted[0]
        if len(same_sorted) > 1 and pick == same_sorted[0]:
            chosen_lex += 1
        if _AUG_RE.search(pick.stem) or "." in pick.stem.replace(str(did), "", 1):
            # Roboflow-style stems often contain long hashes after the id
            if len(pick.stem) > len(str(did)) + 2:
                aug_suffix_count += 1
        pairs.append(
            {
                "id": did,
                "depth": str(dp),
                "rgb": str(pick),
                "n_rgb_same_id": len(cands),
                "n_same_size": len(same),
            }
        )

    id_counts = Counter(dep_ids)
    duplicate_depth_ids = sum(1 for _, c in id_counts.items() if c > 1)

    return {
        "split": split,
        "n_rgb": len(rgb_files),
        "n_depth": len(dep_files),
        "unique_depth_ids": len(id_counts),
        "duplicate_depth_id_rows": duplicate_depth_ids,
        "matched_same_size": matched,
        "coverage": (matched / len(dep_files)) if dep_files else 0.0,
        "multi_candidate_same_id": multi,
        "same_id_but_size_mismatch": size_fail,
        "collisions_multiple_same_size": collisions,
        "lexicographic_selection_events": chosen_lex,
        "matched_rgb_with_long_stem_suffix": aug_suffix_count,
        "depth_ids": sorted(id_counts.keys()),
        "pairs_sample": pairs[:5],
        "pairs": pairs,
    }


def _cross_split_overlap(reports: dict[str, dict]) -> dict:
    sets = {s: set(reports[s]["depth_ids"]) for s in reports}
    out = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sets[a] & sets[b]
            out[f"{a}_vs_{b}"] = {
                "n_overlap_ids": len(inter),
                "overlap_ids_sample": sorted(inter)[:20],
            }
    return out


def _pixel_corr_sample(pairs: list[dict], n: int, seed: int = 42) -> dict:
    """Cheap proxy: resize both to 64x64 grayscale and report Pearson corr of pixels."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    if not pairs:
        return {"n": 0, "mean_corr": None, "std_corr": None}
    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    corrs = []
    for i in idx:
        rp = Path(pairs[i]["rgb"])
        dp = Path(pairs[i]["depth"])
        try:
            r = np.asarray(Image.open(rp).convert("L").resize((64, 64)), dtype=np.float32).ravel()
            d = np.asarray(Image.open(dp).convert("L").resize((64, 64)), dtype=np.float32).ravel()
            if r.std() < 1e-6 or d.std() < 1e-6:
                continue
            c = float(np.corrcoef(r, d)[0, 1])
            if np.isfinite(c):
                corrs.append(c)
        except Exception:
            continue
    if not corrs:
        return {"n": 0, "mean_corr": None, "std_corr": None, "note": "no valid samples"}
    arr = np.asarray(corrs, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean_corr": float(arr.mean()),
        "std_corr": float(arr.std()),
        "min_corr": float(arr.min()),
        "max_corr": float(arr.max()),
        "note": "Proxy only: grayscale 64x64 correlation; not a geometric registration error.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel-sample", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "results" / "exports")
    args = ap.parse_args()

    rgb_root = rgb_dataset_root()
    depth_root = depth_dataset_root()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for split in ("train", "valid", "test"):
        print("Auditing", split, "...")
        reports[split] = _audit_split(rgb_root, depth_root, split)

    cross = _cross_split_overlap(reports)
    pixel = {}
    if args.pixel_sample > 0:
        # Prefer test pairs for the proxy (held-out)
        pixel = _pixel_corr_sample(reports["test"]["pairs"], args.pixel_sample)

    # Strip bulky pair lists from summary JSON (keep counts); full pairs in separate files
    slim = {}
    for s, r in reports.items():
        slim[s] = {k: v for k, v in r.items() if k not in ("pairs", "depth_ids")}
        (args.out_dir / f"pairs_{s}_strict_audit.json").write_text(
            json.dumps({"split": s, "pairs": r["pairs"]}, indent=2), encoding="utf-8"
        )

    payload = {
        "rgb_root": str(rgb_root),
        "depth_root": str(depth_root),
        "splits": slim,
        "cross_split_depth_id_overlap": cross,
        "pixel_correlation_proxy_test": pixel,
        "terminology": {
            "pairing_rule": "identity token + equal (W,H) + lexicographic RGB selection",
            "establishes": "filename/identity-level correspondence within a split",
            "does_not_guarantee": "pixel-level geometric registration after RGB augmentation",
        },
    }
    out_json = args.out_dir / "pairing_audit.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "Pairing audit summary",
        f"RGB root: {rgb_root}",
        f"Depth root: {depth_root}",
        "",
    ]
    for s, r in slim.items():
        lines.append(
            f"[{s}] depth={r['n_depth']} rgb={r['n_rgb']} matched={r['matched_same_size']} "
            f"cov={r['coverage']:.3f} multi_id={r['multi_candidate_same_id']} "
            f"collisions={r['collisions_multiple_same_size']} size_fail={r['same_id_but_size_mismatch']}"
        )
    lines.append("")
    lines.append("Cross-split depth ID overlap:")
    for k, v in cross.items():
        lines.append(f"  {k}: {v['n_overlap_ids']}")
    if pixel:
        lines.append("")
        lines.append(
            f"Pixel corr proxy (test, n={pixel.get('n')}): "
            f"mean={pixel.get('mean_corr')} std={pixel.get('std_corr')}"
        )
    summary = "\n".join(lines) + "\n"
    (args.out_dir / "pairing_audit_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
