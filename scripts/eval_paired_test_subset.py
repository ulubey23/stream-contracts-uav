"""
Evaluate Rc (and Rx) on the full official RGB test and on the n=135 paired RGB frames.

  python scripts/eval_paired_test_subset.py

Writes results/exports/paired_subset_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.metrics_util import det_metrics_payload, resolve_ultralytics_device, ultralytics_val_output_kw


def _write_paired_yaml(pairs_json: Path, out_yaml: Path, out_img_list_dir: Path) -> int:
    """Build a YOLO data yaml whose 'test' points to a folder of symlinks/copies of paired RGB test images."""
    import shutil

    payload = json.loads(pairs_json.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    img_out = out_img_list_dir / "images"
    lb_out = out_img_list_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lb_out.mkdir(parents=True, exist_ok=True)

    # clear previous
    for p in img_out.glob("*"):
        p.unlink()
    for p in lb_out.glob("*"):
        p.unlink()

    n = 0
    for item in pairs:
        rgb = Path(item["rgb"])
        if not rgb.is_file():
            continue
        dst = img_out / rgb.name
        if not dst.exists():
            try:
                dst.hardlink_to(rgb)
            except Exception:
                shutil.copy2(rgb, dst)
        # labels next to RGB tree
        src_lb = rgb.parent.parent / "labels" / f"{rgb.stem}.txt"
        if src_lb.is_file():
            shutil.copy2(src_lb, lb_out / f"{rgb.stem}.txt")
        else:
            (lb_out / f"{rgb.stem}.txt").write_text("", encoding="utf-8")
        n += 1

    # Ultralytics wants train/val/test; reuse official val for val, paired folder for test
    from src.paths import rgb_dataset_root

    rgb_root = rgb_dataset_root()
    yaml_text = (
        f"train: {(rgb_root / 'train' / 'images').as_posix()}\n"
        f"val: {(rgb_root / 'valid' / 'images').as_posix()}\n"
        f"test: {img_out.resolve().as_posix()}\n"
        "nc: 1\n"
        "names: [drone]\n"
    )
    out_yaml.write_text(yaml_text, encoding="utf-8")
    return n


def _eval(weights: Path, data: Path, tag: str, split: str) -> dict:
    from ultralytics import YOLO

    device = resolve_ultralytics_device("0")
    model = YOLO(str(weights))
    kw = ultralytics_val_output_kw(run_id=tag, split=split)
    metrics = model.val(
        data=str(data),
        split=split,
        device=device,
        imgsz=640,
        batch=8,
        plots=False,
        verbose=False,
        **kw,
    )
    return det_metrics_payload(
        metrics, data=data, weights=weights, split=split, device=device, imgsz=640, batch=8
    )


def main() -> None:
    out_dir = _ROOT / "results" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _ROOT / "results" / "pairs_test_strict.json"
    if not pairs.is_file():
        raise SystemExit(f"Missing {pairs}")

    paired_root = _ROOT / "data" / "derived" / "paired_rgb_test135"
    paired_yaml = _ROOT / "configs" / "generated" / "dbev_rgb_paired_test135.yaml"
    n = _write_paired_yaml(pairs, paired_yaml, paired_root)
    print(f"Prepared paired test set: n={n}", flush=True)

    full_yaml = _ROOT / "configs" / "generated" / "dbev_rgb.yaml"
    runs = {
        "Rx": _ROOT / "runs" / "detect" / "train_rgb_full" / "weights" / "best.pt",
        "Rc_original": _ROOT / "runs" / "detect" / "train_rgb_paired_correct" / "weights" / "best.pt",
    }
    # Prefer matched-budget Rc seed42 if available
    rc_matched = _ROOT / "runs" / "detect" / "matched_rc_seed42" / "weights" / "best.pt"
    if rc_matched.is_file():
        runs["Rc_matched_seed42"] = rc_matched

    results = {"n_paired_test": n, "evals": {}}
    for tag, w in runs.items():
        if not w.is_file():
            print("SKIP missing", w, flush=True)
            continue
        print(f"Eval {tag} on FULL official test...", flush=True)
        results["evals"][f"{tag}_full_test"] = _eval(w, full_yaml, f"pairedsub_{tag}_full", "test")
        print(f"Eval {tag} on PAIRED n={n} test...", flush=True)
        results["evals"][f"{tag}_paired135_test"] = _eval(
            w, paired_yaml, f"pairedsub_{tag}_p135", "test"
        )

    out = out_dir / "paired_subset_eval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", out, flush=True)
    for k, v in results["evals"].items():
        m = v["metrics"]
        print(f"  {k}: mAP50-95={m['mAP50-95']:.4f} mAP50={m['mAP50']:.4f}", flush=True)


if __name__ == "__main__":
    main()
