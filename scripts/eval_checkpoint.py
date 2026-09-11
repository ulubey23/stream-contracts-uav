"""
Re-evaluate a trained checkpoint without retraining (val/test, JSON metrics, optional plots).

Evaluate a frozen checkpoint on a given data YAML (same weights, same protocol).

  python scripts/eval_checkpoint.py --weights runs/detect/train_rgb_full/weights/best.pt \\
      --data configs/generated/dbev_rgb.yaml --splits val,test
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.metrics_util import (  # noqa: E402
    det_metrics_payload,
    resolve_ultralytics_device,
    ultralytics_val_output_kw,
)


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", type=str, default="val,test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers (0 avoids extra processes; use on Windows if page-file / DLL errors).",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results/reval_<stamp>/",
    )
    args = ap.parse_args()

    w = args.weights.resolve()
    if not w.is_file():
        raise SystemExit(f"Missing weights {w}")

    out_root = args.out_dir
    if out_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = _ROOT / "results" / f"reval_{stamp}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dev = resolve_ultralytics_device(args.device)
    model = YOLO(str(w))
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    summary: dict[str, object] = {"weights": str(w), "data": str(args.data), "splits": {}}
    rid = out_root.name
    for split in splits:
        metrics = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            split=split,
            device=dev,
            batch=args.batch,
            workers=args.workers,
            plots=args.plots,
            **ultralytics_val_output_kw(run_id=rid, split=split),
        )
        payload = det_metrics_payload(
            metrics,
            data=args.data,
            weights=w,
            split=split,
            device=dev,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        summary["splits"][split] = payload
        (out_root / f"{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
