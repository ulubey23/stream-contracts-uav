"""
Train or validate YOLO models on generated DBEV configs (Ultralytics API).

Examples (from the repository root):
  python scripts/run_yolo.py val --data configs/generated/dbev_rgb.yaml
  python scripts/run_yolo.py train --data configs/generated/dbev_rgb.yaml --weights yolov8n.pt --epochs 100
  python scripts/run_yolo.py train --data configs/generated/dbev_rgb.yaml --weights yolo11s.pt --epochs 100 --batch 6 --name train_rgb_yolo11s_full
  python scripts/run_yolo.py train --data configs/generated/dbev_rgb.yaml --weights rtdetr-l.pt --epochs 100 --batch 4 --name train_rgb_rtdetr_l_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.metrics_util import det_metrics_payload, resolve_ultralytics_device, ultralytics_val_output_kw


def _train_project_dir(project: str) -> Path:
    """Absolute project root so Ultralytics does not nest under runs/detect twice on Windows."""
    p = Path(project)
    return p.resolve() if p.is_absolute() else (_ROOT / p).resolve()


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("val")
    p_val.add_argument("--data", type=Path, required=True)
    p_val.add_argument("--weights", type=str, default="yolov8n.pt")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p_val.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (GPU if available), cpu, or CUDA index e.g. 0",
    )
    p_val.add_argument("--batch", type=int, default=16, help="Inference batch size for validation.")
    p_val.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write metrics here instead of results/last_val_metrics.json.",
    )

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", type=Path, required=True)
    p_train.add_argument("--weights", type=str, default="yolov8n.pt")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--workers", type=int, default=8, help="Dataloader workers (lower if validation crashes).")
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (GPU if available), cpu, or CUDA index e.g. 0",
    )
    p_train.add_argument("--project", type=str, default="runs/detect")
    p_train.add_argument("--name", type=str, default="train")
    p_train.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Ultralytics `fraction`: subsample the train set (1.0 = all). Useful for quick pipeline checks.",
    )
    p_train.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <project>/<name>/weights/last.pt (same --project and --name as the interrupted run).",
    )
    p_train.add_argument(
        "--paper-snapshot",
        action="store_true",
        help="After training, copy weights/curves/CSV and run val+test into results/paper_snapshots/...",
    )
    p_train.add_argument(
        "--snapshot-splits",
        type=str,
        default="val,test",
        help="With --paper-snapshot: comma-separated splits for extra val passes (default val,test).",
    )
    p_train.add_argument(
        "--snapshot-no-val",
        action="store_true",
        help="With --paper-snapshot: only copy artifacts, skip YOLO val.",
    )
    p_train.add_argument("--seed", type=int, default=42, help="Ultralytics training seed.")
    p_train.add_argument(
        "--patience",
        type=int,
        default=18,
        help="Early stopping (align with configs/training_policy.yaml).",
    )

    args = ap.parse_args()
    dev = resolve_ultralytics_device(args.device)

    if args.cmd == "val":
        model = YOLO(args.weights)
        wstem = Path(args.weights).stem
        metrics = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            split=args.split,
            device=dev,
            batch=args.batch,
            **ultralytics_val_output_kw(run_id=wstem, split=args.split),
        )
        out = det_metrics_payload(
            metrics,
            data=args.data,
            weights=args.weights,
            split=args.split,
            device=dev,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        res_dir = _ROOT / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        outp = args.out_json if args.out_json is not None else (res_dir / "last_val_metrics.json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(out)
        print("Wrote", outp)
        return

    proj = _train_project_dir(args.project)

    if args.resume:
        last_pt = proj / args.name / "weights" / "last.pt"
        if not last_pt.is_file():
            raise SystemExit(f"--resume: missing checkpoint {last_pt}")
        model = YOLO(str(last_pt))
        # Override workers/batch so resume does not inherit OOM-prone settings (e.g. workers=8).
        model.train(resume=True, workers=args.workers, batch=args.batch, device=dev)
    else:
        model = YOLO(args.weights)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=str(proj),
            name=args.name,
            device=dev,
            exist_ok=True,
            fraction=args.fraction,
            workers=args.workers,
            seed=args.seed,
            patience=args.patience,
        )

    if args.paper_snapshot:
        from scripts.export_paper_snapshot import export_snapshot

        run_dir = proj / args.name
        splits = tuple(s.strip() for s in args.snapshot_splits.split(",") if s.strip())
        export_snapshot(
            run_dir,
            args.data,
            val_splits=splits,
            skip_val=args.snapshot_no_val,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
