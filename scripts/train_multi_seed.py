"""
Launch the same YOLO training with multiple seeds (separate runs for variance reporting).

  python scripts/train_multi_seed.py --seeds 42,43,44 --data configs/generated/dbev_rgb.yaml \\
      --epochs 100 --batch 8 --name-prefix train_rgb_full_seed

Each run uses: runs/detect/<name-prefix>_<seed>/
After training, aggregate with: python scripts/aggregate_seed_metrics.py --glob "runs/detect/train_rgb_full_seed_*"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--name-prefix", type=str, required=True)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--paper-snapshot", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    py = sys.executable
    run_yolo = _ROOT / "scripts" / "run_yolo.py"
    for s in seeds:
        name = f"{args.name_prefix}_{s}"
        cmd = [
            py,
            str(run_yolo),
            "train",
            "--data",
            str(args.data),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--imgsz",
            str(args.imgsz),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--name",
            name,
            "--patience",
            str(args.patience),
            "--seed",
            str(s),
        ]
        if args.paper_snapshot:
            cmd.append("--paper-snapshot")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_ROOT), check=True)


if __name__ == "__main__":
    main()
