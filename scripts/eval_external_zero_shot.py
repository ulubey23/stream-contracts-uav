"""
Zero-shot evaluation on external YOLO datasets (default: val + test); writes JSON for LaTeX sync.

  python scripts/eval_external_zero_shot.py \\
      --weights runs/detect/train_rgb_full/weights/best.pt \\
      --data data/external/roboflow_drone_tutorial_v2_yolo/dataset.yaml \\
      --tag tutorial_v2

Writes:
  results/external_roboflow_{tag}_zeroshot.json  (val + test metrics)
  results/external_roboflow_{tag}_test.json        (legacy test-only, same as zeroshot test slice)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0, help="Passed to eval_checkpoint (0 = safest on Windows).")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated Ultralytics split names (YAML uses val: valid/images).",
    )
    args = ap.parse_args()

    out_dir = _ROOT / "results" / f"reval_external_{args.tag}"
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "eval_checkpoint.py"),
        "--weights",
        str(args.weights),
        "--data",
        str(args.data),
        "--splits",
        args.splits,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--out-dir",
        str(out_dir),
        "--no-plots",
    ]
    subprocess.run(cmd, cwd=str(_ROOT), check=True)
    split_list = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    by_split: dict[str, dict] = {}
    for sp in split_list:
        pj = out_dir / f"{sp}.json"
        if not pj.is_file():
            raise SystemExit(f"Missing split output {pj}")
        payload = json.loads(pj.read_text(encoding="utf-8"))
        by_split[sp] = payload.get("metrics") or {}

    final_z = {
        "tag": args.tag,
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "val": by_split.get("val"),
        "test": by_split.get("test"),
    }
    zpath = _ROOT / "results" / f"external_roboflow_{args.tag}_zeroshot.json"
    zpath.write_text(json.dumps(final_z, indent=2), encoding="utf-8")
    print("Wrote", zpath)

    test_metrics = by_split.get("test")
    if test_metrics:
        legacy = {
            "tag": args.tag,
            "weights": str(args.weights.resolve()),
            "data": str(args.data.resolve()),
            "split": "test",
            "metrics": test_metrics,
        }
        tpath = _ROOT / "results" / f"external_roboflow_{args.tag}_test.json"
        tpath.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        print("Wrote", tpath)


if __name__ == "__main__":
    main()
