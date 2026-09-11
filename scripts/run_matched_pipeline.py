"""
Orchestrates matched Rc/Rk evaluation steps; skips training when checkpoints already exist.

Order:
  1) Pairing audit (if missing)
  2) Build Rk subsets 42/43/44
  3) Train Rc×3 + Rk×3 under identical 100/100 schedule (skip if best.pt exists)
  4) Eval val/test + aggregate mean±std
  5) Eval Rx/Rc on n=135 paired RGB test frames

  python scripts/run_matched_pipeline.py
  python scripts/run_matched_pipeline.py --skip-train   # only post steps
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXPORTS = _ROOT / "results" / "exports"


def _run(cmd: list[str]) -> None:
    print("\n>>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    py = sys.executable
    _EXPORTS.mkdir(parents=True, exist_ok=True)

    print("CUDA check:", flush=True)
    subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"], cwd=str(_ROOT), check=False)

    # 1) Pairing audit
    audit = _EXPORTS / "pairing_audit.json"
    if not audit.is_file():
        _run([py, "scripts/audit_pairing_geometry.py", "--pixel-sample", "40"])
    else:
        print("SKIP pairing audit (exists)", flush=True)

    # 2–4) matched Rc/Rk
    if not args.skip_train:
        _run(
            [
                py,
                "scripts/run_rk_rc_matched.py",
                "--phase",
                "all",
                "--sample-seeds",
                "42,43,44",
                "--train-seeds",
                "42,43,44",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
            ]
        )
    else:
        _run([py, "scripts/run_rk_rc_matched.py", "--phase", "eval"])
        _run([py, "scripts/run_rk_rc_matched.py", "--phase", "aggregate"])

    # 5) Paired-test subset eval (full-RGB vs n=135) if helper exists
    paired_script = _ROOT / "scripts" / "eval_paired_test_subset.py"
    if paired_script.is_file():
        _run([py, str(paired_script)])
    else:
        print("NOTE: eval_paired_test_subset.py not yet present; will add after train completes.", flush=True)

    summary = _EXPORTS / "rk_rc_matched" / "aggregate_summary.json"
    status = {
        "aggregate_summary": str(summary) if summary.is_file() else None,
        "pairing_audit": str(audit) if audit.is_file() else None,
    }
    (_EXPORTS / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("\n=== PIPELINE FINISHED ===", flush=True)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
