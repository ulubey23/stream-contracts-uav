"""
Aggregate val/test mAP from multiple paper_snapshots or reval folders (mean ± std).

  python scripts/aggregate_seed_metrics.py --summary-glob "results/reval_* /summary.json"
  (Adjust glob in shell; this script accepts repeated --json paths.)

Example after manual revals:
  python scripts/aggregate_seed_metrics.py --json results/reval_a/test.json results/reval_b/test.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _extract(payload: dict) -> dict[str, float]:
    m = payload.get("metrics") or payload.get("results_dict")
    if m is None:
        raise ValueError("No metrics in payload")
    if "mAP50" in m:
        return {
            "mAP50": float(m["mAP50"]),
            "mAP50-95": float(m.get("mAP50-95") or m.get("mAP50_95") or 0.0),
        }
    # ultralytics results_dict style
    return {
        "mAP50": float(m.get("metrics/mAP50(B)", m.get("map50", 0))),
        "mAP50-95": float(m.get("metrics/mAP50-95(B)", m.get("map", 0))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, nargs="+", required=True, help="Per-seed metric JSON (e.g. test.json).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for p in args.json:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.append(_extract(d))

    def stat(key: str) -> dict[str, float]:
        xs = [r[key] for r in rows]
        return {"mean": float(statistics.mean(xs)), "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0}

    out = {
        "n": len(rows),
        "mAP50": stat("mAP50"),
        "mAP50-95": stat("mAP50-95"),
        "sources": [str(p) for p in args.json],
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
