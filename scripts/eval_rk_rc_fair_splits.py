"""Re-eval matched Rc/Rk on shared test populations (official full + paired135)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO

from scripts.metrics_util import det_metrics_payload, resolve_ultralytics_device, ultralytics_val_output_kw

_OUT = _ROOT / "results" / "exports" / "rk_rc_matched"
_OFFICIAL = _ROOT / "configs" / "generated" / "dbev_rgb.yaml"
_PAIRED = _ROOT / "configs" / "generated" / "dbev_rgb_paired_test135.yaml"

RUNS = [
    ("Rc", "matched_rc_seed42"),
    ("Rc", "matched_rc_seed43"),
    ("Rc", "matched_rc_seed44"),
    ("Rk", "matched_rk_s42_t42"),
    ("Rk", "matched_rk_s43_t43"),
    ("Rk", "matched_rk_s44_t44"),
]


def _eval_one(name: str, contract: str, data: Path, tag: str) -> dict:
    out = _OUT / f"{name}_{tag}.json"
    if out.is_file():
        print(f"SKIP exists {out.name}", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))
    weights = _ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    print(f"EVAL {name} {tag}", flush=True)
    model = YOLO(str(weights))
    device = resolve_ultralytics_device("0")
    kw = ultralytics_val_output_kw(run_id=f"{name}_{tag}", split="test")
    metrics = model.val(
        data=str(data),
        split="test",
        device=device,
        imgsz=640,
        batch=8,
        plots=False,
        verbose=False,
        **kw,
    )
    payload = det_metrics_payload(
        metrics, data=data, weights=weights, split="test", device=device, imgsz=640, batch=8
    )
    payload["contract"] = contract
    payload["run_name"] = name
    payload["eval_tag"] = tag
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    m = payload["metrics"]
    print(f"  mAP50-95={m['mAP50-95']:.4f} mAP50={m['mAP50']:.4f}", flush=True)
    return payload


def _aggregate(tag: str, note: str) -> dict:
    rc, rk = [], []
    detail = {"Rc": [], "Rk": []}
    for contract, name in RUNS:
        p = _OUT / f"{name}_{tag}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["metrics"]
        row = {
            "file": p.name,
            "run": name,
            "mAP50": m["mAP50"],
            "mAP50-95": m["mAP50-95"],
            "P": m.get("precision_mean"),
            "R": m.get("recall_mean"),
            "F1": m.get("f1_mean"),
        }
        detail[contract].append(row)
        (rc if contract == "Rc" else rk).append(float(m["mAP50-95"]))

    def pack(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "values": vals,
            "mean": statistics.fmean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    summary = {
        "eval_tag": tag,
        "schedule": {"epochs": 100, "patience": 100, "batch": 8, "model": "yolov8n"},
        "Rc_test_mAP50-95": pack(rc),
        "Rk_test_mAP50-95": pack(rk),
        "delta_mean_Rc_minus_Rk": pack(rc)["mean"] - pack(rk)["mean"],
        "detail": detail,
        "note": note,
    }
    out = _OUT / f"aggregate_{tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("Wrote", out, flush=True)
    return summary


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    for contract, name in RUNS:
        _eval_one(name, contract, _OFFICIAL, "official_test")
        _eval_one(name, contract, _PAIRED, "paired135_test")
    _aggregate(
        "official_test",
        "Shared official RGB test (n=1999). Matched 100/100 schedule. "
        "Rc: co-observable 2070, seeds 42/43/44. Rk: independent 2070 subsets.",
    )
    _aggregate(
        "paired135_test",
        "Shared strictly paired RGB test (n=135). Matched 100/100 schedule.",
    )


if __name__ == "__main__":
    main()
