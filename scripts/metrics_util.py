"""Serialize Ultralytics DetMetrics and other objects to JSON-safe dicts."""
from __future__ import annotations

import math
import numbers
import re
from pathlib import Path
from typing import Any


def ultralytics_val_output_kw(*, run_id: str, split: str) -> dict[str, object]:
    """
    Pin Ultralytics val() save dir. Default (no project/name) creates runs/detect/val, val2, val13, ...
    and interrupted runs can leave empty folders. We use one project + stable names + exist_ok.
    """
    root = Path(__file__).resolve().parents[1]
    project = root / "runs" / "detect" / "_metrics_val"
    safe = re.sub(r"[^\w.\-]+", "_", f"{run_id}_{split}").strip("_")[:200]
    if not safe:
        safe = "val"
    return {"project": str(project), "name": safe, "exist_ok": True}


def resolve_ultralytics_device(device: str) -> str | int | None:
    """Ultralytics `device`: None = auto; 'cpu'; '0'.. for CUDA. Empty / 'auto' uses GPU if available."""
    d = (device or "").strip().lower()
    if d in ("", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return None
    if d == "cpu":
        return "cpu"
    if d.isdigit():
        return int(d)
    return d


def f1_and_accuracy_mean(precision: float, recall: float) -> tuple[float, float]:
    """
    Return F1 and the arithmetic mean (P+R)/2 from mean precision and recall.
    """
    p, r = float(precision), float(recall)
    eps = 1e-12
    f1 = 2.0 * p * r / (p + r + eps) if (p + r) > eps else 0.0
    p_r_mean = (p + r) / 2.0
    return f1, p_r_mean


def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, numbers.Integral) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, numbers.Real):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    mod = type(x).__module__
    if mod == "numpy" or mod.startswith("numpy."):
        import numpy as np

        if isinstance(x, np.generic):
            return json_safe(x.item())
        if isinstance(x, np.ndarray):
            return x.tolist()
    name = type(x).__name__
    if name in ("Tensor", "device"):
        try:
            return json_safe(x.detach().cpu().tolist())  # type: ignore[union-attr]
        except Exception:
            return str(x)
    return str(x)


def det_metrics_payload(
    metrics: Any,
    *,
    data: str | Path,
    weights: str | Path,
    split: str,
    device: Any,
    imgsz: int,
    batch: int,
) -> dict[str, Any]:
    box = metrics.box
    p_mean = float(box.mp)
    r_mean = float(box.mr)
    f1_m, acc_m = f1_and_accuracy_mean(p_mean, r_mean)
    out: dict[str, Any] = {
        "data": str(data),
        "weights": str(weights),
        "split": split,
        "device": json_safe(device),
        "imgsz": imgsz,
        "batch": batch,
        "metrics": {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "mAP75": float(box.map75) if getattr(box, "map75", None) is not None else None,
            "precision_mean": p_mean,
            "recall_mean": r_mean,
            "f1_mean": f1_m,
            "accuracy_mean": acc_m,
            "nc": int(box.nc) if box.nc is not None else None,
        },
    }
    maps = getattr(box, "maps", None)
    if maps is not None:
        try:
            out["metrics"]["per_class_map50-95"] = json_safe(maps)
        except Exception:
            pass
    rd = getattr(metrics, "results_dict", None)
    if rd:
        out["results_dict"] = json_safe(rd)
    speed = getattr(metrics, "speed", None)
    if speed:
        out["speed_ms_per_image"] = json_safe(speed)
    return out
