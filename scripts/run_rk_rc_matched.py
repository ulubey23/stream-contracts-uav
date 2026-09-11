"""
Matched-budget Rc vs Rk training and evaluation.

Identical Ultralytics schedule for Rc and Rk (default 100 epochs / patience 100).
- Rc: co-observable RGB train (2070), training seeds 42/43/44
- Rk: three independently sampled 2070 subsets (sample seeds 42/43/44),
      each trained with matching train seed

Skips runs that already have weights/best.pt so a crash/reboot can resume.

  python scripts/run_rk_rc_matched.py --phase all
  python scripts/run_rk_rc_matched.py --phase train

Outputs: results/exports/rk_rc_matched/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_OUT = _ROOT / "results" / "exports" / "rk_rc_matched"


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def _has_best(name: str) -> bool:
    return (_ROOT / "runs" / "detect" / name / "weights" / "best.pt").is_file()


def _epoch_count(name: str) -> int:
    csv_path = _ROOT / "runs" / "detect" / name / "results.csv"
    if not csv_path.is_file():
        return 0
    try:
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        return max(0, len(lines) - 1)
    except Exception:
        return 0


def _is_complete(name: str, epochs: int) -> bool:
    """Finished only if results.csv reached target epochs (best.pt alone is not enough)."""
    return _has_best(name) and _epoch_count(name) >= epochs


def _has_last(name: str) -> bool:
    return (_ROOT / "runs" / "detect" / name / "weights" / "last.pt").is_file()


def build_subsets(sample_seeds: list[int], n: int) -> list[Path]:
    """Create Rk YAMLs for each sampling seed (reuse folder if already populated)."""
    yamls = []
    py = sys.executable
    builder = _ROOT / "scripts" / "build_b1_size_matched_subset.py"
    for s in sample_seeds:
        out = _ROOT / "data" / "derived" / f"b1_rgb_train2070_seed{s}"
        yaml_path = _ROOT / "configs" / "generated" / f"dbev_rgb_train2070_control_seed{s}.yaml"
        img_dir = out / "train" / "images"
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.is_dir() else 0
        if n_imgs >= n and yaml_path.is_file():
            print(f"SKIP build seed{s}: already have {n_imgs} images + yaml", flush=True)
            yamls.append(yaml_path)
            continue
        _run(
            [
                py,
                str(builder),
                "--n",
                str(n),
                "--seed",
                str(s),
                "--out",
                str(out),
            ]
        )
        default_yaml = _ROOT / "configs" / "generated" / "dbev_rgb_train2070_control.yaml"
        yaml_path.write_text(default_yaml.read_text(encoding="utf-8"), encoding="utf-8")
        yamls.append(yaml_path)
        print("Rk yaml:", yaml_path, flush=True)
    return yamls


def train_runs(
    *,
    rc_data: Path,
    rk_yamls: list[Path],
    train_seeds: list[int],
    epochs: int,
    patience: int,
    batch: int,
    workers: int,
    dry: bool,
) -> list[dict]:
    py = sys.executable
    run_yolo = _ROOT / "scripts" / "run_yolo.py"
    plan: list[dict] = []

    if not rc_data.is_file():
        raise SystemExit(f"Missing Rc data yaml: {rc_data}")

    def _train_one(name: str, data: Path, seed: int) -> None:
        if dry:
            print("DRY", name, flush=True)
            return
        if _is_complete(name, epochs):
            print(f"SKIP complete: {name} ({_epoch_count(name)}/{epochs} epochs)", flush=True)
            return
        # Resume interrupted run if last.pt exists
        if _has_last(name) and _epoch_count(name) > 0:
            print(
                f"\n==== RESUME {name} from epoch {_epoch_count(name)}/{epochs} (workers={workers}) ====\n",
                flush=True,
            )
            # run_yolo --resume is a flag; it loads runs/detect/<name>/weights/last.pt
            _run(
                [
                    py,
                    str(run_yolo),
                    "train",
                    "--data",
                    str(data),
                    "--name",
                    name,
                    "--resume",
                    "--workers",
                    str(workers),
                    "--batch",
                    str(batch),
                ]
            )
            return
        print(f"\n==== TRAIN {name} seed={seed} (workers={workers}) ====\n", flush=True)
        _run(
            [
                py,
                str(run_yolo),
                "train",
                "--data",
                str(data),
                "--epochs",
                str(epochs),
                "--patience",
                str(patience),
                "--batch",
                str(batch),
                "--seed",
                str(seed),
                "--name",
                name,
                "--workers",
                str(workers),
                "--paper-snapshot",
            ]
        )

    # Rc: same data, multiple training seeds
    for ts in train_seeds:
        name = f"matched_rc_seed{ts}"
        plan.append({"contract": "Rc", "name": name, "data": str(rc_data.resolve()), "seed": ts})
        _train_one(name, rc_data, ts)

    # Rk: one train seed per sample seed (3 subsets → 3 runs)
    for yml in rk_yamls:
        if not yml.is_file():
            raise SystemExit(f"Missing Rk yaml: {yml}")
        sample_seed = int(yml.stem.split("seed")[-1])
        for ts in train_seeds:
            if ts != sample_seed:
                continue
            name = f"matched_rk_s{sample_seed}_t{ts}"
            plan.append(
                {
                    "contract": "Rk",
                    "name": name,
                    "data": str(yml.resolve()),
                    "sample_seed": sample_seed,
                    "seed": ts,
                }
            )
            _train_one(name, yml, ts)

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "train_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("Wrote", _OUT / "train_plan.json", flush=True)
    print("Plan:", json.dumps(plan, indent=2), flush=True)
    return plan


def eval_plan(plan: list[dict], dry: bool) -> None:
    """Ultralytics val on official val/test for each completed run."""
    from ultralytics import YOLO

    from scripts.metrics_util import det_metrics_payload, resolve_ultralytics_device, ultralytics_val_output_kw

    device = resolve_ultralytics_device("0")
    rows = []
    for item in plan:
        name = item["name"]
        weights = _ROOT / "runs" / "detect" / name / "weights" / "best.pt"
        data = item["data"]
        if not weights.is_file():
            print("SKIP missing weights:", weights, flush=True)
            continue
        model = YOLO(str(weights))
        for split in ("val", "test"):
            out_json = _OUT / f"{name}_{split}.json"
            if dry:
                rows.append({"name": name, "split": split})
                continue
            if out_json.is_file():
                print(f"SKIP eval exists: {out_json.name}", flush=True)
                continue
            print(f"EVAL {name} {split}", flush=True)
            ul_split = "val" if split == "val" else "test"
            kw = ultralytics_val_output_kw(run_id=name, split=ul_split)
            metrics = model.val(
                data=data,
                split=ul_split,
                device=device,
                imgsz=640,
                batch=8,
                plots=False,
                verbose=False,
                **kw,
            )
            payload = det_metrics_payload(
                metrics,
                data=data,
                weights=weights,
                split=ul_split,
                device=device,
                imgsz=640,
                batch=8,
            )
            payload["contract"] = item.get("contract")
            payload["run_name"] = name
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"  mAP50-95={payload['metrics']['mAP50-95']:.4f} "
                f"mAP50={payload['metrics']['mAP50']:.4f}",
                flush=True,
            )
        rows.append(item)
    (_OUT / "eval_index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def aggregate() -> None:
    import statistics

    rc_vals, rk_vals = [], []
    detail = {"Rc": [], "Rk": []}
    for p in sorted(_OUT.glob("matched_*_test.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("metrics") or {}
        v = m.get("mAP50-95")
        if v is None:
            continue
        row = {
            "file": p.name,
            "run": d.get("run_name"),
            "mAP50": m.get("mAP50"),
            "mAP50-95": v,
            "P": m.get("precision_mean"),
            "R": m.get("recall_mean"),
            "F1": m.get("f1_mean"),
        }
        if p.name.startswith("matched_rc"):
            rc_vals.append(float(v))
            detail["Rc"].append(row)
        elif p.name.startswith("matched_rk"):
            rk_vals.append(float(v))
            detail["Rk"].append(row)

    def pack(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "values": vals,
            "mean": statistics.fmean(vals) if vals else None,
            "stdev": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None),
        }

    summary = {
        "schedule": {"epochs": 100, "patience": 100, "batch": 8, "model": "yolov8n"},
        "Rc_test_mAP50-95": pack(rc_vals),
        "Rk_test_mAP50-95": pack(rk_vals),
        "detail": detail,
        "note": (
            "Rc: same co-observable 2070 RGB set, training seeds 42/43/44. "
            "Rk: three independent 2070 subsets (sample seeds 42/43/44), each with matching train seed."
        ),
    }
    out = _OUT / "aggregate_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("Wrote", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "train", "eval", "aggregate", "all"], default="all")
    ap.add_argument("--sample-seeds", type=str, default="42,43,44")
    ap.add_argument("--train-seeds", type=str, default="42,43,44")
    ap.add_argument("--n", type=int, default=2070)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers (use 2 to avoid RAM OOM on 8GB systems).",
    )
    ap.add_argument(
        "--rc-data",
        type=Path,
        default=_ROOT / "configs" / "generated" / "dbev_rgb_paired_strict.yaml",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sample_seeds = [int(x) for x in args.sample_seeds.split(",") if x.strip()]
    train_seeds = [int(x) for x in args.train_seeds.split(",") if x.strip()]
    _OUT.mkdir(parents=True, exist_ok=True)

    print("=== Matched-budget Rc/Rk training ===", flush=True)
    print(
        f"phase={args.phase} epochs={args.epochs} patience={args.patience} "
        f"batch={args.batch} workers={args.workers}",
        flush=True,
    )
    print(f"Rc data: {args.rc_data}", flush=True)
    print(f"sample_seeds={sample_seeds} train_seeds={train_seeds}", flush=True)

    rk_yamls: list[Path] = []
    plan: list[dict] = []

    if args.phase in ("build", "all"):
        rk_yamls = (
            [
                _ROOT / "configs" / "generated" / f"dbev_rgb_train2070_control_seed{s}.yaml"
                for s in sample_seeds
            ]
            if args.dry_run
            else build_subsets(sample_seeds, args.n)
        )

    if args.phase in ("train", "all"):
        if not rk_yamls:
            rk_yamls = [
                _ROOT / "configs" / "generated" / f"dbev_rgb_train2070_control_seed{s}.yaml"
                for s in sample_seeds
            ]
        plan = train_runs(
            rc_data=args.rc_data,
            rk_yamls=rk_yamls,
            train_seeds=train_seeds,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            workers=args.workers,
            dry=args.dry_run,
        )

    if args.phase in ("eval", "all"):
        if not plan:
            plan_path = _OUT / "train_plan.json"
            if plan_path.is_file():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            else:
                print("No train_plan.json; run train phase first", flush=True)
                plan = []
        if plan:
            eval_plan(plan, dry=args.dry_run)

    if args.phase in ("aggregate", "all"):
        aggregate()

    print("=== DONE phase", args.phase, "===", flush=True)


if __name__ == "__main__":
    main()
