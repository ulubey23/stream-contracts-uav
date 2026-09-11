"""
Convert Roboflow COCO export folders to Ultralytics YOLO layout.

Expected layout (per split: train, valid, test):
  <split>/_annotations.coco.json
  Images are resolved in this order (first hit wins):
    1) <split>/<file_name> as given in COCO (Roboflow often flattens JPGs next to the JSON)
    2) <split>/<basename(file_name)> if file_name contains a subpath that is not used on disk
    3) <split>/images/<basename(file_name)> (YOLO-style subfolder exports)

Output:
  <out>/{train,valid,test}/{images,labels}/  + dataset.yaml

Usage:
  python scripts/coco_roboflow_to_yolo.py \\
      --coco-root "../../Drone Detection.v6i.coco" \\
      --out "../data/external/roboflow_drone_v6_yolo"

If zero images are found, place Roboflow-exported JPG/PNG files next to the JSON or re-download
the dataset with the \"folder\" export that includes images.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _resolve_source_image(split_dir: Path, file_name: str) -> Path | None:
    """Locate image file under a Roboflow COCO split folder."""
    fn = Path(file_name)
    candidates = [
        split_dir / file_name,
        split_dir / fn.name,
        split_dir / "images" / file_name,
        split_dir / "images" / fn.name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _coco_bbox_to_yolo_line(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int, cls: int
) -> str:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--single-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map all COCO categories to class index 0 (recommended for drone-only exports).",
    )
    args = ap.parse_args()

    coco_root = args.coco_root.resolve()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total_miss = 0
    total_img = 0
    yaml_nc, yaml_names = 1, ["drone"]
    for split in ("train", "valid", "test"):
        if not (coco_root / split / "_annotations.coco.json").is_file():
            print("skip split (no json):", split)
            continue
        # re-read per split
        split_dir = coco_root / split
        js = split_dir / "_annotations.coco.json"
        data = json.loads(js.read_text(encoding="utf-8"))
        cat_ids = sorted({c["id"] for c in data["categories"]})
        if args.single_class:
            cat_map = {cid: 0 for cid in cat_ids}
            nc = 1
            names = ["drone"]
        else:
            cat_map = {cid: i for i, cid in enumerate(cat_ids)}
            nc = len(cat_ids)
            names = [f"class_{i}" for i in range(nc)]
        yaml_nc, yaml_names = nc, names

        id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
        id_to_size = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        by_image: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            cid = cat_map[int(ann["category_id"])]
            x, y, w, h = ann["bbox"]
            iw, ih = id_to_size[iid]
            line = _coco_bbox_to_yolo_line(float(x), float(y), float(w), float(h), iw, ih, cid)
            by_image.setdefault(iid, []).append(line)

        img_out = out_root / split / "images"
        lb_out = out_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lb_out.mkdir(parents=True, exist_ok=True)

        n_img = n_miss = 0
        for iid, fname in id_to_file.items():
            src = _resolve_source_image(split_dir, fname)
            if src is None:
                n_miss += 1
                continue
            dst_name = Path(fname).name
            shutil.copy2(src, img_out / dst_name)
            (lb_out / (Path(fname).stem + ".txt")).write_text(
                "\n".join(by_image.get(iid, [])) + ("\n" if by_image.get(iid) else ""),
                encoding="utf-8",
            )
            n_img += 1
        total_img += n_img
        total_miss += n_miss
        print(f"{split}: images copied={n_img} missing_files={n_miss}")

    if total_img == 0:
        print(
            "\nERROR: No images were copied. Roboflow COCO zips sometimes contain only JSON.\n"
            "Re-export with images, or copy JPG/PNG files into train/valid/test next to _annotations.coco.json.\n"
        )
        raise SystemExit(1)

    names_yaml = "\n".join(f"- {n}" for n in yaml_names)
    yaml_text = f"""path: {out_root.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: {yaml_nc}
names:
{names_yaml}
"""
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    print("Wrote", out_root / "dataset.yaml")
    if total_miss:
        print("WARNING:", total_miss, "annotations referenced missing image files (skipped).")


if __name__ == "__main__":
    main()
