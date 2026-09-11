Roboflow COCO exports used in the paper
=======================================

Expected local folders (not shipped in this repository):

  Drone Detection.v6i.coco
  Drone Detection -Tutorial Dataset-.v2i.coco

Images usually sit next to `_annotations.coco.json` under train/valid/test
(Roboflow COCO zip layout), or under train/images/.

Converter: scripts/coco_roboflow_to_yolo.py

Search order for each file_name:
  <split>/<file_name>, <split>/<basename>, <split>/images/...

Example commands (from the project root that contains scripts/):

1) v6:
   python scripts/coco_roboflow_to_yolo.py --coco-root "../Drone Detection.v6i.coco" ^
      --out "data/external/roboflow_drone_v6_yolo"

2) Tutorial v2:
   python scripts/coco_roboflow_to_yolo.py ^
      --coco-root "../Drone Detection -Tutorial Dataset-.v2i.coco" ^
      --out "data/external/roboflow_drone_tutorial_v2_yolo"

3) Evaluate a frozen RGB checkpoint:
   python scripts/eval_checkpoint.py --weights runs/detect/train_rgb_full/weights/best.pt ^
      --data data/external/roboflow_drone_v6_yolo/dataset.yaml --splits test
