# Stream Contracts (RGB-Depth UAV Detection)

Code and supplementary experimental resources for:

**Stream Contracts: An Auditable and Reproducible Evaluation Framework for RGB–Depth UAV Detection**

This repository contains the scripts, configuration files, and evaluation results used in the RGB–depth UAV detection experiments reported in the paper.

## Contents

- `scripts/` - training, evaluation, pairing audit, matched Rc/Rk runs
- `configs/` - Ultralytics dataset YAML files (paths are placeholders; edit locally)
- `results/exports/` - pairing audit and matched-run evaluation outputs
- `results/external/` - zero-shot scores on the named Roboflow exports
- `requirements.txt` - Python dependencies

## Not included

Large assets are not stored in git:

- DBEV-UAV RGB and depth imagery (IEEE DataPort)
- Roboflow COCO zips:
  - Drone Detection -Tutorial Dataset-.v2i.coco
  - Drone Detection.v6i.coco
- Trained `.pt` weights (regenerate with the scripts, or request from the corresponding author)

After downloading the datasets, set the paths in the YAML files under `configs/`
(replace `<DBEV_RGB_ROOT>` / `<DBEV_DEPTH_ROOT>` or absolute local paths).

## External Roboflow note

Filename-stem overlap with DBEV-UAV RGB:

- Tutorial v2i: 0 overlapping stems (useful external contrast)
- Drone Detection v6i: 9972/9972 stems, split-aligned (train in train, valid in valid, test in test);
  sampled files are byte-identical to the source RGB set

High v6i scores therefore reflect subset/export consistency with DBEV-UAV RGB, not
independent out-of-distribution generalization. See `results/roboflow_provenance.json`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit dataset paths in `configs/`, then for example:

```bash
python scripts/audit_pairing_geometry.py
python scripts/run_matched_pipeline.py --skip-train
# Per-seed raw eval JSON files are omitted; aggregates are under results/exports/
```

(`--skip-train` assumes weights already exist under `runs/detect/`.)

More detail on converting Roboflow COCO exports: `README_ROBOFLOW_COCO.txt`.

## Citation

Please cite the associated IEEE Access article when available.
