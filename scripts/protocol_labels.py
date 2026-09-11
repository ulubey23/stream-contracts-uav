"""
Human-facing display labels for stream-contract experiment keys.

Internal experiment keys in metrics JSON and training folders are unchanged;
only human-facing labels are mapped here.
"""

# Primary dual-stream protocols
RX = r"Rx (RGB exhaustive)"
DX = r"Dx (depth exhaustive)"
RC = r"Rc (RGB co-observable)"
RK = r"Rk (RGB cardinality control)"

RX_V8N = r"Rx (RGB exhaustive, YOLOv8n)"
DX_V8N = r"Dx (depth exhaustive, YOLOv8n)"
RC_V8N = r"Rc (RGB co-observable, YOLOv8n)"

RS_V8S = r"Rs (RGB YOLOv8s)"

PAPER_LABEL_BY_KEY: dict[str, str] = {
    "B1_rgb_full": "Rx (RGB exhaustive)",
    "B2_depth_full": "Dx (depth exhaustive)",
    "P1_rgb_paired_strict_correct_labels": "Rc (RGB co-observable)",
    "B1_control_train2070": "Rk (RGB cardinality control)",
    "M_rgb_yolov8s_full": "Rs (RGB YOLOv8s)",
    "Y11s_rgb_full": "Y11s (RGB exhaustive)",
    "Y11n_rgb_full": "Y11n (RGB exhaustive)",
    "Y11n_depth_full": "Y11n (depth exhaustive)",
    "Y11n_rgb_paired": "Y11n (RGB co-observable)",
    "RTDETR_l_rgb_full": "RT-DETR--L (RGB exhaustive)",
}
