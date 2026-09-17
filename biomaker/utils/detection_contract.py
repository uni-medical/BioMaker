"""Conservative admission to the fixed VOC-style AP50 detection protocol."""

import json
import re

from .evaluator_protocol import metric_name, requested_metrics


def fixed_detection_spec(context, task):
    manifest = context["split_manifest"]
    heads = manifest.get("task_heads", [])
    heads = [h.get("type") if isinstance(h, dict) else h for h in heads]
    if not set(heads) & {"detection", "object_detection", "detection_2d"}:
        return None
    primary, secondary = requested_metrics(task)
    contract = manifest.get("metric_contract", {})
    primary = primary or metric_name(str(contract.get("primary_metric", "")))
    if primary not in {"ap50", "recall"} or any(
        m not in {"ap50", "recall"} for m in secondary
    ):
        return None
    # Alternative definitions must reach the generated evaluator, not be silently
    # replaced by defaults. Explicit supported parameters use structured fields.
    if re.search(
        r"coco|101[- ]?point|11[- ]?point|macro.?recall|average recall|class.agnostic|max.?det|confidence|score threshold|top.?k",
        task,
        re.I,
    ):
        return None
    thresholds = re.findall(r"IoU\s*(?:=|>=|≥|:)\s*([\d.]+)", task, re.I)
    if any(float(x.rstrip(".")) != 0.5 for x in thresholds):
        return None
    if contract.get("ap_interpolation", "voc_all_points") != "voc_all_points":
        return None
    if (
        contract.get("recall_average", "micro") != "micro"
        or contract.get("max_detections") is not None
    ):
        return None
    if contract.get("iou_threshold", 0.5) != 0.5:
        return None
    threshold = contract.get("recall_confidence_threshold", 0.0)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        return None
    keys = manifest.get("key_columns", [])
    if not keys or any(
        k not in context["score_df"] or k not in context["test_df"] for k in keys
    ):
        return None
    output = manifest.get("prediction_contract", {})
    coordinate_text = str(
        output.get("coordinate_convention", output.get("box_format", ""))
    ).lower()
    if "normalized" in coordinate_text:
        coordinates = "normalized"
    elif "pixel" in coordinate_text:
        coordinates = "pixels"
    else:
        return None
    schemas = [
        (["x_center", "y_center", "width", "height"], "cxcywh"),
        (["cx", "cy", "w", "h"], "cxcywh"),
        (["x1", "y1", "x2", "y2"], "xyxy"),
        (["xmin", "ymin", "xmax", "ymax"], "xyxy"),
        (["x", "y", "width", "height"], "xywh"),
    ]
    if output.get("box_format") in {"xyxy", "xywh", "cxcywh"}:
        schemas.append((["bbox"], output["box_format"]))
    candidates = []
    for column in context["answer_columns"]:
        if column not in context["hidden_df"]:
            continue
        boxes = []
        try:
            for value in context["hidden_df"][column]:
                items = json.loads(value)
                if not isinstance(items, list) or any(
                    not isinstance(b, dict) for b in items
                ):
                    raise ValueError("Not explicit JSON box objects")
                boxes.extend(items)
        except (ValueError, TypeError):
            continue
        if not boxes or any(
            any(b.get(k, False) for k in ("ignore", "iscrowd", "difficult"))
            for b in boxes
        ):
            continue
        matches = [
            (fields, fmt, ck)
            for fields, fmt in schemas
            for ck in ["category_id", "class_id", "class", "label"]
            if all(all(k in box for k in fields + [ck]) for box in boxes)
        ]
        if len(matches) == 1:
            fields, fmt, ck = matches[0]
            candidates.append(
                (column, dict(coordinate_keys=fields, box_format=fmt, class_key=ck))
            )
    if len(candidates) != 1:
        return None
    column, schema = candidates[0]
    return dict(
        evaluator_api_version=2,
        task_type="detection",
        prediction_type="columns",
        primary_metric=primary,
        secondary_metrics=secondary,
        maximize=True,
        key_columns=keys,
        prediction_columns=[column],
        answer_columns=[column],
        file_reference_columns=[],
        truth_schema=schema,
        prediction_schema=schema,
        box_format=schema["box_format"],
        coordinate_convention=coordinates,
        box_schema="JSON list of objects using prediction_schema keys plus score; [] for empty images",
        iou_threshold=0.5,
        ap_interpolation="voc_all_points",
        ap_average="macro_over_gt_classes",
        recall_average="micro",
        recall_confidence_threshold=float(threshold),
        max_detections=None,
        class_handling="class-aware; classes absent from split ground truth excluded from AP mean",
        matching="confidence-descending; highest-IoU GT; duplicate matches are false positives",
        validation_artifact="./working/validation_predictions.csv",
        test_submission="./working/submission.csv",
        score_split="./input/biomaker_splits/score_val.csv",
    )
