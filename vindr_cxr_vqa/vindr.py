"""Readers for the local VinDr-CXR copy (CSV annotations + resized PNGs).

Only the standard library is used, so the dataset can be assembled on a plain
CPU box with no scientific-Python stack installed.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field

from .boxes import Box, iou, mean_box
from .constants import NO_FINDING


@dataclass
class RadBox:
    """One radiologist-drawn box, in original DICOM pixel space."""
    image_id: str
    class_name: str
    rad_id: str | None
    bbox: Box


@dataclass
class ImageMeta:
    image_id: str
    path: str
    width: int           # original DICOM width
    height: int          # original DICOM height
    resized_width: int
    resized_height: int
    fx: float            # original -> resized scale, x
    fy: float            # original -> resized scale, y
    partition: str       # "train" or "test" (VinDr's official partition)
    boxes: list[RadBox] = field(default_factory=list)
    labels_union: list[str] = field(default_factory=list)
    labels_majority: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
def read_annotations(path: str) -> dict[str, list[RadBox]]:
    """Read ``annotations_{train,test}.csv``.

    The train file carries a ``rad_id`` column (three radiologists per image);
    the test file is already consensus-fused and has none.  Rows whose
    ``x_min`` is blank are "No finding" rows and carry no geometry.
    """
    out: dict[str, list[RadBox]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("x_min"):
                continue
            out[row["image_id"]].append(RadBox(
                image_id=row["image_id"],
                class_name=row["class_name"],
                rad_id=row.get("rad_id"),
                bbox=(float(row["x_min"]), float(row["y_min"]),
                      float(row["x_max"]), float(row["y_max"])),
            ))
    return dict(out)


def read_resize_factors(path: str) -> dict[str, dict[str, float]]:
    """Read the ``resize_factors.csv`` shipped next to the resized PNGs."""
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["image_id"]] = {
                "width": int(row["original_width"]),
                "height": int(row["original_height"]),
                "resized_width": int(row["target_width"]),
                "resized_height": int(row["target_height"]),
                "fx": float(row["resize_factor_w"]),
                "fy": float(row["resize_factor_h"]),
            }
    return out


def read_image_labels(path: str) -> dict[str, list[list[str]]]:
    """Read ``image_labels_{train,test}.csv`` as image_id -> per-rater label lists.

    The test file has one row per image, the train file three (one per
    radiologist), so the value is always a list of label lists.
    """
    out: dict[str, list[list[str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        label_cols = [c for c in (reader.fieldnames or [])
                      if c not in ("image_id", "rad_id")]
        for row in reader:
            out[row["image_id"]].append(
                [c for c in label_cols if row[c] == "1" and c != NO_FINDING])
    return dict(out)


def majority_labels(per_rater: list[list[str]]) -> list[str]:
    """Labels agreed on by more than half of the raters."""
    if not per_rater:
        return []
    votes: dict[str, int] = defaultdict(int)
    for labels in per_rater:
        for lab in set(labels):
            votes[lab] += 1
    need = len(per_rater) / 2
    return sorted(lab for lab, v in votes.items() if v > need)


def union_labels(per_rater: list[list[str]]) -> list[str]:
    return sorted({lab for labels in per_rater for lab in labels})


# --------------------------------------------------------------------------
def fuse_boxes(boxes: list[RadBox], thr: float = 0.3) -> list[dict]:
    """Cluster same-class radiologist boxes and average each cluster.

    VinDr-CXR train images are read independently by three radiologists, so a
    single lesion usually appears as two or three near-duplicate boxes.  Greedy
    IoU clustering per class collapses them into one consensus box and keeps
    genuinely separate lesions apart.  Returned clusters are sorted by vote
    count (descending), so ``[0]`` is the most strongly agreed-on lesion.
    """
    by_class: dict[str, list[RadBox]] = defaultdict(list)
    for b in boxes:
        by_class[b.class_name].append(b)

    fused: list[dict] = []
    for cls, items in by_class.items():
        clusters: list[list[RadBox]] = []
        # Largest boxes first: a big box anchors the cluster its parts fall into.
        for b in sorted(items, key=lambda r: -( (r.bbox[2]-r.bbox[0]) * (r.bbox[3]-r.bbox[1]) )):
            for cl in clusters:
                if any(iou(b.bbox, o.bbox) >= thr for o in cl):
                    cl.append(b)
                    break
            else:
                clusters.append([b])
        for cl in clusters:
            fused.append({
                "class_name": cls,
                "bbox": mean_box(b.bbox for b in cl),
                "votes": len({b.rad_id for b in cl}) if cl[0].rad_id else 1,
                "n_boxes": len(cl),
                "rad_ids": sorted({b.rad_id for b in cl if b.rad_id}),
            })
    fused.sort(key=lambda d: (-d["votes"], -d["n_boxes"]))
    return fused


def load_images(
    dataset_dir: str, partition: str = "train"
) -> dict[str, ImageMeta]:
    """Assemble :class:`ImageMeta` for every PNG of a VinDr partition."""
    img_dir = os.path.join(dataset_dir, f"{partition}_data", partition)
    factors = read_resize_factors(os.path.join(img_dir, "resize_factors.csv"))
    anns = read_annotations(
        os.path.join(dataset_dir, f"annotations_{partition}.csv"))
    labels = read_image_labels(
        os.path.join(dataset_dir, f"image_labels_{partition}.csv"))

    out: dict[str, ImageMeta] = {}
    for fname in os.listdir(img_dir):
        if not fname.endswith(".png"):
            continue
        iid = fname[:-4]
        f = factors.get(iid)
        if f is None:
            continue
        out[iid] = ImageMeta(
            image_id=iid,
            path=os.path.join(img_dir, fname).replace("\\", "/"),
            width=f["width"], height=f["height"],
            resized_width=f["resized_width"], resized_height=f["resized_height"],
            fx=f["fx"], fy=f["fy"],
            partition=partition,
            boxes=anns.get(iid, []),
            labels_union=union_labels(labels.get(iid, [])),
            labels_majority=majority_labels(labels.get(iid, [])),
        )
    return out
