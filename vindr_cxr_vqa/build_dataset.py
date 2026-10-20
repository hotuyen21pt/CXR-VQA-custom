"""Join the released VQA annotations with the local VinDr-CXR images.

Reads ``data_v1.json`` (4,394 images / 17,597 QA pairs) from the HuggingFace
release together with the local VinDr-CXR copy, and writes one JSONL record per
QA pair carrying the image path, every coordinate space the training code might
want, and a repaired copy of the ground-truth box.

Standard library only -- runs on CPU with nothing installed.

    python -m vindr_cxr_vqa.build_dataset --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from collections import Counter, defaultdict

from . import vindr
from .boxes import (Box, extract_locs, format_loc, iou, loc_token_width,
                    to_normalized, to_resized)
from .config import load_config
from .constants import LOC_NORM_RANGE, PATHOLOGIES

HF_URL = ("https://huggingface.co/datasets/Dangindev/VinDR-CXR-VQA/"
          "resolve/main/data_v1.json")

#: Rounding slack, in original pixels, when deciding whether the published box
#: is one of the radiologist boxes.  The release stores integers while the CSV
#: stores floats, so an exact comparison misses about a third of the boxes that
#: are in fact identical.
BOX_MATCH_TOL = 2.0


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def fetch_vqa_json(path: str, url: str = HF_URL) -> list[dict]:
    """Load ``data_v1.json``, downloading it on first use."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"[build] downloading {url}")
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _same_box(a: Box, b: Box, tol: float = BOX_MATCH_TOL) -> bool:
    return max(abs(x - y) for x, y in zip(a, b)) <= tol


def repair_box(published: Box, finding: str, raw: list[vindr.RadBox],
               fused: list[dict]) -> tuple[Box, str, bool]:
    """Pick the box that actually belongs to ``finding``.

    In the released data the ``gt_location`` of a QA pair is nearly always a
    genuine radiologist box on that image, but about 60% of the time it is the
    box of a *different* pathology than ``gt_finding``: the generation pipeline
    paired the finding list and the box list independently.  This restores the
    join.

    Consistency is judged against the *raw* radiologist boxes, because that is
    what the release copied from.  Repair falls back to the fused consensus
    box, which averages the two or three independent readings of the lesion.

    Returns ``(box, method, consistent)``.  ``method`` is one of

    ``"published"``    the published box already belongs to ``finding``;
    ``"best_iou"``     a consensus box of ``finding`` overlapping the published
                       one was kept;
    ``"consensus"``    no overlap, so the most strongly agreed-on box of
                       ``finding`` was used;
    ``"unavailable"``  the image carries no box for ``finding`` at all.

    ``consistent`` reports whether the *published* box was already correct, and
    is stored on every record so the mismatch rate stays auditable downstream.
    """
    if any(b.class_name == finding and _same_box(published, b.bbox)
           for b in raw):
        return published, "published", True
    cands = [f for f in fused if f["class_name"] == finding]
    if not cands:
        return published, "unavailable", False
    best = max(cands, key=lambda c: iou(published, c["bbox"]))
    method = "best_iou" if iou(published, best["bbox"]) > 0 else "consensus"
    return best["bbox"], method, False


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def stratify_key(meta: vindr.ImageMeta) -> str:
    """Stratification key: the rarest pathology annotated on the image.

    Keying on the rarest class is what keeps Atelectasis (0.1% of QA pairs) and
    Consolidation (0.3%) present in validation at all; keying on the most
    frequent one would let them land entirely in train.
    """
    rank = {name: i for i, name in enumerate(PATHOLOGIES)}
    present = {b.class_name for b in meta.boxes if b.class_name in rank}
    return max(present, key=lambda c: rank[c]) if present else "__none__"


def make_splits(metas: dict[str, vindr.ImageMeta], image_ids: list[str],
                val_ratio: float, test_size: int, seed: int,
                disjoint_test: bool) -> tuple[dict[str, str], list[str]]:
    """Stratified image-level split.

    Table 3 reports 3,735 train (85%) / 659 validation (15%) / 300 test images.
    Since 3,735 + 659 is exactly the 4,394 images in the release, the 300-image
    test split cannot be disjoint from the other two -- it is a subset of
    validation, which is also the set every number in Tables 4-6 is computed on.
    ``disjoint_test`` carves it out of validation instead, for anyone who wants
    a genuinely held-out set at the cost of matching Table 3.

    Returns ``(image_id -> "train"|"val"|"test", test_image_ids)``.  When
    ``disjoint_test`` is false the test ids also keep their ``"val"`` mapping,
    because they remain part of the validation set.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for iid in image_ids:
        buckets[stratify_key(metas[iid])].append(iid)

    split: dict[str, str] = {}
    val_ids: list[str] = []
    for key in sorted(buckets):
        ids = sorted(buckets[key])
        rng.shuffle(ids)
        n_val = round(len(ids) * val_ratio)
        for i, iid in enumerate(ids):
            if i < n_val:
                split[iid] = "val"
                val_ids.append(iid)
            else:
                split[iid] = "train"

    # Test: stratified subset of validation, proportional per stratum.
    val_by_key: dict[str, list[str]] = defaultdict(list)
    for iid in val_ids:
        val_by_key[stratify_key(metas[iid])].append(iid)
    test_ids: list[str] = []
    for key in sorted(val_by_key):
        ids = sorted(val_by_key[key])
        rng.shuffle(ids)
        take = round(len(ids) * test_size / max(len(val_ids), 1))
        test_ids.extend(ids[:take])
    test_ids = sorted(test_ids)[:test_size]

    if disjoint_test:
        for iid in test_ids:
            split[iid] = "test"
    return split, test_ids


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(cfg) -> dict:
    ds_dir = cfg.paths.dataset_dir
    out_dir = cfg.paths.processed_dir
    os.makedirs(out_dir, exist_ok=True)

    print("[build] reading local VinDr-CXR copy ...")
    metas = vindr.load_images(ds_dir, "train")
    print(f"[build]   {len(metas)} images in the train partition")

    data = fetch_vqa_json(cfg.paths.vqa_json)
    n_qa = sum(len(d["vqa"]) for d in data)
    print(f"[build] {len(data)} images / {n_qa} QA pairs in data_v1.json")

    image_ids = [d["image_id"] for d in data]
    missing = [i for i in image_ids if i not in metas]
    if missing:
        raise SystemExit(
            f"[build] {len(missing)} image_id(s) from data_v1.json are not "
            f"present under {ds_dir}; first few: {missing[:5]}")

    split, test_ids = make_splits(
        metas, image_ids, cfg.split.val_ratio, cfg.split.test_size,
        cfg.split.seed, cfg.split.disjoint_test)
    test_set = set(test_ids)

    width = loc_token_width(LOC_NORM_RANGE)
    fused_cache: dict[str, list[dict]] = {}
    records: list[dict] = []
    diag: Counter[str] = Counter()

    for entry in data:
        iid = entry["image_id"]
        m = metas[iid]
        if iid not in fused_cache:
            fused_cache[iid] = vindr.fuse_boxes(m.boxes, cfg.build.fuse_iou)
        fused = fused_cache[iid]

        for k, qa in enumerate(entry["vqa"]):
            locs = extract_locs(qa["gt_location"])
            if not locs:
                diag["unparseable_gt_location"] += 1
                continue
            pub = locs[0]
            fix, method, consistent = repair_box(
                pub, qa["gt_finding"], m.boxes, fused)
            diag[f"repair:{method}"] += 1
            diag["box_label_consistent" if consistent
                 else "box_label_mismatch"] += 1
            if "<loc_" not in qa["answer"]:
                diag["answer_without_loc"] += 1

            records.append({
                "sample_id": f"{iid}_{k}",
                "image_id": iid,
                "image_path": os.path.relpath(m.path, ds_dir).replace("\\", "/"),
                "split": "test" if iid in test_set else split[iid],
                "in_val": split[iid] == "val",

                "question": qa["question"],
                "answer": qa["answer"],
                "reason": qa["reason"],
                "type": qa["type"],
                "difficulty": qa["difficulty"],
                "gt_finding": qa["gt_finding"],

                # --- as published -----------------------------------------
                "gt_location": qa["gt_location"],
                "bbox_orig": [round(v, 2) for v in pub],
                "bbox_resized": [round(v, 2) for v in to_resized(pub, m.fx, m.fy)],
                "bbox_norm": list(to_normalized(pub, m.width, m.height)),
                "loc_norm": format_loc(to_normalized(pub, m.width, m.height), width),

                # --- repaired: the box that belongs to gt_finding ----------
                "box_label_consistent": consistent,
                "repair_method": method,
                "gt_location_fixed": format_loc(fix),
                "bbox_fixed_orig": [round(v, 2) for v in fix],
                "bbox_fixed_resized": [round(v, 2) for v in to_resized(fix, m.fx, m.fy)],
                "bbox_fixed_norm": list(to_normalized(fix, m.width, m.height)),
                "loc_fixed_norm": format_loc(
                    to_normalized(fix, m.width, m.height), width),

                # --- image geometry ----------------------------------------
                "image_width": m.width,
                "image_height": m.height,
                "resized_width": m.resized_width,
                "resized_height": m.resized_height,
            })

    # ---- per-image index: the consensus boxes used as bbox-eval ground truth
    index: dict[str, dict] = {}
    for iid in image_ids:
        m = metas[iid]
        index[iid] = {
            "image_id": iid,
            "image_path": os.path.relpath(m.path, ds_dir).replace("\\", "/"),
            "split": "test" if iid in test_set else split[iid],
            "in_val": split[iid] == "val",
            "width": m.width, "height": m.height,
            "resized_width": m.resized_width,
            "resized_height": m.resized_height,
            "labels_union": m.labels_union,
            "labels_majority": m.labels_majority,
            # Every individual radiologist reading. Section 3.2 scores 5,471
            # ground-truth boxes over 659 validation images (8.3 per image),
            # which is the raw count -- so this, not `boxes_fused`, is the
            # default ground truth for bounding-box evaluation.
            "boxes_raw": [{
                "class_name": b.class_name,
                "rad_id": b.rad_id,
                "bbox_orig": [round(v, 2) for v in b.bbox],
                "bbox_resized": [round(v, 2) for v in to_resized(b.bbox, m.fx, m.fy)],
                "bbox_norm": list(to_normalized(b.bbox, m.width, m.height)),
            } for b in m.boxes],
            # The same boxes after collapsing the two or three independent
            # readings of each lesion into one consensus box.
            "boxes_fused": [{
                "class_name": f["class_name"],
                "bbox_orig": [round(v, 2) for v in f["bbox"]],
                "bbox_resized": [round(v, 2) for v in to_resized(f["bbox"], m.fx, m.fy)],
                "bbox_norm": list(to_normalized(f["bbox"], m.width, m.height)),
                "votes": f["votes"],
            } for f in fused_cache[iid]],
        }

    # ---- write -------------------------------------------------------------
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
    if not cfg.split.disjoint_test:
        # Test is a view on validation, so val must still contain those samples.
        by_split["val"] = [r for r in records if r["in_val"]]

    written: dict[str, int] = {}
    for name in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{name}.jsonl")
        rows = sorted(by_split.get(name, []), key=lambda r: r["sample_id"])
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[name] = len(rows)
        print(f"[build] wrote {path}  ({len(rows)} QA pairs, "
              f"{len({r['image_id'] for r in rows})} images)")

    idx_path = os.path.join(out_dir, "images.json")
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    print(f"[build] wrote {idx_path}  ({len(index)} images)")

    val_imgs = [v for v in index.values() if v["in_val"]]
    summary = {
        "n_images": len(index),
        "n_qa_pairs": len(records),
        "qa_per_split": written,
        "images_per_split": {
            "train": len({r["image_id"] for r in records
                          if r["split"] == "train"}),
            # Table 3 counts all 659 validation images, including the 300 that
            # also form the test subset.
            "val": len(val_imgs),
            "test": len(test_set),
        },
        "val_gt_boxes_raw": sum(len(v["boxes_raw"]) for v in val_imgs),
        "val_gt_boxes_fused": sum(len(v["boxes_fused"]) for v in val_imgs),
        "val_boxes_per_image": round(
            sum(len(v["boxes_raw"]) for v in val_imgs) / max(len(val_imgs), 1), 2),
        "val_multi_pathology_pct": round(100 * sum(
            1 for v in val_imgs
            if len({b["class_name"] for b in v["boxes_raw"]}) >= 2)
            / max(len(val_imgs), 1), 1),
        "diagnostics": dict(diag),
    }
    with open(os.path.join(out_dir, "build_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--disjoint-test", action="store_true",
                    help="carve the 300 test images out of validation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
    if args.out:
        cfg.paths.processed_dir = args.out
    if args.disjoint_test:
        cfg.split.disjoint_test = True

    print("\n[build] summary")
    print(json.dumps(build(cfg), indent=2))


if __name__ == "__main__":
    main()
