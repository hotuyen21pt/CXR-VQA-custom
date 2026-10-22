"""Scoring for Tables 4, 5 and 6 of arXiv:2511.00504v2.

Consumes a predictions JSONL -- one ``{"sample_id", "image_id", "prediction"}``
object per QA pair -- and reports

* **Table 4**  text-based lesion classification: Accuracy, Precision, Recall, F1
* **Table 5**  bounding-box detection at IoU >= 0.3: Precision, Recall, F1,
  and mIoU over true positives only
* **Table 6**  per-image localization quality: share of images reaching
  IoU >= 0.5 ("good") and IoU >= 0.3 ("acceptable"), plus mean IoU over all
  predictions

Standard library only, so results can be scored anywhere the predictions land.

    python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val.jsonl --split val
    python -m vindr_cxr_vqa.evaluate --oracle --split val   # pipeline self-test
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from .boxes import (LOC_RE, Box, extract_locs, from_normalized, greedy_match,
                    iou, match_any)
from .config import load_config
from .constants import LOC_NORM_RANGE, PATHOLOGIES, PATHOLOGY_SYNONYMS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def mentioned(text: str) -> set[str]:
    """Which of the 14 pathologies a generated answer claims to see.

    Matching is on lowercase surface forms (``PATHOLOGY_SYNONYMS``).  A class
    is only credited when the sentence around it is not negated, so that
    "there is no pleural effusion" does not count as a positive prediction.
    """
    low = text.lower()
    found: set[str] = set()
    for cls in PATHOLOGIES:
        for form in PATHOLOGY_SYNONYMS[cls]:
            idx = low.find(form)
            while idx != -1:
                if not _negated(low, idx):
                    found.add(cls)
                    break
                idx = low.find(form, idx + 1)
            if cls in found:
                break
    return found


_NEGATIONS = ("no ", "not ", "without ", "absence of ", "absent ",
              "free of ", "rules out ", "negative for ", "no evidence of ")


def _negated(low: str, idx: int, window: int = 40) -> bool:
    """Whether a negation cue immediately precedes the mention."""
    start = max(0, idx - window)
    ctx = low[start:idx]
    # A clause boundary resets the scope of a negation.
    for sep in (".", ";", " but ", " however ", " although "):
        cut = ctx.rfind(sep)
        if cut != -1:
            ctx = ctx[cut + len(sep):]
    return any(cue in ctx for cue in _NEGATIONS)


def to_pixel_space(boxes: list[Box], img: dict, space: str) -> list[Box]:
    """Bring predicted boxes into original DICOM pixel space."""
    if space == "pixel":
        return boxes
    if space == "norm1000":
        return [from_normalized(b, img["width"], img["height"], LOC_NORM_RANGE)
                for b in boxes]
    if space == "resized":
        fx = img["resized_width"] / img["width"]
        fy = img["resized_height"] / img["height"]
        return [(b[0] / fx, b[1] / fy, b[2] / fx, b[3] / fy) for b in boxes]
    raise SystemExit(f"unknown prediction space: {space!r}")


# --------------------------------------------------------------------------
# Table 4 — text-based lesion classification
# --------------------------------------------------------------------------
def eval_vqa(preds_by_image: dict[str, list[str]],
             index: dict[str, dict]) -> dict:
    """Micro-averaged multi-label scoring over (image x 14 pathologies).

    Ground truth for an image is the set of pathologies a radiologist boxed on
    it; the prediction is the set of pathologies the generated answers claim.

    Note on Table 4: the paper prints the same value in its Accuracy and F1
    columns (0.558 and 0.624). Those two quantities are not equal in general --
    F1 there is exactly 2PR/(P+R) of the printed precision and recall, so the
    Accuracy column appears to be a duplicate of F1. Both are reported here:
    ``f1`` is the comparable number, ``accuracy`` is the genuine
    (TP+TN)/total over all image-class decisions.
    """
    tp = fp = fn = tn = 0
    per_class: dict[str, list[int]] = {c: [0, 0, 0] for c in PATHOLOGIES}
    for iid, texts in preds_by_image.items():
        img = index.get(iid)
        if img is None:
            continue
        gt = {b["class_name"] for b in img["boxes_raw"]} & set(PATHOLOGIES)
        pred = set().union(*(mentioned(t) for t in texts)) if texts else set()
        for cls in PATHOLOGIES:
            g, p = cls in gt, cls in pred
            if g and p:
                tp += 1; per_class[cls][0] += 1
            elif p:
                fp += 1; per_class[cls][1] += 1
            elif g:
                fn += 1; per_class[cls][2] += 1
            else:
                tn += 1
    p, r, f = prf(tp, fp, fn)
    total = tp + fp + fn + tn
    return {
        "n_images": len(preds_by_image),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "per_class": {c: dict(zip(("precision", "recall", "f1"),
                                  (round(v, 4) for v in prf(*per_class[c]))))
                      for c in PATHOLOGIES},
    }


# --------------------------------------------------------------------------
# Tables 5 and 6 — bounding-box detection and localization quality
# --------------------------------------------------------------------------
def eval_boxes(preds_by_image: dict[str, list[str]], index: dict[str, dict],
               thr: float, good: float, acceptable: float,
               space: str, match: str, gt_source: str) -> tuple[dict, dict]:
    """Table 5 (detection) and Table 6 (per-image localization).

    Under Section 3.2's matching (``match="any"``) a true positive is a
    PREDICTION that overlaps some ground-truth box and a false negative is a
    ground-truth box no prediction reached, so ``tp + fn`` is not the number
    of ground-truth boxes and ``tp + fp`` is the number of predictions only
    because every prediction is one or the other.  The box totals are
    therefore counted directly rather than inferred from the confusion
    counts; under ``match="greedy"`` they come out the same.
    """
    matcher = greedy_match if match == "greedy" else match_any
    tp = fp = fn = 0
    n_pred_boxes = n_gt_boxes = 0
    tp_ious: list[float] = []
    all_ious: list[float] = []
    n_good = n_acceptable = 0
    n_images = 0
    gt_key = "boxes_raw" if gt_source == "raw" else "boxes_fused"

    for iid, texts in preds_by_image.items():
        img = index.get(iid)
        if img is None:
            continue
        n_images += 1
        preds: list[Box] = []
        for t in texts:
            preds.extend(extract_locs(t))
        preds = to_pixel_space(preds, img, space)
        gts = [tuple(b["bbox_orig"]) for b in img[gt_key]]
        n_pred_boxes += len(preds)
        n_gt_boxes += len(gts)

        matches, unmatched_p, unmatched_g = matcher(preds, gts, thr)
        tp += len(matches)
        fp += len(unmatched_p)
        fn += len(unmatched_g)
        tp_ious.extend(v for *_, v in matches)

        # Table 6 is per-image and uses the best IoU each prediction achieves,
        # whether or not it survived matching.
        best_per_pred = [max((iou(p, g) for g in gts), default=0.0)
                         for p in preds]
        all_ious.extend(best_per_pred)
        top = max(best_per_pred, default=0.0)
        n_good += top >= good
        n_acceptable += top >= acceptable

    p, r, f = prf(tp, fp, fn)
    detection = {
        "iou_threshold": thr, "match": match, "gt_source": gt_source,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "miou_tp": round(sum(tp_ious) / len(tp_ious), 4) if tp_ious else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "n_predicted_boxes": n_pred_boxes, "n_gt_boxes": n_gt_boxes,
    }
    localization = {
        "n_images": n_images,
        f"iou_ge_{good}_pct": round(100 * n_good / n_images, 2) if n_images else 0.0,
        f"iou_ge_{acceptable}_pct": round(100 * n_acceptable / n_images, 2) if n_images else 0.0,
        "mean_iou_all_preds": round(sum(all_ious) / len(all_ious), 4) if all_ious else 0.0,
    }
    return detection, localization


# --------------------------------------------------------------------------
def diagnose_predictions(by_image: dict[str, list[str]]) -> dict:
    """Why the box metrics look the way they do.

    A detection score of zero has two very different causes: the model never
    emitted a ``<loc_...>`` span at all, or it emitted them in the wrong
    place. Only the second is a model failure; the first is a formatting
    failure, and the numbers mean nothing until it is fixed.
    """
    texts = [t for ts in by_image.values() for t in ts]
    with_loc = [t for t in texts if LOC_RE.search(t)]
    empty = [t for t in texts if not t.strip()]
    lengths = sorted(len(t.split()) for t in texts) or [0]
    return {
        "n_predictions": len(texts),
        "with_loc_span": len(with_loc),
        "with_loc_pct": round(100 * len(with_loc) / len(texts), 1) if texts else 0.0,
        "empty": len(empty),
        "median_words": lengths[len(lengths) // 2],
        "sample": next((t for t in texts if t.strip()), "")[:400],
    }


def load_predictions(path: str,
                     keep: set[str] | None = None) -> tuple[dict[str, list[str]], int]:
    """Group predictions by image, keeping only ``keep`` if given.

    Filtering matters because the test split is a subset of validation: one
    inference pass over ``val`` therefore already contains every ``test``
    prediction, and scoring the smaller split is just a matter of dropping the
    images that are not in it.

    Returns ``(by_image, n_dropped)``.
    """
    by_image: dict[str, list[str]] = defaultdict(list)
    dropped = 0
    for row in read_jsonl(path):
        iid = row["image_id"]
        if keep is not None and iid not in keep:
            dropped += 1
            continue
        by_image[iid].append(row.get("prediction") or row.get("answer") or "")
    return dict(by_image), dropped


def oracle_predictions(records: list[dict], box_source: str) -> dict[str, list[str]]:
    """Feed the ground truth back in as if it were model output.

    A self-test: scoring the reference answers must yield a perfect VQA recall
    and a detection precision of 1.0, which is what proves the matching,
    coordinate conversion and text parsing are wired up correctly.
    """
    key = "gt_location" if box_source == "published" else "gt_location_fixed"
    by_image: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_image[r["image_id"]].append(f"{r['gt_finding']} {r[key]}")
    return dict(by_image)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pred", help="predictions JSONL")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--oracle", action="store_true",
                    help="score the ground truth as predictions (self-test)")
    ap.add_argument("--pred-space", default=None,
                    choices=("pixel", "norm1000", "resized"),
                    help="coordinate space of the predicted <loc_...> spans")
    ap.add_argument("--gt-source", default="raw", choices=("raw", "fused"),
                    help="raw reproduces the 5,471 boxes of Section 3.2")
    ap.add_argument("--match", default=None, choices=("greedy", "any"),
                    help="any (default) is Section 3.2 as published: a "
                         "prediction is a TP when it overlaps ANY ground-truth "
                         "box. greedy is the stricter one-to-one variant.")
    ap.add_argument("--box-source", default=None,
                    choices=("published", "fixed"),
                    help="which ground-truth box the oracle replays")
    ap.add_argument("--strict", action="store_true",
                    help="score images with no prediction as complete misses")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pdir = cfg.paths.processed_dir
    records = read_jsonl(os.path.join(pdir, f"{args.split}.jsonl"))
    with open(os.path.join(pdir, "images.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    if args.oracle:
        box_source = args.box_source or cfg.data.box_source
        preds = oracle_predictions(records, box_source)
        space, tag = "pixel", f"oracle:{box_source}"
        covered = len(preds)
        split_ids = set(preds)
    else:
        if not args.pred:
            raise SystemExit("pass --pred <predictions.jsonl> or --oracle")
        split_ids = {r["image_id"] for r in records}
        preds, dropped = load_predictions(args.pred, keep=split_ids)
        if dropped:
            print(f"[eval] ignored {dropped} prediction(s) outside the "
                  f"{args.split} split")
        covered = len(preds)
        if covered < len(split_ids):
            if args.strict:
                # Unpredicted images contribute their ground truth, so their
                # boxes and findings count as misses.
                preds |= {i: [] for i in split_ids if i not in preds}
                print(f"[eval] strict: {len(split_ids) - covered} unpredicted "
                      f"image(s) scored as complete misses")
            else:
                print(f"[eval] NOTE: scoring {covered} of {len(split_ids)} "
                      f"{args.split} images — the unpredicted ones are "
                      "excluded, not counted as misses (use --strict to "
                      "include them)")
        space = args.pred_space or (
            "norm1000" if cfg.data.loc_format == "norm1000" else "pixel")
        tag = os.path.basename(args.pred)

    match = args.match or cfg.eval.match
    vqa = eval_vqa(preds, index)
    det, loc = eval_boxes(
        preds, index, cfg.eval.iou_threshold, cfg.eval.good_localization_iou,
        cfg.eval.acceptable_localization_iou, space, match, args.gt_source)

    diag = diagnose_predictions(preds)
    result = {"source": tag, "split": args.split, "pred_space": space,
              "coverage": {"scored_images": covered,
                           "split_images": len(split_ids),
                           "strict": bool(args.strict)},
              "format": diag,
              "vqa": vqa, "detection": det, "localization": loc}

    print(f"\n=== {tag}  (split={args.split}, space={space}, match={match}) ===")
    print("\nTable 4 -- VQA (text-based lesion classification)")
    print(f"  Accuracy {vqa['accuracy']:.3f}   Precision {vqa['precision']:.3f}"
          f"   Recall {vqa['recall']:.3f}   F1 {vqa['f1']:.3f}")
    print(f"\nTable 5 -- Bounding-box detection (IoU>={det['iou_threshold']}, "
          f"gt={args.gt_source})")
    print(f"  Precision {det['precision']:.3f}   Recall {det['recall']:.3f}"
          f"   F1 {det['f1']:.3f}   mIoU(TP) {det['miou_tp']:.3f}")
    print(f"  {det['n_predicted_boxes']} predicted vs {det['n_gt_boxes']} "
          f"ground-truth boxes")
    print(f"\nTable 6 -- Localization quality (per image, n={loc['n_images']})")
    for k, v in loc.items():
        if k.endswith("_pct"):
            print(f"  {k:<22} {v:.1f}")
    print(f"  mean IoU (all preds)   {loc['mean_iou_all_preds']:.3f}")

    print(f"\nOutput format: {diag['with_loc_span']}/{diag['n_predictions']} "
          f"predictions contain a <loc_...> span ({diag['with_loc_pct']}%), "
          f"median {diag['median_words']} words")
    if diag["with_loc_pct"] < 1.0:
        # Without a single span there is nothing to localize, so Tables 5 and
        # 6 are measuring the prompt, not the model.
        print("  WARNING: essentially no boxes were emitted, so the detection "
              "and localization numbers above are vacuous rather than poor.")
        print(f"  A sample prediction: {diag['sample']!r}")

    out = args.out or os.path.join(
        cfg.paths.output_dir,
        f"eval_{args.split}_{'oracle' if args.oracle else 'pred'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
