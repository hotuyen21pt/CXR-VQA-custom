"""Reproduce the dataset statistics of the paper and audit them.

Recomputes every number in Table 3 from the built JSONL files, places it next
to the value printed in arXiv:2511.00504v2, and flags the ones that disagree.
Writes ``outputs/dataset_report.{json,md}``.

Standard library only.

    python -m vindr_cxr_vqa.stats --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

from .config import load_config
from .constants import DIFFICULTIES, PATHOLOGIES, QUESTION_TYPES

#: What the paper / dataset card claim, for side-by-side auditing.
PAPER_CLAIMS = {
    "n_images": 4394,
    "n_qa_pairs": 17597,
    "qa_per_image": 4.0,
    "n_pathologies": 14,
    "n_question_types": 6,
    "train_images": 3735,
    "val_images": 659,
    "test_images": 300,
    "val_gt_boxes": 5471,
    "val_boxes_per_image": 8.3,
    "val_multi_pathology_pct": 54.8,
    "positive_pct": 41.7,
    "negative_pct": 58.3,
    "reason_words_min": 150,
    "reason_words_max": 250,
    "question_type_pct_range": (16.4, 17.0),
    # Dataset-card prevalence per pathology, in percent of QA pairs.
    "pathology_pct": {
        "Cardiomegaly": 28.0, "Pulmonary fibrosis": 14.0, "Lung Opacity": 13.3,
        "Pleural effusion": 13.2, "Aortic enlargement": 10.4,
        "Calcification": 8.1, "Other lesion": 3.7, "Pleural thickening": 2.9,
        "Infiltration": 1.8, "Pneumothorax": 1.7, "Nodule/Mass": 1.2,
        "ILD": 1.1, "Consolidation": 0.3, "Atelectasis": 0.1,
    },
}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compute(processed_dir: str) -> dict:
    splits = {name: read_jsonl(os.path.join(processed_dir, f"{name}.jsonl"))
              for name in ("train", "val", "test")}
    with open(os.path.join(processed_dir, "images.json"), encoding="utf-8") as fh:
        index: dict[str, dict] = json.load(fh)

    # Train + val is the whole release (test is a subset of val).
    allrecs = splits["train"] + splits["val"]
    n_qa = len(allrecs)
    images = {r["image_id"] for r in allrecs}
    val_imgs = [v for v in index.values() if v["in_val"]]

    types = Counter(r["type"] for r in allrecs)
    diffs = Counter(r["difficulty"] for r in allrecs)
    paths = Counter(r["gt_finding"] for r in allrecs)
    reason_words = sorted(len(r["reason"].split()) for r in allrecs)

    def pct(c: Counter, key: str) -> float:
        return round(100 * c[key] / n_qa, 2)

    qpi = Counter()
    per_image: dict[str, int] = defaultdict(int)
    for r in allrecs:
        per_image[r["image_id"]] += 1
    for v in per_image.values():
        qpi[v] += 1

    return {
        "n_images": len(images),
        "n_qa_pairs": n_qa,
        "qa_per_image": round(n_qa / len(images), 2),
        "qa_per_image_distribution": dict(sorted(qpi.items())),
        "n_pathologies": len(paths),
        "n_question_types": len(types),

        "train_images": len({r["image_id"] for r in splits["train"]}),
        "val_images": len(val_imgs),
        "test_images": len({r["image_id"] for r in splits["test"]}),
        "train_qa": len(splits["train"]),
        "val_qa": len(splits["val"]),
        "test_qa": len(splits["test"]),

        "question_type_pct": {t: pct(types, t) for t in QUESTION_TYPES},
        "difficulty_pct": {d: pct(diffs, d) for d in DIFFICULTIES},
        "pathology_pct": {p: pct(paths, p) for p in PATHOLOGIES},
        "pathology_counts": {p: paths[p] for p in PATHOLOGIES},

        "val_gt_boxes_raw": sum(len(v["boxes_raw"]) for v in val_imgs),
        "val_gt_boxes_fused": sum(len(v["boxes_fused"]) for v in val_imgs),
        "val_boxes_per_image": round(
            sum(len(v["boxes_raw"]) for v in val_imgs) / max(len(val_imgs), 1), 2),
        "val_multi_pathology_pct": round(100 * sum(
            1 for v in val_imgs
            if len({b["class_name"] for b in v["boxes_raw"]}) >= 2)
            / max(len(val_imgs), 1), 1),

        # Polarity: the release carries a gt_finding + gt_location on every
        # pair, so there are no negative ("No finding") samples at all.
        "samples_with_finding": sum(1 for r in allrecs if r["gt_finding"]),
        "negative_samples": sum(1 for r in allrecs if not r["gt_finding"]),
        "answers_starting_no": sum(
            1 for r in allrecs if r["answer"].strip().lower().startswith("no")),

        "reason_words_mean": round(sum(reason_words) / len(reason_words), 1),
        "reason_words_min": reason_words[0],
        "reason_words_median": reason_words[len(reason_words) // 2],
        "reason_words_max": reason_words[-1],
        "reason_in_150_250_words": sum(1 for w in reason_words if 150 <= w <= 250),

        "box_label_consistent": sum(1 for r in allrecs if r["box_label_consistent"]),
        "box_label_consistent_pct": round(
            100 * sum(1 for r in allrecs if r["box_label_consistent"]) / n_qa, 1),
        "repair_methods": dict(Counter(r["repair_method"] for r in allrecs)),
    }


# --------------------------------------------------------------------------
def audit(s: dict) -> list[dict]:
    """Compare recomputed statistics against the published claims."""
    c = PAPER_CLAIMS
    rows: list[dict] = []

    def add(name: str, claimed, actual, ok: bool, note: str = "") -> None:
        rows.append({"metric": name, "paper": claimed, "rebuilt": actual,
                     "status": "match" if ok else "MISMATCH", "note": note})

    for key in ("n_images", "n_qa_pairs", "n_pathologies", "n_question_types",
                "train_images", "val_images", "test_images"):
        add(key, c[key], s[key], s[key] == c[key])

    add("qa_per_image", c["qa_per_image"], s["qa_per_image"],
        abs(s["qa_per_image"] - c["qa_per_image"]) <= 0.05)

    lo, hi = c["question_type_pct_range"]
    worst = min(s["question_type_pct"].values()), max(s["question_type_pct"].values())
    add("question type balance (%)", f"{lo}-{hi}",
        f"{worst[0]:.1f}-{worst[1]:.1f}",
        worst[0] >= lo - 0.3 and worst[1] <= hi + 0.3)

    add("val gt boxes", c["val_gt_boxes"], s["val_gt_boxes_raw"],
        abs(s["val_gt_boxes_raw"] - c["val_gt_boxes"]) / c["val_gt_boxes"] < 0.05,
        "within 5%; the exact split seed of the paper is not published")
    add("val boxes / image", c["val_boxes_per_image"], s["val_boxes_per_image"],
        abs(s["val_boxes_per_image"] - c["val_boxes_per_image"]) <= 0.3)
    add("val images with >=2 pathologies (%)", c["val_multi_pathology_pct"],
        s["val_multi_pathology_pct"],
        abs(s["val_multi_pathology_pct"] - c["val_multi_pathology_pct"]) <= 3,
        "the release is 93% multi-pathology; the 54.8% in Table 3 is not "
        "reproducible from the published data")

    add("negative samples (%)", c["negative_pct"],
        round(100 * s["negative_samples"] / s["n_qa_pairs"], 1), False,
        "every released QA pair carries a gt_finding and a gt_location; "
        "there are no 'No finding' samples")

    add("reason length (words)", f"{c['reason_words_min']}-{c['reason_words_max']}",
        f"{s['reason_words_min']}-{s['reason_words_max']} "
        f"(mean {s['reason_words_mean']})",
        s["reason_in_150_250_words"] > 0,
        f"{s['reason_in_150_250_words']} of {s['n_qa_pairs']} pairs fall in "
        "the claimed range")

    add("gt_location belongs to gt_finding (%)", 100.0,
        s["box_label_consistent_pct"], s["box_label_consistent_pct"] >= 99,
        "Section 2.2 claims zero mismatches against the VinDr-CXR source")

    for name, claimed in c["pathology_pct"].items():
        actual = s["pathology_pct"][name]
        add(f"prevalence: {name}", claimed, actual, abs(actual - claimed) <= 0.6)
    return rows


# --------------------------------------------------------------------------
def to_markdown(s: dict, rows: list[dict]) -> str:
    out = ["# VinDr-CXR-VQA — rebuilt dataset report", "",
           "Every number below is recomputed from `data/processed/*.jsonl` and",
           "compared against arXiv:2511.00504v2 and the HuggingFace dataset card.",
           "", "## Audit vs. published claims", "",
           "| Metric | Paper | Rebuilt | Status | Note |",
           "| --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['metric']} | {r['paper']} | {r['rebuilt']} | "
                   f"{r['status']} | {r['note']} |")

    n_mismatch = sum(1 for r in rows if r["status"] != "match")
    out += ["", f"**{len(rows) - n_mismatch} of {len(rows)} checks match; "
            f"{n_mismatch} do not.**", "",
            "## Composition", "",
            "| Attribute | Value |", "| --- | --- |",
            f"| Source images | {s['n_images']:,} |",
            f"| QA pairs | {s['n_qa_pairs']:,} |",
            f"| QA per image | {s['qa_per_image']} |",
            f"| Pathology classes | {s['n_pathologies']} |",
            f"| Question types | {s['n_question_types']} |",
            f"| Train | {s['train_images']:,} images / {s['train_qa']:,} QA |",
            f"| Validation | {s['val_images']:,} images / {s['val_qa']:,} QA |",
            f"| Test (subset of validation) | {s['test_images']:,} images / "
            f"{s['test_qa']:,} QA |",
            f"| Validation GT boxes (raw) | {s['val_gt_boxes_raw']:,} |",
            f"| Validation GT boxes (fused) | {s['val_gt_boxes_fused']:,} |",
            "", "## Question types", "",
            "| Type | % of QA pairs |", "| --- | --- |"]
    for t, v in s["question_type_pct"].items():
        out.append(f"| {t} | {v} |")
    out += ["", "## Pathology prevalence", "",
            "| Pathology | Count | % | Card % |", "| --- | --- | --- | --- |"]
    for p in PATHOLOGIES:
        out.append(f"| {p} | {s['pathology_counts'][p]:,} | "
                   f"{s['pathology_pct'][p]} | "
                   f"{PAPER_CLAIMS['pathology_pct'][p]} |")
    out += ["", "## Ground-truth box integrity", "",
            f"- Published `gt_location` belongs to `gt_finding` in "
            f"**{s['box_label_consistent']:,} / {s['n_qa_pairs']:,} "
            f"({s['box_label_consistent_pct']}%)** of pairs.",
            f"- Repair strategy applied: `{s['repair_methods']}`.",
            "- See `docs/DATA_NOTES.md` for what this means for training.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    s = compute(cfg.paths.processed_dir)
    rows = audit(s)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    jpath = os.path.join(cfg.paths.output_dir, "dataset_report.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump({"statistics": s, "audit": rows}, fh, indent=2)
    mpath = os.path.join(cfg.paths.output_dir, "dataset_report.md")
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(s, rows))

    bad = [r for r in rows if r["status"] != "match"]
    print(f"[stats] {len(rows) - len(bad)}/{len(rows)} checks match")
    for r in bad:
        print(f"[stats] MISMATCH  {r['metric']}: paper={r['paper']} "
              f"rebuilt={r['rebuilt']}")
    print(f"[stats] wrote {jpath}")
    print(f"[stats] wrote {mpath}")


if __name__ == "__main__":
    main()
