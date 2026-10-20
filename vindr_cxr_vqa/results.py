"""Turn two evaluation JSONs into the final Tables 4, 5 and 6.

Takes the baseline and fine-tuned scores produced by
:mod:`vindr_cxr_vqa.evaluate` and renders them side by side, with the published
numbers alongside for reference.

    python -m vindr_cxr_vqa.results \
        --base outputs/eval_test_base.json \
        --ft   outputs/eval_test_ft.json

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os

#: Tables 4 and 5 of arXiv:2511.00504v2, measured on 659 validation images.
PAPER = {
    "vqa": {
        "Pretrained": {"accuracy": 0.558, "precision": 0.421,
                       "recall": 0.827, "f1": 0.558},
        "Fine-tuned": {"accuracy": 0.624, "precision": 0.471,
                       "recall": 0.926, "f1": 0.624},
    },
    "detection": {
        "Pretrained": {"precision": 0.096, "recall": 0.068,
                       "f1": 0.070, "miou_tp": 0.036},
        "Fine-tuned": {"precision": 0.120, "recall": 0.090,
                       "f1": 0.103, "miou_tp": 0.615},
    },
    "localization": {
        "Pretrained": {"iou_ge_0.5_pct": 1.1, "iou_ge_0.3_pct": 8.5,
                       "mean_iou_all_preds": 0.036},
        "Fine-tuned": {"iou_ge_0.5_pct": 22.8, "iou_ge_0.3_pct": 48.6,
                       "mean_iou_all_preds": 0.133},
    },
}

TABLES = (
    ("Table 4 — VQA (text-based lesion classification)", "vqa",
     (("accuracy", "Acc."), ("precision", "Prec."),
      ("recall", "Recall"), ("f1", "F1"))),
    ("Table 5 — Bounding-box detection (IoU >= 0.3)", "detection",
     (("precision", "Prec."), ("recall", "Recall"),
      ("f1", "F1"), ("miou_tp", "mIoU (TP)"))),
    ("Table 6 — Localization quality (per image)", "localization",
     (("iou_ge_0.5_pct", "IoU>=0.5 (%)"),
      ("iou_ge_0.3_pct", "IoU>=0.3 (%)"),
      ("mean_iou_all_preds", "Mean IoU"))),
)


def _load(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value, key: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if key.endswith("_pct") else f"{value:.3f}"


def render(base: dict | None, ft: dict | None, split: str) -> str:
    runs = [("Pretrained", base), ("Fine-tuned", ft)]
    meta = ft or base or {}
    n_images = meta.get("localization", {}).get("n_images", "?")

    out = [f"# VinDr-CXR-VQA — results on the {split} split", "",
           f"Scored over {n_images} images "
           f"(`{meta.get('source', '?')}`, coordinate space "
           f"`{meta.get('pred_space', '?')}`).", "",
           "Paper columns are from arXiv:2511.00504v2, which reports on its "
           "659-image validation split — comparable in protocol, not in "
           "sample.", ""]

    for title, section, cols in TABLES:
        out += [f"## {title}", "",
                "| Model | " + " | ".join(label for _, label in cols)
                + " | " + " | ".join(f"{label} (paper)" for _, label in cols)
                + " |",
                "| --- " * (1 + 2 * len(cols)) + "|"]
        for name, res in runs:
            got = [_fmt(res[section].get(k) if res else None, k)
                   for k, _ in cols]
            want = [_fmt(PAPER[section][name].get(k), k) for k, _ in cols]
            out.append(f"| {name} | " + " | ".join(got) + " | "
                       + " | ".join(want) + " |")
        if base and ft:
            deltas = []
            for k, _ in cols:
                b, f = base[section].get(k), ft[section].get(k)
                deltas.append("—" if b is None or f is None
                              else f"{f - b:+.3f}" if not k.endswith("_pct")
                              else f"{f - b:+.1f}")
            out.append("| **Δ** | " + " | ".join(deltas) + " | "
                       + " | ".join("" for _ in cols) + " |")
        out.append("")

    if base and ft:
        b, f = base["vqa"]["f1"], ft["vqa"]["f1"]
        rel = 100 * (f - b) / b if b else 0.0
        out += [f"**VQA F1: {b:.3f} -> {f:.3f} ({rel:+.1f}%).** "
                f"The paper reports 0.558 -> 0.624 (+11.8%).", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="eval JSON for the zero-shot baseline")
    ap.add_argument("--ft", help="eval JSON for the fine-tuned model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base, ft = _load(args.base), _load(args.ft)
    if base is None and ft is None:
        raise SystemExit("pass --base and/or --ft with existing eval JSONs")
    if base is None:
        print(f"[results] no baseline at {args.base}; reporting fine-tuned only")
    if ft is None:
        print(f"[results] no fine-tuned run at {args.ft}; baseline only")

    text = render(base, ft, args.split)
    print("\n" + text)

    out = args.out or os.path.join("outputs", f"results_{args.split}.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[results] wrote {out}")


if __name__ == "__main__":
    main()
