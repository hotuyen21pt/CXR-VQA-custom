"""Render Figure 2: ground truth vs. zero-shot vs. fine-tuned localization.

Ground truth is drawn in green, the pretrained baseline in yellow and the
fine-tuned model in red, matching the paper's colour scheme.  Needs Pillow
only.

    python -m vindr_cxr_vqa.visualize --split val --n 6 \
        --pred-base outputs/preds_val_base.jsonl \
        --pred-ft   outputs/preds_val_ft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from PIL import Image, ImageDraw

from .boxes import Box, extract_locs, iou, to_canvas
from .config import load_config

GREEN, YELLOW, RED = (60, 220, 60), (250, 215, 40), (240, 60, 60)
UPSCALE = 3  # the stored PNGs are 224x224; enlarge so boxes stay legible


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _preds_by_image(path: str | None) -> dict[str, list[str]]:
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    for r in _read_jsonl(path):
        out[r["image_id"]].append(r.get("prediction", ""))
    return dict(out)


def _to_canvas(box: Box, img: dict, space: str, w: int, h: int) -> Box:
    return to_canvas(box, space, img["width"], img["height"], w, h)


def _draw(d: ImageDraw.ImageDraw, box: Box, colour, label: str = "") -> None:
    d.rectangle(box, outline=colour, width=2)
    if label:
        d.text((box[0] + 3, max(0, box[1] - 11)), label, fill=colour)


def render(cfg, split: str, base: str | None, ft: str | None, n: int,
           out_dir: str, space: str) -> list[str]:
    pdir = cfg.paths.processed_dir
    with open(os.path.join(pdir, "images.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    records = _read_jsonl(os.path.join(pdir, f"{split}.jsonl"))

    pb, pf = _preds_by_image(base), _preds_by_image(ft)
    # Prefer images where the fine-tuned model actually emitted a box, since
    # those are the cases the figure is meant to illustrate.
    candidates = sorted({r["image_id"] for r in records},
                        key=lambda i: (-len(extract_locs(" ".join(pf.get(i, [])))),
                                       i))
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    for iid in candidates[:n]:
        img = index[iid]
        path = os.path.join(cfg.paths.dataset_dir, img["image_path"])
        if not os.path.exists(path):
            continue
        pic = Image.open(path).convert("RGB")
        pic = pic.resize((pic.width * UPSCALE, pic.height * UPSCALE),
                         Image.LANCZOS)
        d = ImageDraw.Draw(pic)
        w, h = pic.size

        gts = [tuple(b["bbox_orig"]) for b in img["boxes_fused"]]
        gts_canvas = [_to_canvas(b, img, "pixel", w, h) for b in gts]
        for box, meta in zip(gts_canvas, img["boxes_fused"]):
            _draw(d, box, GREEN, meta["class_name"])

        for texts, colour in ((pb.get(iid, []), YELLOW), (pf.get(iid, []), RED)):
            for t in texts:
                for b in extract_locs(t):
                    px = _to_canvas(b, img, space, w, h)
                    best = max((iou(px, g) for g in gts_canvas), default=0.0)
                    _draw(d, px, colour, f"IoU {best:.2f}")

        out = os.path.join(out_dir, f"{iid}.png")
        pic.save(out)
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--pred-base", default=None)
    ap.add_argument("--pred-ft", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred-space", default=None,
                    choices=("pixel", "norm1000"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    space = args.pred_space or (
        "norm1000" if cfg.data.loc_format == "norm1000" else "pixel")
    out_dir = args.out or os.path.join(cfg.paths.output_dir, "figures")
    files = render(cfg, args.split, args.pred_base, args.pred_ft, args.n,
                   out_dir, space)
    print(f"[viz] wrote {len(files)} figure(s) to {out_dir}")
    print("[viz] green = ground truth, yellow = pretrained, red = fine-tuned")


if __name__ == "__main__":
    main()
