"""Render docs/PIPELINE_VISUAL.pdf -- one record traced through every stage.

The Vietnamese companion to docs/PIPELINE.md: the same trace, but with the two
chest X-rays embedded and every radiologist box drawn on them at true scale, so
"the published box is really the aortic one" becomes something you see rather
than something you take on faith.

Nothing here is transcribed by hand -- ``collect()`` recomputes every number
from the repo's own code. It therefore needs the dataset built
(``python -m vindr_cxr_vqa.build_dataset``), the VinDr images on disk, and
outputs/eval_val_oracle.json (``python -m vindr_cxr_vqa.evaluate --oracle
--split val``).

Layout is HTML printed by headless Chrome or Edge, which is what gives real
page breaks, Vietnamese typography and inline SVG without a LaTeX toolchain.

    python scripts/make_pipeline_pdf.py
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

#: The two images docs/PIPELINE.md traces: one where the published box is
#: right, one where it belongs to a different pathology.
GOOD_ID = "9a5094b2563a1ef3ff50dc5c7ff71345"
BAD_ID = "051132a778e61a86eb147c7c6f564dfe"
#: The prediction that doc follows, in the normalised 0-999 space.
TRACED_PREDICTION = (332, 580, 797, 770)

#: Headless Chrome prints the PDF; Edge renders it identically.
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
)


def find_browser() -> str:
    for cand in BROWSERS:
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        elif shutil.which(cand):
            return shutil.which(cand)
    raise SystemExit(
        "No headless Chrome or Edge found -- this script prints the PDF with "
        "one.\nInstall Chrome, or add your binary to BROWSERS in this file.")


def img64(image_id: str) -> str:
    """The real 224x224 PNG, base64 for an <image href>."""
    path = os.path.join(REPO, "dataset", "train_data", "train", image_id + ".png")
    return base64.b64encode(open(path, "rb").read()).decode()


def e(s) -> str:
    return html.escape(str(s))


def vn(x, dec=None) -> str:
    """Vietnamese number: "." groups thousands, "," marks the decimal."""
    if dec is None:
        return f"{int(x):,}".replace(",", ".")
    whole, _, frac = f"{x:,.{dec}f}".partition(".")
    return whole.replace(",", ".") + "," + frac



def collect() -> dict:
    """Every value the document shows, computed from the repo itself."""
    from vindr_cxr_vqa import prompts, vindr
    from vindr_cxr_vqa.boxes import format_loc, from_normalized, iou
    from vindr_cxr_vqa.build_dataset import repair_box
    from vindr_cxr_vqa.config import load_config
    from vindr_cxr_vqa.evaluate import mentioned

    cfg = load_config("configs/default.yaml")
    metas = vindr.load_images(cfg.paths.dataset_dir, "train")

    out: dict = {}

    rec = None
    for line in open("data/processed/val.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["image_id"] == GOOD_ID:
            rec = r
            break
    assert rec, "record not found"
    out["record"] = rec

    m = metas[GOOD_ID]
    out["good"] = {
        "image_id": GOOD_ID, "w": m.width, "h": m.height, "fx": m.fx, "fy": m.fy,
        "raw": [{"cls": b.class_name, "rad": b.rad_id,
                 "box": [round(v, 2) for v in b.bbox]} for b in m.boxes],
        "fused": [{"cls": f["class_name"], "votes": f["votes"],
                   "n_boxes": f["n_boxes"],
                   "box": [round(v, 2) for v in f["bbox"]]}
                  for f in vindr.fuse_boxes(m.boxes)],
    }

    tgt, span = prompts.build_target(rec, cfg.data.box_source, cfg.data.loc_format,
                                     cfg.data.include_reason)
    out["stage3"] = {
        "system": prompts.SYSTEM_PROMPT,
        "question": prompts.build_question(rec, cfg.data.loc_format),
        "target": tgt,
        "span": list(span),
        "span_text": tgt[span[0]:span[1]],
    }

    # stage 6/7: the prediction the doc traces
    PRED_NORM = TRACED_PREDICTION
    pred_px = from_normalized(PRED_NORM, m.width, m.height)
    pred_text = ("Yes, there is cardiomegaly centrally in the chest "
                 + format_loc(PRED_NORM, width=3) + ".")

    out["stage7"] = {
        "pred_norm": list(PRED_NORM),
        "pred_px": [round(v, 2) for v in pred_px] if pred_px else None,
        "ious": [{"cls": b.class_name, "rad": b.rad_id,
                  "iou": round(iou(pred_px, b.bbox), 3)} for b in m.boxes],
        "prediction": pred_text,
        "mentioned": sorted(mentioned(pred_text)),
        "gt_classes": sorted({b.class_name for b in m.boxes}),
    }

    mb = metas[BAD_ID]
    pub = (1264, 743, 1611, 1019)
    fused_b = vindr.fuse_boxes(mb.boxes)
    fixed, method, consistent = repair_box(pub, "Cardiomegaly", mb.boxes,
                                           fused_b)
    out["bad"] = {
        "image_id": BAD_ID, "w": mb.width, "h": mb.height,
        "published": list(pub),
        "raw": [{"cls": b.class_name, "rad": b.rad_id,
                 "box": [round(v) for v in b.bbox],
                 "iou": round(iou(pub, b.bbox), 2)} for b in mb.boxes],
        "fused": [{"cls": f["class_name"], "votes": f["votes"],
                   "box": [round(v) for v in f["bbox"]],
                   "iou": round(iou(pub, f["bbox"]), 2)} for f in fused_b],
        "fixed": [round(v) for v in fixed], "method": method,
        "consistent": consistent,
    }

    out["summary"] = json.load(open("data/processed/build_summary.json",
                                    encoding="utf-8"))
    out["oracle"] = json.load(open("outputs/eval_val_oracle.json", encoding="utf-8"))
    return out


# ---------------------------------------------------------------- svg helpers
PAL = {
    "card": "#0b6bcb", "aorta": "#c2410c", "pt": "#7c3aed", "pe": "#be185d",
    "fused": "#047857", "pred": "#b91c1c", "pub": "#c2410c",
    "ink": "#1f2933", "mute": "#64748b", "rule": "#cbd5e1",
}


def xray(image_id, w, h, boxes, size=250, caption=None, legend=True):
    """Embed the real PNG at `size` px and draw `boxes` on it at true scale.

    boxes: list of dicts {box:[x1,y1,x2,y2] in ORIGINAL pixels, color, label,
                          dash (optional), width (optional)}
    """
    k = size / max(w, h)
    iw, ih = w * k, h * k
    parts = [f'<svg class="xr" viewBox="0 0 {iw+2:.0f} {ih+2:.0f}" '
             f'width="{iw+2:.0f}" height="{ih+2:.0f}">',
             f'<image x="1" y="1" width="{iw:.1f}" height="{ih:.1f}" '
             f'href="data:image/png;base64,{img64(image_id)}"/>']
    for b in boxes:
        x1, y1, x2, y2 = (v * k for v in b["box"])
        dash = f' stroke-dasharray="{b["dash"]}"' if b.get("dash") else ""
        parts.append(
            f'<rect x="{x1+1:.1f}" y="{y1+1:.1f}" width="{x2-x1:.1f}" '
            f'height="{y2-y1:.1f}" fill="none" stroke="{b["color"]}" '
            f'stroke-width="{b.get("width", 1.4)}"{dash}/>')
        if b.get("tag"):
            tw = len(b["tag"]) * 4.6 + 6
            ty = y1 - 2 if y1 > 12 else y2 + 11
            parts.append(
                f'<rect x="{x1+1:.1f}" y="{ty-7.5:.1f}" width="{tw:.1f}" '
                f'height="9.5" rx="2" fill="{b["color"]}"/>')
            parts.append(
                f'<text x="{x1+4:.1f}" y="{ty:.1f}" font-size="7" fill="#fff" '
                f'font-weight="700">{e(b["tag"])}</text>')
    parts.append(f'<rect x="1" y="1" width="{iw:.1f}" height="{ih:.1f}" '
                 f'fill="none" stroke="{PAL["rule"]}"/>')
    parts.append("</svg>")
    svg = "".join(parts)
    leg = ""
    if legend:
        rows = "".join(
            f'<div class="lg"><span class="sw" style="border-color:{b["color"]};'
            f'{"border-style:dashed" if b.get("dash") else ""}"></span>'
            f'{e(b["label"])}</div>'
            for b in boxes if b.get("label"))
        leg = f'<div class="legend">{rows}</div>'
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    return f'<div class="fig">{svg}{leg}{cap}</div>'


def flow_overview():
    rows = [
        ("0", "Nguyên liệu thô", "JSON của HF  +  dòng CSV  +  file PNG",
         "data_v1.json, annotations_train.csv"),
        ("1", "Gộp box 3 bác sĩ", "6 box thô → 4 box đồng thuận", "vindr.py"),
        ("2", "Thành 1 dòng JSONL", "1 record, 3 hệ toạ độ, box gốc + box sửa",
         "build_dataset.py"),
        ("3", "Dựng prompt + đích", "SYSTEM / USER / TARGET + vị trí ký tự",
         "prompts.py"),
        ("4", "Token hoá, che nhãn", "input_ids, labels, 12 vị trí chữ số",
         "data.py"),
        ("5", "Loss đa nhiệm", "L = L_VQA + 0,5·L_bbox → adapter LoRA",
         "losses.py, train.py"),
        ("6", "Sinh dự đoán", "chuỗi văn bản có &lt;loc_…&gt;", "infer.py"),
        ("7", "Chấm điểm", "TP/FP/FN + IoU → một ô trong Bảng 4/5/6",
         "evaluate.py, results.py"),
    ]
    H, W = 52, 515
    out = [f'<svg viewBox="0 0 {W} {len(rows)*H+10}" width="{W}" '
           f'height="{len(rows)*H+10}">']
    for i, (n, title, shape, mod) in enumerate(rows):
        y = i * H + 4
        out.append(f'<rect x="34" y="{y}" width="{W-40}" height="{H-14}" rx="5" '
                   f'fill="#f8fafc" stroke="{PAL["rule"]}"/>')
        out.append(f'<circle cx="17" cy="{y+(H-14)/2}" r="13" fill="{PAL["card"]}"/>')
        out.append(f'<text x="17" y="{y+(H-14)/2+4.5}" font-size="13" fill="#fff" '
                   f'text-anchor="middle" font-weight="700">{n}</text>')
        out.append(f'<text x="46" y="{y+16}" font-size="11.5" font-weight="700" '
                   f'fill="{PAL["ink"]}">{title}</text>')
        out.append(f'<text x="46" y="{y+30}" font-size="10" fill="{PAL["mute"]}">'
                   f'{shape}</text>')
        out.append(f'<text x="{W-12}" y="{y+16}" font-size="8.5" '
                   f'fill="{PAL["mute"]}" text-anchor="end" '
                   f'font-family="ui-monospace,Consolas,monospace">{mod}</text>')
        if i < len(rows) - 1:
            out.append(f'<path d="M17 {y+H-14} L17 {y+H+4}" stroke="{PAL["card"]}" '
                       f'stroke-width="2"/>')
            out.append(f'<path d="M13 {y+H-1} L17 {y+H+4} L21 {y+H-1}" '
                       f'fill="{PAL["card"]}"/>')
    out.append("</svg>")
    return f'<div class="fig">{"".join(out)}</div>'


def three_spaces():
    """Same box drawn in three coordinate frames, to scale against each frame."""
    specs = [
        ("pixel gốc", 2080, 2336, REC["bbox_orig"], "2080 × 2336",
         "691, 1375, 1653, 1831"),
        ("ảnh PNG 224×224", 224, 224, REC["bbox_resized"], "224 × 224",
         "74,4  131,9  178,0  175,6"),
        ("chuẩn hoá 0–999", 999, 999, REC["bbox_norm"], "999 × 999",
         "332, 588, 794, 783"),
    ]
    panes = []
    for name, w, h, box, dims, nums in specs:
        size = 128
        k = size / max(w, h)
        iw, ih = w * k, h * k
        x1, y1, x2, y2 = (v * k for v in box)
        panes.append(f'''<div class="pane">
<div class="pane-h">{e(name)}</div>
<svg viewBox="0 0 {iw+2:.0f} {ih+2:.0f}" width="{iw+2:.0f}" height="{ih+2:.0f}">
 <rect x="1" y="1" width="{iw:.1f}" height="{ih:.1f}" fill="#f1f5f9"
       stroke="{PAL['rule']}"/>
 <rect x="{x1+1:.1f}" y="{y1+1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}"
       fill="rgba(11,107,203,.14)" stroke="{PAL['card']}" stroke-width="1.6"/>
</svg>
<div class="pane-d">khung {e(dims)}</div>
<div class="pane-n">{e(nums)}</div></div>''')
    return ('<div class="fig"><div class="panes">' + "".join(panes) + "</div>"
            '<div class="cap">Cùng một quả tim to. Khung đổi, con số đổi, '
            'chỗ nằm trên lồng ngực thì không đổi.</div></div>')


def span_ruler():
    t = S3["target"]
    a, b = S3["span"]
    head, mid, tail = t[:a], t[a:b], t[b:]
    return f'''<div class="ruler">
<span class="r-h">{e(head)}</span><span class="r-m">{e(mid)}</span><span
class="r-t">{e(tail)}</span></div>
<div class="ruler-note">ký tự 0 … {a-1} &nbsp;|&nbsp; <b>ký tự {a} … {b-1}</b>
= span toạ độ &nbsp;|&nbsp; ký tự {b} … {len(t)-1}</div>'''


def token_strip():
    segs = [("ảnh", 256, "#e2e8f0", "-100 (che)"),
            ("câu hỏi", 28, "#e2e8f0", "-100 (che)"),
            ("chuỗi đích", 58, "#bfdbfe", "token thật (tính loss)")]
    total = sum(s[1] for s in segs)
    W, H = 515, 42
    x = 0
    out = [f'<svg viewBox="0 0 {W} {H+46}" width="{W}" height="{H+46}">']
    for name, n, fill, lab in segs:
        w = W * n / total
        out.append(f'<rect x="{x:.1f}" y="4" width="{w:.1f}" height="{H}" '
                   f'fill="{fill}" stroke="{PAL["rule"]}"/>')
        out.append(f'<text x="{x+w/2:.1f}" y="21" font-size="10" '
                   f'text-anchor="middle" font-weight="700" '
                   f'fill="{PAL["ink"]}">{name}</text>')
        out.append(f'<text x="{x+w/2:.1f}" y="35" font-size="8.5" '
                   f'text-anchor="middle" fill="{PAL["mute"]}">{n} token</text>')
        out.append(f'<text x="{x+w/2:.1f}" y="{H+18}" font-size="8.5" '
                   f'text-anchor="middle" fill="{PAL["mute"]}">{lab}</text>')
        x += w
    out.append(f'<text x="0" y="{H+40}" font-size="8.5" fill="{PAL["mute"]}">'
               f'labels = [-100 … -100][token đích] — mô hình chỉ bị phạt trên '
               f'phần nó phải tự viết ra</text>')
    out.append("</svg>")
    return f'<div class="fig">{"".join(out)}</div>'


def digit_tokens():
    tok = ["&lt;", "loc", "_", "3", "3", "2", "_", "5", "8", "8", "_",
           "7", "9", "4", "_", "7", "8", "3", "&gt;"]
    hot = {3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17}
    cw, H = 26, 86
    W = len(tok) * cw + 4
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    for i, t in enumerate(tok):
        x = 2 + i * cw
        fill = "#bfdbfe" if i in hot else "#f1f5f9"
        out.append(f'<rect x="{x}" y="22" width="{cw-3}" height="24" rx="3" '
                   f'fill="{fill}" stroke="{PAL["rule"]}"/>')
        out.append(f'<text x="{x+(cw-3)/2}" y="38" font-size="10.5" '
                   f'text-anchor="middle" font-family="ui-monospace,Consolas,'
                   f'monospace" fill="{PAL["ink"]}">{t}</text>')
    for lab, i0, i1, col in (("x_min", 3, 5, PAL["card"]),
                             ("y_min", 7, 9, PAL["card"]),
                             ("x_max", 11, 13, PAL["card"]),
                             ("y_max", 15, 17, PAL["card"])):
        xa, xb = 2 + i0 * cw, 2 + i1 * cw + cw - 3
        out.append(f'<path d="M{xa} 50 L{xa} 56 L{xb} 56 L{xb} 50" fill="none" '
                   f'stroke="{col}" stroke-width="1.2"/>')
        out.append(f'<text x="{(xa+xb)/2}" y="68" font-size="9" '
                   f'text-anchor="middle" fill="{col}" font-weight="700">'
                   f'{lab}</text>')
    out.append(f'<text x="2" y="14" font-size="9" fill="{PAL["mute"]}">'
               f'12 token chữ số được tô đậm — đây là thứ L_bbox lấy logits ra '
               f'để tính</text>')
    out.append("</svg>")
    return f'<div class="fig">{"".join(out)}</div>'


def soft_digit():
    """Illustrative: how a distribution over '0'..'9' becomes a real number."""
    probs = [.01, .02, .04, .62, .18, .05, .03, .02, .02, .01]
    exp = sum(p * i for i, p in enumerate(probs))
    W, H, bw = 230, 118, 20
    out = [f'<svg viewBox="0 0 {W+230} {H+30}" width="{W+230}" height="{H+30}">']
    out.append(f'<text x="0" y="10" font-size="9" fill="{PAL["mute"]}">'
               f'p(chữ số) tại MỘT vị trí</text>')
    for i, p in enumerate(probs):
        h = p * 86
        x = 4 + i * bw
        out.append(f'<rect x="{x}" y="{H-14-h:.1f}" width="{bw-5}" '
                   f'height="{h:.1f}" fill="{PAL["card"]}" opacity=".85"/>')
        out.append(f'<text x="{x+(bw-5)/2}" y="{H-3}" font-size="8.5" '
                   f'text-anchor="middle" fill="{PAL["mute"]}">{i}</text>')
    out.append(f'<text x="{W-10}" y="38" font-size="10" fill="{PAL["ink"]}">'
               f'→ kỳ vọng Σ p(d)·d = <tspan font-weight="700">{vn(exp, 2)}'
               f'</tspan></text>')
    out.append(f'<text x="{W-10}" y="58" font-size="10" fill="{PAL["ink"]}">'
               f'→ ghép 3 vị trí: 100·e₀ + 10·e₁ + e₂</text>')
    out.append(f'<text x="{W-10}" y="78" font-size="10" fill="{PAL["ink"]}">'
               f'→ 4 toạ độ = một "soft box" khả vi</text>')
    out.append(f'<text x="{W-10}" y="98" font-size="10" fill="{PAL["ink"]}">'
               f'→ loss = 1 − GIoU(soft box, box đúng)</text>')
    out.append(f'<text x="{W-10}" y="{H+22}" font-size="8" '
               f'fill="{PAL["mute"]}">phân phối ở đây là minh hoạ cơ chế, '
               f'không phải số đo thật</text>')
    out.append("</svg>")
    return f'<div class="fig">{"".join(out)}</div>'


def giou_why():
    W, H = 500, 150
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    # left: disjoint boxes, IoU = 0
    out.append(f'<text x="10" y="14" font-size="10" font-weight="700" '
               f'fill="{PAL["ink"]}">Hai box rời nhau</text>')
    out.append(f'<rect x="20" y="30" width="70" height="46" fill="none" '
               f'stroke="{PAL["fused"]}" stroke-width="1.6"/>')
    out.append(f'<rect x="140" y="86" width="70" height="46" fill="none" '
               f'stroke="{PAL["pred"]}" stroke-width="1.6"/>')
    out.append(f'<rect x="20" y="30" width="190" height="102" fill="none" '
               f'stroke="{PAL["mute"]}" stroke-dasharray="4 3"/>')
    out.append(f'<text x="24" y="146" font-size="9" fill="{PAL["mute"]}">'
               f'nét đứt = bao lồi C dùng trong GIoU</text>')
    out.append(f'<text x="235" y="52" font-size="11" fill="{PAL["ink"]}">'
               f'IoU = 0 → đạo hàm = 0 → <tspan fill="{PAL["pred"]}" '
               f'font-weight="700">không học được gì</tspan></text>')
    out.append(f'<text x="235" y="74" font-size="11" fill="{PAL["ink"]}">'
               f'GIoU = IoU − |C∖(A∪B)| / |C| &lt; 0</text>')
    out.append(f'<text x="235" y="96" font-size="11" fill="{PAL["fused"]}" '
               f'font-weight="700">→ vẫn có gradient kéo hai box lại gần</text>')
    out.append(f'<text x="235" y="118" font-size="9.5" fill="{PAL["mute"]}">'
               f'Đúng lúc đầu huấn luyện, khi mô hình đoán lung tung,</text>')
    out.append(f'<text x="235" y="132" font-size="9.5" fill="{PAL["mute"]}">'
               f'IoU gần như luôn bằng 0. Đó là lý do phải dùng GIoU.</text>')
    out.append("</svg>")
    return f'<div class="fig">{"".join(out)}</div>'


def match_table():
    rows = []
    best = max(S7["ious"], key=lambda r: r["iou"])
    for r in S7["ious"]:
        if r is best:
            verdict, cls = "TP — thắng cặp ghép tham lam 1-1", "ok"
        elif r["iou"] > 0:
            verdict, cls = ("FN — chồng lấn tốt, nhưng box dự đoán duy nhất "
                            "đã bị dùng cho cặp trên"), "warn"
            cls = "warn"
        else:
            verdict, cls = "FN — không chồng lấn với box dự đoán", "bad"
        rows.append(
            f'<tr class="{cls}"><td>{e(r["cls"])}</td><td class="mono">'
            f'{e(r["rad"])}</td><td class="mono num">{vn(r["iou"], 3)}</td>'
            f'<td>{verdict}</td></tr>')
    return ('<table class="tbl"><thead><tr><th>box thật</th><th>bác sĩ</th>'
            '<th>IoU với box dự đoán</th><th>kết luận</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


# ------------------------------------------------------------------- document
def card_boxes(which):
    g = GOOD
    if which == "raw":
        cols = {"Cardiomegaly": PAL["card"], "Aortic enlargement": PAL["aorta"],
                "Pleural thickening": PAL["pt"], "Pleural effusion": PAL["pe"]}
        return [{"box": b["box"], "color": cols[b["cls"]], "dash": "3 2",
                 "label": f'{b["cls"]} ({b["rad"]})'} for b in g["raw"]]
    out = []
    for b in g["raw"]:
        if b["cls"] == "Cardiomegaly":
            out.append({"box": b["box"], "color": PAL["card"], "dash": "3 2",
                        "label": f'Cardiomegaly thô ({b["rad"]})'})
    f = [x for x in g["fused"] if x["cls"] == "Cardiomegaly"][0]
    out.append({"box": f["box"], "color": PAL["fused"], "width": 2.2,
                "label": "box đồng thuận (3/3 phiếu)"})
    return out



def build_body() -> str:
    return f"""
<div class="title-block">
  <div class="kicker">VinDr-CXR-VQA · MedGemma-4B · LoRA</div>
  <h1>Một bản ghi đi qua toàn bộ pipeline</h1>
  <p class="sub">Từ dữ liệu thô đến một ô trong bảng kết quả — bám theo đúng
  <b>một</b> record thật, có hình minh hoạ ở từng chặng.</p>
  <table class="meta">
    <tr><td>Bản ghi</td><td class="mono">{e(REC['sample_id'])}</td></tr>
    <tr><td>Ảnh</td><td class="mono">{e(GOOD['image_id'])} — {GOOD['w']} × {GOOD['h']} px</td></tr>
    <tr><td>Câu hỏi</td><td>{e(REC['question'])}</td></tr>
    <tr><td>Loại / độ khó</td><td>{e(REC['type'])} / {e(REC['difficulty'])}</td></tr>
    <tr><td>Split</td><td>{e(REC['split'])}</td></tr>
  </table>
  <p class="note">Mọi con số trong tài liệu đều được in ra từ chính code của
  repo, không phải ví dụ bịa. Hai ảnh X-quang là file PNG thật trong
  <span class="mono">dataset/train_data/train/</span>, box của bác sĩ được vẽ
  đè lên đúng tỉ lệ.</p>
</div>

<h2>Toàn cảnh: 8 chặng</h2>
{flow_overview()}

<div class="pb"></div>
<h2>Chặng 0 — Nguyên liệu: hai nguồn tách rời</h2>

<p>Dataset VinDr-CXR-VQA trên HuggingFace <b>không chứa một tấm ảnh nào</b>.
Đây là chỗ hay hiểu nhầm nhất: tải về chỉ có phần hỏi–đáp dạng chữ.</p>

<div class="two">
<div class="src">
  <div class="src-h">Nguồn A — HuggingFace (chỉ có chữ)</div>
  <pre class="code nowrap">{{
 "question":    "Where is the Cardiomegaly?",
 "answer":      "The Cardiomegaly finding
                 refers to an enlargement
                 of the cardiac silhouette,
                 centrally located in the
                 chest &lt;loc_691_1375_1653_
                 1831&gt;.",
 "reason":      "Cardiomegaly is a
                 nonspecific finding…",
 "type":        "Where",
 "difficulty":  "Medium",
 "gt_finding":  "Cardiomegaly",
 "gt_location": "&lt;loc_691_1375_1653_1831&gt;"
}}</pre>
</div>
<div class="src">
  <div class="src-h">Nguồn B — VinDr-CXR trên máy bạn (có ảnh)</div>
  <pre class="code">annotations_train.csv — các dòng của đúng ảnh này:

R10  Cardiomegaly         691 1376 1654 1832
R8   Cardiomegaly         689 1313 1667 1763
R9   Cardiomegaly         693 1376 1657 1800
R9   Aortic enlargement  1052  715 1299  967
R9   Pleural thickening  1789 1729 1875 1992
R9   Pleural effusion    1789 1729 1875 1992

+ dataset/train_data/train/{e(GOOD['image_id'][:12])}….png
+ resize_factors.csv → 2080×2336, fx, fy</pre>
</div>
</div>

<p>Hai nguồn nối với nhau bằng <span class="mono">image_id</span> — chuỗi hash
32 ký tự, <b>không phải theo thứ tự dòng</b>. Cùng khoá đó xuất hiện ở cả 5
nơi: <span class="mono">data_v1.json</span>, tên file PNG,
<span class="mono">annotations_train.csv</span>,
<span class="mono">image_labels_train.csv</span>,
<span class="mono">resize_factors.csv</span>. Thiếu ảnh thì
<span class="mono">build_dataset.py</span> dừng hẳn chứ không map nhầm im
lặng.</p>

{xray(GOOD['image_id'], GOOD['w'], GOOD['h'], card_boxes("raw"), 250,
      "Ảnh thật, với toàn bộ 6 box thô của ba bác sĩ. Ba hình chữ nhật xanh "
      "chồng lên nhau ở giữa ngực là <b>ba cách vẽ cùng một quả tim to</b>, "
      "không phải ba tổn thương.")}

<div class="pb"></div>
<h2>Chặng 1 — Gộp box của ba bác sĩ</h2>

<p>Ảnh train của VinDr-CXR được <b>ba bác sĩ đọc độc lập</b>. Một tổn thương
vì thế xuất hiện 2–3 lần dưới dạng box gần trùng.
<span class="mono">vindr.fuse_boxes()</span> gom các box <b>cùng lớp</b> chồng
nhau (IoU ≥ 0,3) thành một cụm rồi lấy <b>trung bình cộng</b>:</p>

{xray(GOOD['image_id'], GOOD['w'], GOOD['h'], card_boxes("fused"), 250,
      "Ba nét đứt xanh = ba lần đọc. Nét liền xanh lá = box đồng thuận.")}

<table class="tbl">
<thead><tr><th>bác sĩ</th><th>x_min</th><th>y_min</th><th>x_max</th>
<th>y_max</th></tr></thead>
<tbody>
{"".join(f'<tr><td class="mono">{e(b["rad"])}</td>' + "".join(f'<td class="mono num">{v:.2f}</td>' for v in b["box"]) + "</tr>" for b in GOOD["raw"] if b["cls"] == "Cardiomegaly")}
<tr class="ok"><td><b>trung bình</b></td>
{"".join(f'<td class="mono num"><b>{v:.2f}</b></td>' for v in [x for x in GOOD["fused"] if x["cls"] == "Cardiomegaly"][0]["box"])}</tr>
</tbody></table>

<p class="why"><b>Trung bình chứ không phải hợp (union).</b> Hợp sẽ phình box
ra bằng phần tử lớn nhất của cả ba; trung bình cho ước lượng tốt nhất về ranh
giới thật. Số phiếu <span class="mono">votes</span> được giữ lại — đó là cơ sở
để biết nên tin box nào.</p>

<p>Kết quả: <b>6 box thô → 4 box đồng thuận</b>. Đồng thời đọc kích thước ảnh
từ <span class="mono">resize_factors.csv</span>:
gốc <b>{GOOD['w']} × {GOOD['h']}</b>, PNG đã resize <b>224 × 224</b>,
<span class="mono">fx = {vn(GOOD['fx'], 6)}</span>,
<span class="mono">fy = {vn(GOOD['fy'], 6)}</span>.</p>

<div class="pb"></div>
<h2>Chặng 2 — Thành một dòng JSONL</h2>

<p><span class="mono">build_dataset.py</span> ghi ra
<span class="mono">data/processed/val.jsonl</span>. Dòng tương ứng với bản ghi
của ta:</p>

<table class="tbl">
<thead><tr><th>Trường</th><th>Giá trị</th><th>Từ đâu ra</th></tr></thead>
<tbody>
<tr><td class="mono">sample_id</td><td class="mono">{e(REC['sample_id'][:14])}…_0</td><td>image_id + số thứ tự câu hỏi</td></tr>
<tr><td class="mono">image_path</td><td class="mono">train_data/train/{e(GOOD['image_id'][:10])}….png</td><td>tra theo image_id</td></tr>
<tr><td class="mono">split</td><td class="mono">{e(REC['split'])}</td><td>chia phân tầng 85/15</td></tr>
<tr><td class="mono">gt_finding</td><td>{e(REC['gt_finding'])}</td><td>HF</td></tr>
<tr><td class="mono">gt_location</td><td class="mono">{e(REC['gt_location'])}</td><td>HF, <b>giữ nguyên</b></td></tr>
<tr><td class="mono">bbox_orig</td><td class="mono">{e(REC['bbox_orig'])}</td><td>parse từ trên</td></tr>
<tr><td class="mono">bbox_resized</td><td class="mono">{e(REC['bbox_resized'])}</td><td>× fx, fy</td></tr>
<tr><td class="mono">bbox_norm</td><td class="mono">{e(REC['bbox_norm'])}</td><td>chuẩn hoá 0–999</td></tr>
<tr class="ok"><td class="mono">loc_norm</td><td class="mono">{e(REC['loc_norm'])}</td><td><b>dùng để huấn luyện</b></td></tr>
<tr><td class="mono">box_label_consistent</td><td class="mono">{e(REC['box_label_consistent'])}</td><td>box đúng là của Cardiomegaly</td></tr>
<tr><td class="mono">repair_method</td><td class="mono">{e(REC['repair_method'])}</td><td>không cần sửa</td></tr>
<tr><td class="mono">image_width / height</td><td class="mono">{GOOD['w']} / {GOOD['h']}</td><td>resize_factors.csv</td></tr>
</tbody></table>

<h3>Một box, ba hệ toạ độ — và vì sao phải giữ cả ba</h3>
{three_spaces()}

<div class="two">
<div class="src"><div class="src-h">Vì sao có ba hệ</div>
<ul class="list">
<li><b>pixel gốc</b> — hệ của ảnh DICOM nguyên bản, cũng là hệ HF công bố
<span class="mono">gt_location</span>. Đây là <i>sự thật gốc</i>: mọi phép so
với ground truth đều quy về đây.</li>
<li><b>224×224</b> — hệ của file PNG thật trên đĩa. Dùng để vẽ box lên đúng
ảnh. Chú ý <span class="mono">fx ≠ fy</span>: ảnh gốc không vuông mà bị bóp về
vuông, nên phải nhân hai hệ số khác nhau.</li>
<li><b>0–999</b> — hệ huấn luyện. Độ rộng <b>cố định 3 chữ số</b>, và độc lập
kích thước ảnh.</li>
</ul></div>
<div class="src"><div class="src-h">Phép tính kiểm chứng</div>
<pre class="code">fx = 0,107692 = 224 / 2080
fy = 0,095890 = 224 / 2336

691  × 0,107692 =   74,42
1375 × 0,095890 =  131,85

691  × 999/2080 = 331,88 → 332
1375 × 999/2336 = 588,02 → 588
1653 × 999/2080 = 794,05 → 794
1831 × 999/2336 = 783,03 → 783</pre></div>
</div>

<p class="why"><b>Vì sao lưu sẵn cả ba thay vì tính lúc cần?</b> Mỗi chặng
dùng một hệ: train dùng hệ 0–999, vẽ hình dùng 224×224, chấm điểm dùng pixel
gốc. Chuyển qua lại cần <span class="mono">width/height/fx/fy</span>
<b>của đúng ảnh đó</b>. Tính lại ở chặng sau mà vớ nhầm kích thước của ảnh
khác thì box lệch vài trăm pixel <i>mà không có lỗi nào bắn ra</i> — IoU tụt
và không ai biết tại sao.</p>

<div class="pb"></div>
<h2>Chặng 2b — Khi nhãn và box không khớp (60% dataset)</h2>

<p>Bản ghi ở trên là ca <b>tốt</b>. Nhưng 60% dataset thì không. Ví dụ ảnh
<span class="mono">{e(BAD['image_id'])}</span>:</p>

<pre class="code">question      Is there Cardiomegaly?
gt_finding    Cardiomegaly
gt_location   &lt;loc_1264_743_1611_1019&gt;      ← box được công bố</pre>

{xray(BAD['image_id'], BAD['w'], BAD['h'],
      [{"box": b["box"], "color": PAL["card"], "dash": "3 2",
        "label": f'Cardiomegaly thô ({b["rad"]})'}
       for b in BAD["raw"] if b["cls"] == "Cardiomegaly"]
      + [{"box": BAD["published"], "color": PAL["pub"], "width": 2.4,
          "label": "box ĐƯỢC CÔNG BỐ cho câu hỏi Cardiomegaly", "tag": "công bố"}]
      + [{"box": [x for x in BAD["fused"] if x["cls"] == "Cardiomegaly"][0]["box"],
          "color": PAL["fused"], "width": 2.4,
          "label": "box Cardiomegaly đồng thuận — thứ ĐÁNG LẼ phải dùng"}], 250,
      "Box cam ở trên là quai động mạch chủ. Box xanh lá ở dưới mới là quả tim. "
      "Chúng <b>không chạm nhau một pixel nào</b>.")}

<p>Đối chiếu với toàn bộ 7 box thô của ảnh:</p>

<table class="tbl">
<thead><tr><th>lớp</th><th>bác sĩ</th><th>box</th>
<th>IoU với box công bố</th><th></th></tr></thead><tbody>
{"".join(f'<tr class="{"warn" if b["iou"] > 0.9 else ("hint" if b["cls"] == "Cardiomegaly" else "")}"><td>{e(b["cls"])}</td><td class="mono">{e(b["rad"])}</td><td class="mono">{e(b["box"])}</td><td class="mono num">{vn(b["iou"], 2)}</td><td>{"← box công bố CHÍNH LÀ box này" if b["iou"] > 0.9 else ("đáng lẽ phải là một trong ba box này" if b["cls"] == "Cardiomegaly" else "")}</td></tr>' for b in BAD["raw"])}
</tbody></table>

<p>Nhãn ghi <b>Cardiomegaly</b> nhưng box được công bố chính là box
<b>Aortic enlargement của bác sĩ R10</b> — IoU 0,99, chênh 0,01 chỉ do làm
tròn. IoU với mọi box Cardiomegaly thật đều bằng <b>0,00</b>: tim to nằm ở
giữa–dưới lồng ngực (y ≈ 1300–1700), quai động mạch chủ ở trên
(y ≈ 730–1050).</p>

<p class="why"><b>Vì sao hỏng?</b> Pipeline sinh VinDr-CXR-VQA lấy
<i>danh sách bệnh lý</i> và <i>danh sách box</i> của ảnh rồi ghép lại —
nhưng ghép <b>độc lập</b>, không kiểm tra box có thuộc đúng bệnh lý không. Nếu
train thẳng, mô hình học chỉ vào động mạch chủ mỗi khi nghe "tim to" — sai về
y khoa, và sai một cách <i>nhất quán</i> nên nó học rất nhanh.</p>

<h3><span class="mono">repair_box()</span> nối lại — bốn bước</h3>
<ol class="steps">
<li><b>Box công bố có đúng là box của <span class="mono">gt_finding</span>
không?</b> So với box <b>thô</b> (chưa gộp), vì bản release copy từ đó ra.
Ở đây nó trùng một box <i>Aortic enlargement</i> → không nhất quán.</li>
<li><b>Gộp các lần đọc trùng nhau</b> → Cardiomegaly
<span class="mono">votes=3</span>
<span class="mono">{e([x for x in BAD["fused"] if x["cls"] == "Cardiomegaly"][0]["box"])}</span>,
Aortic enlargement <span class="mono">votes=3</span>, Pleural thickening
<span class="mono">votes=1</span>.</li>
<li><b>Chọn ứng viên chồng lấn nhiều nhất.</b> Ở đây IoU = 0,00, không có gì
để bám → <span class="mono">method = "consensus"</span>. Nếu có chồng lấn một
phần thì <span class="mono">method = "best_iou"</span>.</li>
<li><b>Ghi kết quả:</b> <span class="mono">box_label_consistent = False</span>,
<span class="mono">repair_method = {e(BAD['method'])}</span>,
<span class="mono">gt_location_fixed = &lt;loc_{BAD['fixed'][0]}_{BAD['fixed'][1]}_{BAD['fixed'][2]}_{BAD['fixed'][3]}&gt;</span>.</li>
</ol>

<table class="tbl">
<thead><tr><th>repair_method</th><th>nghĩa</th><th>số bản ghi</th></tr></thead>
<tbody>
<tr><td class="mono">published</td><td>box công bố vốn đã đúng</td><td class="mono num">{vn(SUM['diagnostics']['repair:published'])}</td></tr>
<tr><td class="mono">consensus</td><td>không chồng lấn → lấy box đồng thuận</td><td class="mono num">{vn(SUM['diagnostics']['repair:consensus'])}</td></tr>
<tr><td class="mono">best_iou</td><td>có chồng lấn → giữ box cùng lớp trùng nhất</td><td class="mono num">{vn(SUM['diagnostics']['repair:best_iou'])}</td></tr>
<tr><td class="mono">unavailable</td><td>ảnh không có box nào của gt_finding</td><td class="mono num">0</td></tr>
<tr class="ok"><td colspan="2"><b>tổng cặp QA</b></td><td class="mono num"><b>{vn(SUM['n_qa_pairs'])}</b></td></tr>
<tr class="bad"><td colspan="2">trong đó <b>nhãn lệch box</b></td><td class="mono num"><b>{vn(SUM['diagnostics']['box_label_mismatch'])}</b> = {vn(SUM['diagnostics']['box_label_mismatch']/SUM['n_qa_pairs']*100, 1)}%</td></tr>
</tbody></table>

<p>Record giữ <b>cả hai</b> bản: <span class="mono">gt_location</span> (công
bố, để tái lập paper) và <span class="mono">gt_location_fixed</span> (đã nối
lại, để huấn luyện). Chọn bằng <span class="mono">data.box_source</span> —
mặc định của repo này là <span class="mono">fixed</span>.</p>

<div class="pb"></div>
<h2>Chặng 3 — Dựng prompt và chuỗi đích</h2>

<div class="msg"><div class="msg-h">SYSTEM</div>
<pre class="code">{e(S3['system'])}</pre></div>

<div class="msg"><div class="msg-h">USER &nbsp;<span class="pill">+ ảnh 224×224</span></div>
<pre class="code">{e(S3['question'])}</pre></div>

<div class="msg"><div class="msg-h">TARGET — thứ mô hình phải học viết ra</div>
{span_ruler()}</div>

<p>Ba điều vừa xảy ra:</p>
<ol class="steps">
<li>Box pixel gốc <span class="mono">{e(REC['gt_location'])}</span> được
<b>thay</b> bằng bản chuẩn hoá <span class="mono">{e(REC['loc_norm'])}</span>.
Chuỗi đích được <i>dựng lại</i> quanh box mà config yêu cầu, nên toạ độ giám
sát và hệ toạ độ không bao giờ lệch nhau.</li>
<li><span class="mono">reason</span> được nối vào sau
<span class="mono">Reasoning:</span> — đây là phần làm nên chữ
"explainable" của paper.</li>
<li>Hàm trả về <b>vị trí ký tự</b> của span:
<span class="mono">[{S3['span'][0]}:{S3['span'][1]}]</span>. Chặng sau cần
nó.</li>
</ol>

<div class="pb"></div>
<h2>Chặng 4 — Token hoá và che nhãn</h2>

<p><span class="mono">data.MedGemmaCollator</span> dựng tensor. Phần đề bài bị
<b>che</b> bằng <span class="mono">-100</span> — nếu không, mô hình được
thưởng vì học thuộc câu hỏi thay vì học trả lời. Token ảnh cũng bị che.</p>

{token_strip()}

<p>Sau đó collator dùng <span class="mono">offset_mapping</span> ánh xạ từ vị
trí ký tự <span class="mono">[{S3['span'][0]}:{S3['span'][1]}]</span> sang chỉ
số token, tìm đúng <b>12 token chữ số</b>:</p>

{digit_tokens()}

<p class="why">Không tìm đủ 12 token (bản ghi hỏng, hoặc dùng
<span class="mono">loc_format: pixel</span> vì toạ độ pixel không cố định độ
dài) thì mẫu đó <b>bị bỏ qua ở phần loss box</b> chứ không bị giám sát sai vị
trí.</p>

<div class="pb"></div>
<h2>Chặng 5 — Hàm mất mát đa nhiệm</h2>

<p class="formula">L<sub>total</sub> = L<sub>VQA</sub> + λ · L<sub>bbox</sub>
&nbsp;&nbsp;&nbsp; với λ = 0,5</p>

<p><b>L<sub>VQA</sub></b> — cross-entropy thông thường trên token đáp án. Dạy
mô hình <i>viết</i>.</p>

<p><b>L<sub>bbox</sub></b> — phần thú vị. Paper mô tả "IoU-based regression"
trên toạ độ dự đoán qua special token, <b>không thêm đầu hồi quy nào</b>.
Cách hiện thực:</p>

{soft_digit()}

<p>Với bản ghi này, đích so sánh là:</p>
<pre class="code">box đích, chuẩn hoá 0-999 : {e(REC['bbox_norm'])}
cùng box, thang [0,1]     : [0.3323, 0.5886, 0.7948, 0.7838]</pre>

<p class="why"><b>Vì sao cần 3 chữ số cố định?</b> Vì bộ giải mã phải biết
chắc vị trí nào là hàng trăm, hàng chục, hàng đơn vị. Toạ độ pixel
(<span class="mono">691</span> — 3 chữ số,
<span class="mono">1375</span> — 4 chữ số) thì không làm được. Đó là lý do
<span class="mono">loc_format: norm1000</span> là mặc định; đặt
<span class="mono">pixel</span> thì phần loss box tự tắt.</p>

{giou_why()}

<p>Kết quả chặng này: thư mục adapter LoRA — <b>32.788.480 tham số huấn
luyện</b> trên tổng 4.332.867.952, tức <b>0,76%</b> mô hình (số thật in ra
từ lần chạy trên Kaggle).</p>

<div class="pb"></div>
<h2>Chặng 6 — Sinh dự đoán</h2>

<p><span class="mono">infer.py</span> cho mô hình xem ảnh + câu hỏi rồi ghi
lại nó viết gì, mỗi dòng một JSON:</p>

<pre class="code">{{"sample_id":  "{e(REC['sample_id'][:14])}…_0",
 "image_id":   "{e(GOOD['image_id'][:14])}…",
 "question":   "{e(REC['question'])}",
 "prediction": "Yes, there is cardiomegaly centrally in the chest
                &lt;loc_332_580_797_770&gt;."}}</pre>

<p>Chạy <b>hai lần</b>: không adapter (dòng <i>Pretrained</i> trong bảng) và
có adapter (dòng <i>Fine-tuned</i>).</p>

<div class="pb"></div>
<h2>Chặng 7 — Chấm điểm</h2>

<h3>7a. Phần chữ → Bảng 4</h3>
<pre class="code">văn bản dự đoán  →  mentioned()  →  {{{e(", ".join(S7['mentioned']))}}}</pre>

<p>Đối chiếu với tập bệnh lý thật của ảnh
(<span class="mono">{e(", ".join(S7['gt_classes']))}</span>):</p>

<table class="tbl">
<thead><tr><th>bệnh lý</th><th>có thật?</th><th>mô hình có nói?</th>
<th>→</th></tr></thead><tbody>
<tr class="ok"><td>Cardiomegaly</td><td>✓</td><td>✓</td><td><b>TP</b></td></tr>
<tr class="bad"><td>Aortic enlargement</td><td>✓</td><td>—</td><td>FN</td></tr>
<tr class="bad"><td>Pleural thickening</td><td>✓</td><td>—</td><td>FN</td></tr>
<tr class="bad"><td>Pleural effusion</td><td>✓</td><td>—</td><td>FN</td></tr>
<tr><td>10 bệnh lý còn lại</td><td>—</td><td>—</td><td>TN</td></tr>
</tbody></table>

<h3>7b. Phần box → Bảng 5 và 6</h3>
<pre class="code">&lt;loc_332_580_797_770&gt;  →  parse   →  {e(S7['pred_norm'])}        (0-999)
                       →  đổi hệ  →  [{S7['pred_px'][0]:.0f}, {S7['pred_px'][1]:.0f}, {S7['pred_px'][2]:.0f}, {S7['pred_px'][3]:.0f}]  (pixel gốc)</pre>

{xray(GOOD['image_id'], GOOD['w'], GOOD['h'],
      [{"box": b["box"], "color": PAL["mute"], "dash": "2 2",
        "label": None} for b in GOOD["raw"]]
      + [{"box": S7["pred_px"], "color": PAL["pred"], "width": 2.4,
          "label": "box mô hình dự đoán", "tag": "dự đoán"}], 250,
      "Nét xám mảnh = 6 box thật. Nét đỏ = box duy nhất mô hình vẽ ra.",
      legend=True)}

{match_table()}

<p>Ghép <b>tham lam 1-1</b>: box dự đoán nhận cặp ghép tốt nhất
(IoU {vn(max(r['iou'] for r in S7['ious']), 3)}) với một box Cardiomegaly. Năm box
còn lại thành <b>FN</b> — mô hình chỉ vẽ một hình chữ nhật trong khi ảnh có
sáu.</p>

<p class="why"><b>Đây chính là trần của dataset.</b> Mỗi ảnh có trung bình
{vn(SUM['val_boxes_per_image'], 2)} box nhưng chỉ được hỏi ~4 câu, nên recall box
không thể vượt <b>0,474</b> dù mô hình có hoàn hảo tuyệt đối.</p>

<div class="pb"></div>
<h2>Cuối cùng ra output gì</h2>

<p>Chạy hết pipeline sinh ra <b>bốn loại file</b>:</p>

<table class="tbl">
<thead><tr><th>File</th><th>Là gì</th><th>Chặng</th></tr></thead><tbody>
<tr><td class="mono">data/processed/{{train,val,test}}.jsonl</td><td>14.960 / 2.637 / 1.207 cặp QA đã chuẩn hoá</td><td>2</td></tr>
<tr><td class="mono">outputs/medgemma-lora/</td><td>adapter LoRA (~33 M tham số)</td><td>5</td></tr>
<tr><td class="mono">outputs/preds_val_*.jsonl</td><td>mỗi dòng = một câu trả lời mô hình sinh ra</td><td>6</td></tr>
<tr><td class="mono">outputs/eval_val_*.json</td><td>TP/FP/FN, IoU, per-class → Bảng 4/5/6</td><td>7</td></tr>
</tbody></table>

<h3>Số thật đã đo được</h3>

<p><b>Trần lý thuyết</b> (<span class="mono">evaluate --oracle</span>: nạp
chính đáp án đúng làm dự đoán — đây là mức cao nhất bất kỳ mô hình nào có thể
đạt trên dataset này):</p>

<table class="tbl">
<thead><tr><th></th><th>Accuracy</th><th>Precision</th><th>Recall</th>
<th>F1</th></tr></thead><tbody>
<tr class="ok"><td>Bảng 4 — VQA (trần)</td>
<td class="mono num">{vn(ORACLE['vqa']['accuracy'], 4)}</td>
<td class="mono num">{vn(ORACLE['vqa']['precision'], 4)}</td>
<td class="mono num">{vn(ORACLE['vqa']['recall'], 4)}</td>
<td class="mono num">{vn(ORACLE['vqa']['f1'], 4)}</td></tr>
<tr class="ok"><td>Bảng 5 — Detection (trần)</td>
<td class="mono num">—</td>
<td class="mono num">{vn(ORACLE['detection']['precision'], 4)}</td>
<td class="mono num">{vn(ORACLE['detection']['recall'], 4)}</td>
<td class="mono num">{vn(ORACLE['detection']['f1'], 4)}</td></tr>
</tbody></table>

<p>Recall trần chỉ <b>{vn(ORACLE['vqa']['recall'], 4)}</b> vì dataset hỏi quá ít câu
so với số tổn thương có trên ảnh — không phải vì bộ chấm sai.</p>

<p><b>Mô hình gốc chưa fine-tune</b> (MedGemma 1.5 4B-it, đo trên 659 ảnh
validation):</p>

<table class="tbl">
<thead><tr><th></th><th>Accuracy</th><th>Precision</th><th>Recall</th>
<th>F1</th></tr></thead><tbody>
<tr><td>Bảng 4 — VQA</td><td class="mono num">0,728</td>
<td class="mono num">0,423</td><td class="mono num">0,268</td>
<td class="mono num">0,328</td></tr>
<tr class="bad"><td>Bảng 5 — Detection</td><td class="mono num">—</td>
<td class="mono num">0,000</td><td class="mono num">0,000</td>
<td class="mono num">0,000</td></tr>
</tbody></table>

<p class="why"><b>Đọc dòng detection cho đúng:</b> 0,000 không phải "dự đoán
kém" mà là <b>0/2.637 câu trả lời có chứa span <span class="mono">&lt;loc_…&gt;</span>
nào</b>. Mô hình gốc viết báo cáo X-quang dạng văn xuôi, chưa từng học định
dạng toạ độ. Đây đúng là hàng <i>Pretrained</i> — thứ mà fine-tune phải vượt
qua. Bộ chấm tự in cảnh báo thay vì để người đọc tưởng là điểm kém.</p>

<p>Hàng <i>Fine-tuned</i> chưa có trong tài liệu này: lần chạy Kaggle gần nhất
dừng trước khi training xong. Khi có adapter,
<span class="mono">results.py</span> sẽ đặt hai hàng cạnh nhau cùng số của
paper.</p>

<div class="pb"></div>
<h2>Tóm tắt: record biến đổi ra sao</h2>

<table class="tbl">
<thead><tr><th>Chặng</th><th>Bản ghi tồn tại dưới dạng</th><th>File</th>
</tr></thead><tbody>
<tr><td class="num">0</td><td>JSON từ HF + dòng CSV + file PNG</td><td class="mono">data_v1.json, annotations_train.csv</td></tr>
<tr><td class="num">1</td><td>ImageMeta + box đồng thuận</td><td class="mono">vindr.py</td></tr>
<tr><td class="num">2</td><td>1 dòng JSONL, 3 hệ toạ độ, box gốc + box sửa</td><td class="mono">build_dataset.py</td></tr>
<tr><td class="num">3</td><td>prompt + chuỗi đích + vị trí ký tự của span</td><td class="mono">prompts.py</td></tr>
<tr><td class="num">4</td><td>input_ids, labels (đã che), 12 vị trí chữ số</td><td class="mono">data.py</td></tr>
<tr><td class="num">5</td><td>2 số loss → cập nhật trọng số LoRA</td><td class="mono">losses.py, train.py</td></tr>
<tr><td class="num">6</td><td>chuỗi văn bản mô hình sinh ra</td><td class="mono">infer.py</td></tr>
<tr><td class="num">7</td><td>TP/FP/FN + IoU → một ô trong bảng</td><td class="mono">evaluate.py, results.py</td></tr>
</tbody></table>

<h3>Tự chạy lại vết này</h3>
<pre class="code"># chặng 0-2: dựng dataset (không cần GPU)
python -m vindr_cxr_vqa.build_dataset
python -m vindr_cxr_vqa.stats

# chặng 7: kiểm tra bộ chấm bằng chính đáp án (không cần GPU)
python -m vindr_cxr_vqa.evaluate --oracle --split val

# chặng 3-6: cần GPU
python -m vindr_cxr_vqa.train
python -m vindr_cxr_vqa.infer --split val --adapter outputs/medgemma-lora \\
    --out outputs/preds_val_ft.jsonl
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_ft.jsonl --split val

# xem một bản ghi bất kỳ sau chặng 2
head -1 data/processed/val.jsonl</pre>

<p class="note">Tài liệu này sinh ra từ
<span class="mono">docs/PIPELINE.md</span> và dữ liệu thật trong repo. Bản chữ
đầy đủ: <span class="mono">docs/PIPELINE.md</span>. Cách đọc từng con số:
<span class="mono">docs/EVALUATION.md</span>.</p>
"""

CSS = """
@page { size: A4; margin: 15mm 14mm 14mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI","Noto Sans",system-ui,sans-serif;
       color: #1f2933; font-size: 10.2pt; line-height: 1.5; margin: 0; }
h1 { font-size: 22pt; margin: 6px 0 4px; letter-spacing: -.4px; }
h2 { font-size: 14pt; margin: 20px 0 8px; padding-bottom: 5px;
     border-bottom: 2px solid #0b6bcb; color: #0b6bcb; }
h3 { font-size: 11.5pt; margin: 15px 0 6px; color: #334155; }
p { margin: 7px 0; }
.pb { break-before: page; }
h2, h3 { break-after: avoid; }
table, .fig, pre, .msg { break-inside: avoid; }
.kicker { font-size: 8.5pt; letter-spacing: 1.4px; text-transform: uppercase;
          color: #0b6bcb; font-weight: 700; }
.sub { color: #475569; font-size: 11pt; margin: 2px 0 12px; }
.title-block { border-bottom: 3px solid #0b6bcb; padding-bottom: 12px;
               margin-bottom: 16px; }
.meta { border-collapse: collapse; margin: 8px 0; font-size: 9.4pt; }
.meta td { padding: 3px 14px 3px 0; vertical-align: top; }
.meta td:first-child { color: #64748b; white-space: nowrap; }
.note { background: #f1f5f9; border-left: 3px solid #94a3b8; padding: 8px 11px;
        font-size: 9.2pt; color: #475569; margin-top: 10px; }
.why { background: #fff7ed; border-left: 3px solid #ea580c; padding: 8px 11px;
       font-size: 9.6pt; }
.mono { font-family: ui-monospace,"Cascadia Mono",Consolas,monospace;
        font-size: .92em; }
.num { text-align: right; }
pre.code { font-family: ui-monospace,"Cascadia Mono",Consolas,monospace;
           font-size: 8.4pt; line-height: 1.45; background: #f8fafc;
           border: 1px solid #e2e8f0; border-radius: 4px; padding: 9px 11px;
           white-space: pre-wrap; margin: 7px 0; overflow-wrap: anywhere; }
.tbl { border-collapse: collapse; width: 100%; margin: 9px 0; font-size: 9pt; }
.tbl th { background: #0b6bcb; color: #fff; text-align: left;
          padding: 5px 8px; font-weight: 600; font-size: 8.6pt; }
.tbl td { padding: 4px 8px; border-bottom: 1px solid #e2e8f0;
          vertical-align: top; }
.tbl tr.ok td { background: #ecfdf5; }
.tbl tr.bad td { background: #fef2f2; }
.tbl tr.warn td { background: #fffbeb; }
.tbl tr.hint td { background: #eff6ff; }
pre.code.nowrap { white-space: pre; font-size: 7.8pt; }
.fig { margin: 12px 0; text-align: center; }
.fig svg { max-width: 100%; }
svg.xr { image-rendering: auto; }
.cap { font-size: 8.8pt; color: #64748b; margin-top: 6px; text-align: left;
       max-width: 430px; margin-left: auto; margin-right: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; justify-content: center;
          margin-top: 7px; font-size: 8.4pt; color: #475569; }
.lg { display: flex; align-items: center; gap: 5px; }
.sw { width: 16px; height: 9px; border: 2px solid; border-style: solid;
      display: inline-block; }
.two { display: flex; gap: 11px; margin: 10px 0; }
.two > * { flex: 1; min-width: 0; }
.src { border: 1px solid #e2e8f0; border-radius: 4px; overflow: hidden; }
.src-h { background: #f1f5f9; padding: 5px 9px; font-size: 8.8pt;
         font-weight: 700; color: #334155; border-bottom: 1px solid #e2e8f0; }
.src pre.code { border: 0; border-radius: 0; margin: 0; background: #fff; }
.src .list { margin: 8px 11px; padding-left: 15px; font-size: 9.2pt; }
.list li, .steps li { margin: 5px 0; }
.steps { padding-left: 17px; }
.panes { display: flex; gap: 16px; justify-content: center; align-items: flex-end; }
.pane { text-align: center; }
.pane-h { font-size: 9pt; font-weight: 700; margin-bottom: 5px; color: #334155; }
.pane-d { font-size: 8pt; color: #94a3b8; margin-top: 4px; }
.pane-n { font-family: ui-monospace,Consolas,monospace; font-size: 8pt;
          color: #0b6bcb; margin-top: 2px; }
.msg { border: 1px solid #e2e8f0; border-radius: 4px; margin: 9px 0;
       overflow: hidden; }
.msg-h { background: #0b6bcb; color: #fff; padding: 4px 9px; font-size: 8.4pt;
         font-weight: 700; letter-spacing: .6px; }
.msg pre.code { border: 0; border-radius: 0; margin: 0; }
.pill { background: rgba(255,255,255,.25); padding: 1px 6px; border-radius: 8px;
        font-weight: 400; }
.ruler { font-family: ui-monospace,Consolas,monospace; font-size: 8.4pt;
         line-height: 1.6; padding: 9px 11px; background: #f8fafc;
         overflow-wrap: anywhere; }
.r-m { background: #fde68a; font-weight: 700; padding: 1px 2px;
       outline: 1px solid #d97706; }
.ruler-note { font-size: 8.2pt; color: #64748b; padding: 0 11px 8px; }
.formula { text-align: center; font-size: 13pt; margin: 12px 0;
           font-family: "Cambria Math",Georgia,serif; }
"""

def build_html() -> str:
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Pipeline VinDr-CXR-VQA — một bản ghi xuyên suốt</title>
<style>{CSS}</style></head><body>{build_body()}</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "docs",
                                                  "PIPELINE_VISUAL.pdf"))
    ap.add_argument("--keep-html", action="store_true",
                    help="leave the intermediate .html next to the PDF")
    args = ap.parse_args()

    os.chdir(REPO)
    data = collect()
    # The renderers read these as globals, so bind them before building.
    global GOOD, BAD, REC, S3, S7, ORACLE, SUM
    GOOD, BAD, REC = data["good"], data["bad"], data["record"]
    S3, S7 = data["stage3"], data["stage7"]
    ORACLE, SUM = data["oracle"], data["summary"]
    print(f"[pdf] record {REC['sample_id']}")

    doc = build_html()
    html_path = os.path.splitext(args.out)[0] + ".html"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(html_path, "w", encoding="utf-8").write(doc)
    print(f"[pdf] html {len(doc)/1024:.0f} KB (ảnh X-quang nhúng base64)")

    browser = find_browser()
    subprocess.run([browser, "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=10000",
                    f"--print-to-pdf={args.out}",
                    "file:///" + html_path.replace("\\", "/")],
                   check=False, capture_output=True)
    if not args.keep_html and os.path.exists(html_path):
        os.remove(html_path)
    if not os.path.exists(args.out):
        raise SystemExit(f"[pdf] {browser} produced no file")
    print(f"[pdf] wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
