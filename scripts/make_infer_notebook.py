"""Generate notebooks/kaggle_infer.ipynb from the cell sources below.

The training notebook's last cell prints the paths this one asks for: the code
bundle, the images, the base weights and the LoRA adapter it just wrote.  Paste
those four in and this notebook loads the model once, then answers a single
image, an image and question of your own, and finally a whole split with the
same scorer the paper's tables come from.

    python scripts/make_infer_notebook.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import notebook_cells as nc  # noqa: E402

OUT = os.path.join(nc.REPO, "notebooks", "kaggle_infer.ipynb")

MD_INTRO = """\
# VinDr-CXR-VQA — run a trained MedGemma adapter on Kaggle

Inference only: no training happens here. Give it the paths the training run
(`notebooks/kaggle_train.ipynb`) left behind and it loads the weights **once**,
then

1. answers a few records from a split and draws the predicted box against the
   ground truth,
2. answers an image and a question of your own,
3. generates over a whole split and scores Tables 4-6 on it,
4. renders the Figure 2 overlays.

**The four paths.** Paste them into the cells below; every one can be left
empty, in which case the cell searches the attached inputs instead.

| Variable | What to paste | Where it comes from |
|---|---|---|
| `BUNDLE` | code + built JSONL | the Kaggle Dataset from `scripts/make_kaggle_bundle.py` |
| `IMAGES` | the chest X-rays | the same dataset, or the `--part images` one |
| `MODEL_PATH` | MedGemma base weights | *Add Input → Models* |
| `ADAPTER_PATH` | the LoRA adapter | the training notebook's output — `medgemma-lora/` or `medgemma-lora.zip` |

`ADAPTER_PATH` accepts a directory, one of its `checkpoint-N` subdirectories,
or the `.zip` the training notebook saves (it is unzipped for you). Set it to
`None` to run the **zero-shot baseline** — the same model without the adapter,
which is the "Pretrained" row of the paper's tables.

**Settings.** Accelerator → any GPU; on 16 GB (T4/P100) the base weights are
loaded in 4-bit, which needs `bitsandbytes`. Internet on or off is detected
automatically, and with it off everything — code, images, base weights,
adapter — must be attached.

> **Import this notebook before attaching your inputs.** Kaggle applies a
> notebook's stored settings on import; this one stores none, but an import on
> top of a configured notebook can still clear the panel.

**Cost.** Loading the weights takes a few minutes; after that a single answer
is a couple of seconds, and a full 2,637-pair validation pass is 1.5-2 h on a
4-bit T4. `N_EVAL` in section 5 caps it."""

PATHS = '''\
import os, sys, glob, json, shutil, subprocess

# ---------------------------------------------------------------------------
# Paths from the training run. Leave empty to search the attached inputs.
#   BUNDLE  the code (and the built JSONL, unless it is in a separate input)
#   IMAGES  the chest X-rays, when they were attached as their own dataset
# e.g. "/kaggle/input/datasets/<user>/vqa-repo"
BUNDLE = ""
IMAGES = ""
# ---------------------------------------------------------------------------

'''

#: The training notebook only runs the package through its CLIs, so `def run`
#: is enough for it. This one imports `load_processor` and `answer` to keep one
#: model in memory across cells, which an older bundle does not have.
EXTRA_NEEDS = '''\
NEEDS["infer"] = ["def run", "def answer", "def load_processor"]

'''

SETUP = PATHS + nc.FINDERS + nc.CODE_CHECK + EXTRA_NEEDS + nc.RESOLVE

MD_ADAPTER = """\
### The adapter

What training produced. `/kaggle/working` is wiped when a session ends, so the
adapter is normally reached one of two ways:

* **the training notebook's output** — *Add Input → Notebook Output*, which
  gives a path like `/kaggle/input/<notebook-slug>/medgemma-lora.zip` or a
  `medgemma-lora/` directory;
* **a Dataset** you uploaded the downloaded `medgemma-lora.zip` to.

Either works. A `checkpoint-N` directory works too — that is how you compare
step 300 against step 600 without retraining. `ADAPTER_PATH = None` skips the
adapter entirely and gives the zero-shot baseline."""

ADAPTER = '''\
# ---------------------------------------------------------------------------
# The LoRA adapter to load. Accepts a directory, a checkpoint-N inside one, or
# the .zip the training notebook writes. "" searches the attached inputs;
# None runs the zero-shot baseline (no adapter at all).
ADAPTER_PATH = ""
# ---------------------------------------------------------------------------

WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")

def is_adapter(path):
    """A directory PEFT can load: a config plus weights next to it."""
    return (os.path.isdir(path)
            and os.path.exists(os.path.join(path, "adapter_config.json"))
            and any(os.path.exists(os.path.join(path, f))
                    for f in WEIGHT_FILES))

def step_of(path):
    """The N in checkpoint-N, or -1 for anything else."""
    name = os.path.basename(path.rstrip("/"))
    tail = name.rsplit("-", 1)[-1]
    return int(tail) if name.startswith("checkpoint-") and tail.isdigit() else -1

def newest_checkpoint(path):
    """The highest-numbered checkpoint-N under `path`, when the run dir itself
    holds no weights -- an interrupted session saves checkpoints but never
    writes the final adapter."""
    cks = [p for p in glob.glob(os.path.join(path, "checkpoint-*"))
           if is_adapter(p)]
    return max(cks, key=step_of) if cks else None

def unzip_adapter(path, dest="/kaggle/working/adapter"):
    import zipfile
    shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)
    print(f"unzipped {path} -> {dest}")
    # make_archive zips the *contents*, but a hand-made zip may hold a folder.
    return dest if is_adapter(dest) or newest_checkpoint(dest) else next(
        (os.path.join(dest, d) for d in sorted(os.listdir(dest))
         if is_adapter(os.path.join(dest, d))), dest)

def find_adapter(root="/kaggle/input"):
    """An attached adapter directory, or a zip holding one."""
    hits = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "train"]
        if "adapter_config.json" in files and is_adapter(dirpath):
            hits.append(dirpath)
    if hits:
        # Prefer a final adapter over a checkpoint; among checkpoints, the
        # furthest-trained one. Shallowest path breaks any remaining tie.
        return sorted(hits, key=lambda p: (step_of(p) >= 0, -step_of(p),
                                           len(p)))[0]
    import zipfile
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "train"]
        for name in files:
            if not name.endswith(".zip"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with zipfile.ZipFile(path) as z:
                    if any(n.endswith("adapter_config.json")
                           for n in z.namelist()):
                        return unzip_adapter(path)
            except zipfile.BadZipFile:
                continue
    return None

if ADAPTER_PATH is None:
    print("no adapter: zero-shot baseline (the paper's 'Pretrained' row)")
else:
    if ADAPTER_PATH.endswith(".zip"):
        ADAPTER_PATH = unzip_adapter(ADAPTER_PATH)
    elif not ADAPTER_PATH:
        ADAPTER_PATH = find_adapter()
        if ADAPTER_PATH:
            print("auto-detected adapter:", ADAPTER_PATH)
    if ADAPTER_PATH and not is_adapter(ADAPTER_PATH):
        nested = newest_checkpoint(ADAPTER_PATH)
        if not nested:
            raise SystemExit(
                f"{ADAPTER_PATH} is not a LoRA adapter: no adapter_config.json "
                f"+ {' / '.join(WEIGHT_FILES)} in it, and no checkpoint-N "
                f"under it.\\n\\nĐang attach:\\n{tree()}\\n\\n"
                "Attach the training notebook's output (Add Input -> Notebook "
                "Output), or set ADAPTER_PATH = None for the zero-shot run.")
        print(f"{ADAPTER_PATH} holds no final adapter; using {nested}")
        ADAPTER_PATH = nested
    if not ADAPTER_PATH:
        raise SystemExit(
            "No adapter found under /kaggle/input.\\n\\n"
            f"Đang attach:\\n{tree()}\\n\\n"
            "Attach the training notebook's output, paste its path into "
            "ADAPTER_PATH, or set ADAPTER_PATH = None to run zero-shot.")

if ADAPTER_PATH:
    print("adapter  :", ADAPTER_PATH)
    acfg = json.load(open(os.path.join(ADAPTER_PATH, "adapter_config.json")))
    print(json.dumps({k: acfg.get(k) for k in
                      ("base_model_name_or_path", "r", "lora_alpha",
                       "lora_dropout", "target_modules")}, indent=2,
                     default=list))
    size = sum(os.path.getsize(os.path.join(ADAPTER_PATH, f))
               for f in os.listdir(ADAPTER_PATH)
               if os.path.isfile(os.path.join(ADAPTER_PATH, f)))
    print(f"size     : {size / 2**20:.1f} MB")'''

CONFIG = '''\
# One config for this session: the attached data, the attached weights, and a
# batch size the GPU can hold. Nothing here trains, so only the loading knobs
# of the train section are used.
import torch, yaml

CFG_PATH = os.path.join(WORK, "configs", "kaggle_infer.yaml")
with open(os.path.join(WORK, "configs", "default.yaml")) as fh:
    cfg = yaml.safe_load(fh)

cfg["paths"]["dataset_dir"]   = os.path.join(IMAGES_ROOT, "dataset")
cfg["paths"]["processed_dir"] = os.path.join(PROCESSED_ROOT, "data", "processed")
cfg["paths"]["output_dir"]    = "/kaggle/working/outputs"
cfg["model"]["name"] = MODEL_PATH

# Generation needs far less memory than training, so quantize only to fit:
# 4-bit below 24 GB, full precision above it. Keep this the same as the run
# that produced the adapter if you are comparing numbers -- 4-bit weights move
# the answers a little.
VRAM = (torch.cuda.get_device_properties(0).total_memory / 2**30
        if torch.cuda.is_available() else 0)
FOUR_BIT = VRAM < 24
INFER_BATCH = 16 if VRAM >= 40 else (8 if VRAM >= 24 else 2)
cfg["train"]["load_in_4bit"] = FOUR_BIT
print(f"{VRAM:.0f} GB: 4-bit={FOUR_BIT}, batch {INFER_BATCH}, "
      f"max_new_tokens {cfg['eval']['max_new_tokens']}")

with open(CFG_PATH, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)

print(json.dumps({"paths": cfg["paths"], "model": cfg["model"],
                  "data": cfg["data"], "generate": cfg["generate"]}, indent=2))
assert os.path.isdir(cfg["paths"]["dataset_dir"]), cfg["paths"]["dataset_dir"]
assert os.path.isdir(cfg["paths"]["processed_dir"]), cfg["paths"]["processed_dir"]'''

MD_PREFLIGHT = """\
### Preflight

Checks package versions, the GPU, the attached weights and the dataset files
in one pass, so a missing piece surfaces now rather than after the four
minutes it takes to load the weights. It exits non-zero if anything is
blocking."""

MD_LOAD = """\
## 2. Load the model once

Reading 8 GB of weights takes a few minutes, so it happens here and every cell
below reuses `model` and `processor`. If you change `ADAPTER_PATH`, re-run
this cell — and only this one."""

LOAD = '''\
import time
from vindr_cxr_vqa.config import load_config
from vindr_cxr_vqa import infer as vqa_infer

CFG = load_config(CFG_PATH)

t0 = time.time()
processor = vqa_infer.load_processor(CFG, ADAPTER_PATH)
model = vqa_infer.load_model(CFG, ADAPTER_PATH, FOUR_BIT)
print(f"\\nloaded in {time.time() - t0:.0f}s on "
      f"{next(model.parameters()).device}, "
      f"{'with' if ADAPTER_PATH else 'without'} adapter")'''

MD_DEMO = """\
## 3. One record, end to end

The smallest real test: take records straight out of a split, ask the model
its own question about them, and draw what came back.

* **green** — the fused ground-truth boxes for that image, every finding;
* **red** — each `<loc_...>` span the model emitted, labelled with its best
  IoU against the green boxes.

A prediction with no `<loc_...>` in it draws nothing — common for the zero-shot
baseline, which has not learned the span format."""

DEMO = '''\
SPLIT  = "test"     # train / val / test
N_DEMO = 3          # how many records to answer here
PICK   = "first"    # "first" or "random"

import random
from PIL import Image, ImageDraw
from IPython.display import display
from vindr_cxr_vqa import boxes as B
from vindr_cxr_vqa.infer import answer, load_split, open_image

records = load_split(CFG, SPLIT)
with open(os.path.join(CFG.paths.processed_dir, "images.json"),
          encoding="utf-8") as fh:
    INDEX = json.load(fh)

# Which space the model was told to answer in; the boxes mean nothing without
# it, since <loc_300_...> is a different place in pixels than in 0-999.
SPACE = "norm1000" if CFG.data.loc_format == "norm1000" else "pixel"
UPSCALE = 3        # the stored PNGs are 224x224; enlarge so boxes stay legible
GREEN, RED = (60, 220, 60), (240, 60, 60)

def draw(image, prediction, image_id=None):
    """The X-ray with ground truth in green and every predicted box in red."""
    pic = image.resize((image.width * UPSCALE, image.height * UPSCALE),
                       Image.LANCZOS)
    d, (w, h) = ImageDraw.Draw(pic), pic.size
    meta = INDEX.get(image_id or "", {})
    # Ground truth is in original DICOM pixels, so the canvas mapping needs the
    # original size -- not the 224x224 the PNG was resized to.
    iw = meta.get("width", image.width)
    ih = meta.get("height", image.height)

    gts = []
    for box in meta.get("boxes_fused", []):
        canvas = B.to_canvas(box["bbox_orig"], "pixel", iw, ih, w, h)
        gts.append(canvas)
        d.rectangle(canvas, outline=GREEN, width=2)
        d.text((canvas[0] + 3, max(0, canvas[1] - 11)), box["class_name"],
               fill=GREEN)

    ious = []
    for box in B.extract_locs(prediction):
        canvas = B.to_canvas(box, SPACE, iw, ih, w, h)
        best = max((B.iou(canvas, g) for g in gts), default=0.0)
        ious.append(best)
        d.rectangle(canvas, outline=RED, width=2)
        d.text((canvas[0] + 3, max(0, canvas[1] - 11)), f"IoU {best:.2f}",
               fill=RED)
    display(pic)
    return ious

picked = (random.sample(records, N_DEMO) if PICK == "random"
          else records[:N_DEMO])
images = [open_image(CFG, r) for r in picked]

t0 = time.time()
predictions = answer(model, processor, CFG, picked, images)
print(f"{len(picked)} answers in {time.time() - t0:.1f}s "
      f"({(time.time() - t0) / len(picked):.1f}s each)\\n")

for r, pred, img in zip(picked, predictions, images):
    print("image   :", r["image_id"])
    print("Q       :", r["question"])
    print("GT      :", r["gt_finding"], "|", r["gt_location"])
    print("answer  :", pred)
    ious = draw(img, pred, r["image_id"])
    print("boxes   :", f"{len(ious)} predicted, best IoU "
          + (f"{max(ious):.2f}" if ious else "n/a — no <loc_...> emitted"))
    print("-" * 78)'''

MD_ASK = """\
## 4. Your own image and question

The same call with nothing from the dataset behind it: any image file, any
question. This is what using the model outside this repo looks like — no
ground truth, so only the prediction is drawn.

Point `IMAGE` at any `.png`/`.jpg` on the machine (an attached input, or a file
you uploaded). Leaving it empty uses the first record of the split above."""

ASK = '''\
IMAGE    = ""      # e.g. f"{CFG.paths.dataset_dir}/train_data/train/<id>.png"
QUESTION = "What abnormality is present in this chest X-ray, and where is it?"

if not IMAGE:
    IMAGE = os.path.join(CFG.paths.dataset_dir, records[0]["image_path"])
    print("IMAGE is empty; using", IMAGE)
if not os.path.exists(IMAGE):
    raise SystemExit(f"no such image: {IMAGE}")

picture = Image.open(IMAGE).convert("RGB")
t0 = time.time()
# A record is just a question here: build_messages reads nothing else, so an
# image off the street goes through exactly the path the dataset does.
reply = answer(model, processor, CFG, [{"question": QUESTION}], [picture])[0]
print(f"{time.time() - t0:.1f}s\\n")
print(reply, "\\n")

ious = draw(picture, reply)       # no image_id => no ground truth to compare
print(f"{len(ious)} box(es) predicted")'''

MD_SCORE = """\
## 5. Score a whole split

Generation over the split, then the same scorer the paper's tables come from:
Table 4 (VQA), Table 5 (box detection at IoU ≥ 0.3), Table 6 (per-image
localization quality).

`N_EVAL` caps the pairs. Capping makes the run quick but the numbers partial —
and when `SPLIT = "val"` the predictions are also scored against **test**,
which is a subset of validation, so a small cap can leave very few test pairs
in it. Set `N_EVAL = None` for the numbers to be comparable with the paper."""

SCORE = '''\
N_EVAL = 200       # None = the whole split

OUTDIR = CFG.paths.output_dir
os.makedirs(OUTDIR, exist_ok=True)
TAG = "ft" if ADAPTER_PATH else "base"
PRED = f"{OUTDIR}/preds_{SPLIT}_{TAG}.jsonl"

t0 = time.time()
# The model is already in memory from section 2; run() takes it as it is
# rather than reading the weights a second time.
vqa_infer.run(CFG, SPLIT, ADAPTER_PATH, PRED, N_EVAL, INFER_BATCH,
              model=model, processor=processor)
print(f"generation took {(time.time() - t0) / 60:.1f} min")'''

EVAL = '''\
# A val prediction file also covers test, which is a subset of it.
SPLITS = ("val", "test") if SPLIT == "val" else (SPLIT,)

for split in SPLITS:
    r = subprocess.run(
        [sys.executable, "-m", "vindr_cxr_vqa.evaluate", "--config", CFG_PATH,
         "--pred", PRED, "--split", split,
         "--out", f"{OUTDIR}/eval_{split}_{TAG}.json"],
        capture_output=True, text=True)
    print(r.stdout or r.stderr)'''

COMPARE = '''\
# Run this notebook twice -- once with ADAPTER_PATH set, once with it None --
# and both eval files are on disk, so the paper's side-by-side table can be
# rendered. With only one of the two, this cell says so and does nothing.
from IPython.display import Markdown, display

for split in SPLITS:
    base, ft = (f"{OUTDIR}/eval_{split}_base.json",
                f"{OUTDIR}/eval_{split}_ft.json")
    if not (os.path.exists(base) and os.path.exists(ft)):
        print(f"{split}: need both eval_{split}_base.json and "
              f"eval_{split}_ft.json; have "
              f"{[os.path.basename(p) for p in (base, ft) if os.path.exists(p)]}")
        continue
    subprocess.run([sys.executable, "-m", "vindr_cxr_vqa.results",
                    "--base", base, "--ft", ft, "--split", split,
                    "--out", f"{OUTDIR}/results_{split}.md"],
                   capture_output=True, text=True)
    display(Markdown(open(f"{OUTDIR}/results_{split}.md").read()))'''

MD_VIS = """\
## 6. Figure 2 overlays

The same drawing as section 3, done in bulk by `vindr_cxr_vqa.visualize` and
written to PNGs: ground truth green, zero-shot yellow, fine-tuned red. It
picks the images where a box was actually predicted."""

VIS = '''\
flag = "--pred-ft" if ADAPTER_PATH else "--pred-base"
!{sys.executable} -m vindr_cxr_vqa.visualize --config {CFG_PATH} \\
    {flag} {PRED} --split {SPLIT} --n 6 --out {OUTDIR}/figures

from IPython.display import Image as IPyImage
for p in sorted(glob.glob(f"{OUTDIR}/figures/*.png"))[:6]:
    print(p)
    display(IPyImage(filename=p))'''

SAVE = '''\
# /kaggle/working is wiped when the session ends; Save Version keeps whatever
# is listed here as this notebook's output.
for f in sorted(glob.glob(f"{OUTDIR}/*.json") + glob.glob(f"{OUTDIR}/*.md")
                + glob.glob(f"{OUTDIR}/*.jsonl")):
    print(f"  {os.path.basename(f):<28} {os.path.getsize(f) / 2**10:8.1f} KB")

print(f"\\nmodel   : {MODEL_PATH}")
print(f"adapter : {ADAPTER_PATH or 'none (zero-shot)'}")
print(f"split   : {SPLIT}, {'all' if N_EVAL is None else N_EVAL} QA pairs")'''


def build() -> list[dict]:
    md, code = nc.md, nc.code
    return [
        md(MD_INTRO),
        md("## 0. Setup"),
        code(SETUP),
        code(nc.OFFLINE),
        code(nc.INSTALL),
        code(nc.TOKEN),
        md(nc.MD_MODEL),
        code(nc.MODEL),
        md(MD_ADAPTER),
        code(ADAPTER),
        md("## 1. Config"),
        code(CONFIG),
        md(MD_PREFLIGHT),
        code(nc.PREFLIGHT),
        md(MD_LOAD),
        code(LOAD),
        md(MD_DEMO),
        code(DEMO),
        md(MD_ASK),
        code(ASK),
        md(MD_SCORE),
        code(SCORE),
        code(EVAL),
        code(COMPARE),
        md(MD_VIS),
        code(VIS),
        md("## 7. What is on disk"),
        code(SAVE),
    ]


if __name__ == "__main__":
    nc.write(OUT, build())
