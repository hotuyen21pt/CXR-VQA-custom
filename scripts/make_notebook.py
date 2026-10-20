"""Generate notebooks/kaggle_train.ipynb from the cell sources below.

Keeping the notebook generated rather than hand-edited means the code in it
stays readable in diffs and can never ship invalid JSON.

    python scripts/make_notebook.py
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "notebooks", "kaggle_train.ipynb")

MD_INTRO = """\
# VinDr-CXR-VQA — MedGemma-4B-it fine-tuning on Kaggle

Reproduces arXiv:2511.00504v2 — *VinDr-CXR-VQA: A Visual Question Answering
Dataset for Explainable Chest X-ray Analysis with Multi-Task Learning*.

**Before running**

1. **Attach the data.** Upload `vindr-cxr-vqa-kaggle.zip`
   (`python scripts/make_kaggle_bundle.py`) as a Kaggle Dataset and attach it
   via *Add Input*.
2. **Accelerator** → `GPU T4 x2` or `GPU P100`. *Settings → Accelerator*.
3. **Internet** → **On** (the base model is downloaded from HuggingFace).
4. **HF token** → accept the licence for
   [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it),
   then *Add-ons → Secrets* → add `HF_TOKEN`.

**On run time.** The paper trains 3 epochs over ~21k samples in ~36 h on an
A6000 (48 GB). A Kaggle session is capped at 12 h on a 16 GB GPU, so the
defaults below run a single capped pass that finishes well inside one session.
Raise `MAX_STEPS` (or set it to `-1` for the full 3 epochs) and resume across
sessions from the checkpoint in `/kaggle/working` if you want the paper's
exact budget."""

SETUP = '''\
import os, sys, glob, json, shutil, subprocess

# --- locate the attached bundle -------------------------------------------
CANDIDATES = [p for p in glob.glob("/kaggle/input/*")
              if os.path.isdir(os.path.join(p, "vindr_cxr_vqa"))]
if not CANDIDATES:
    raise SystemExit(
        "No attached dataset contains vindr_cxr_vqa/. Use 'Add Input' to "
        "attach the bundle produced by scripts/make_kaggle_bundle.py.\\n"
        f"Seen: {glob.glob('/kaggle/input/*')}")
BUNDLE = CANDIDATES[0]
print("bundle:", BUNDLE)

# /kaggle/input is read-only, so work from a writable copy.
WORK = "/kaggle/working/vindr"
os.makedirs(WORK, exist_ok=True)
for item in ("vindr_cxr_vqa", "configs", "requirements.txt", "docs"):
    src = os.path.join(BUNDLE, item)
    if not os.path.exists(src):
        continue
    dst = os.path.join(WORK, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

os.chdir(WORK)
sys.path.insert(0, WORK)
print("cwd:", os.getcwd())
print(os.listdir(WORK))'''

INSTALL = '''\
# MedGemma is a Gemma 3 model: it needs a recent transformers.
%pip install -q -U "transformers>=4.52" "accelerate>=1.4" "peft>=0.14" \\
    "bitsandbytes>=0.45" "datasets>=3.0" pyyaml

import torch, transformers, peft
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
print("cuda        ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"vram         {total/2**30:.1f} GB total, {free/2**30:.1f} GB free")
    print("bf16         ", torch.cuda.is_bf16_supported())'''

TOKEN = '''\
# google/medgemma-4b-it is gated: accept its licence, then add HF_TOKEN under
# Add-ons -> Secrets.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as exc:
    print("Could not read the HF_TOKEN secret:", exc)
    print("Add it under Add-ons -> Secrets, or set os.environ['HF_TOKEN'] here.")

os.environ.setdefault("HF_HOME", "/kaggle/working/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")'''

CONFIG = '''\
# Point the config at the attached bundle and size the run for a 16 GB GPU.
import torch, yaml

CFG_PATH = os.path.join(WORK, "configs", "kaggle.yaml")
with open(os.path.join(WORK, "configs", "default.yaml")) as fh:
    cfg = yaml.safe_load(fh)

cfg["paths"]["dataset_dir"]   = os.path.join(BUNDLE, "dataset")
cfg["paths"]["processed_dir"] = os.path.join(BUNDLE, "data", "processed")
cfg["paths"]["vqa_json"]      = "/kaggle/working/data_v1.json"
cfg["paths"]["output_dir"]    = "/kaggle/working/outputs"

# The paper uses batch 4 x grad-accum 4 on a 48 GB A6000. A 16 GB T4/P100
# holds one sample at a time, so keep the effective batch at 16 by trading
# batch size for accumulation.
cfg["train"]["batch_size"] = 1
cfg["train"]["grad_accum"] = 16
cfg["train"]["load_in_4bit"] = True
cfg["train"]["gradient_checkpointing"] = True
# T4 and P100 are pre-Ampere: fp16, as in the paper.
cfg["train"]["precision"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

with open(CFG_PATH, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)

print(json.dumps({"paths": cfg["paths"], "train": cfg["train"]}, indent=2))
assert os.path.isdir(cfg["paths"]["dataset_dir"]), cfg["paths"]["dataset_dir"]
assert os.path.isdir(cfg["paths"]["processed_dir"]), cfg["paths"]["processed_dir"]'''

MD_VERIFY = """\
## 1. Verify the dataset

Recomputes every statistic in Table 3 from the attached JSONL and audits it
against the published numbers. Four checks are expected to fail — they are
real discrepancies in the released data, documented in `docs/DATA_NOTES.md`."""

VERIFY = '''\
import subprocess
print(subprocess.run([sys.executable, "-m", "vindr_cxr_vqa.stats",
                      "--config", CFG_PATH],
                     capture_output=True, text=True).stdout)

# Sanity-check the scorer itself: replaying the ground truth as predictions
# must give VQA precision 1.0 and 100% localization at IoU >= 0.5.
print(subprocess.run([sys.executable, "-m", "vindr_cxr_vqa.evaluate",
                      "--config", CFG_PATH, "--oracle", "--split", "val"],
                     capture_output=True, text=True).stdout)'''

MD_BASELINE = """\
## 2. Zero-shot baseline

The "Pretrained" row of Tables 4-6. Runs on a subset by default; drop `--limit`
for the full 2,637 validation QA pairs (a few hours)."""

BASELINE = '''\
N_BASELINE = 200   # QA pairs; set to None for the whole validation split

cmd = [sys.executable, "-m", "vindr_cxr_vqa.infer", "--config", CFG_PATH,
       "--split", "val", "--batch-size", "2",
       "--out", "/kaggle/working/outputs/preds_val_base.jsonl"]
if N_BASELINE:
    cmd += ["--limit", str(N_BASELINE)]
!{" ".join(cmd)}'''

BASELINE_EVAL = '''\
!{sys.executable} -m vindr_cxr_vqa.evaluate --config {CFG_PATH} \\
    --pred /kaggle/working/outputs/preds_val_base.jsonl --split val \\
    --out /kaggle/working/outputs/eval_val_base.json'''

MD_TRAIN = """\
## 3. Fine-tune

LoRA r=16, alpha=32, dropout=0.1, AdamW at 2e-5, weight decay 0.01, effective
batch 16, with the multi-task objective `L = L_VQA + 0.5 * L_bbox` where the
box term is a differentiable soft-GIoU read off the coordinate digit tokens.

`MAX_STEPS = -1` runs the paper's full 3 epochs; expect far more than a single
Kaggle session. Checkpoints land in `/kaggle/working/outputs/medgemma-lora`, so
a later session can resume with `--resume <checkpoint>`."""

TRAIN = '''\
MAX_STEPS = 600          # -1 for the paper's full 3 epochs
ADAPTER = "/kaggle/working/outputs/medgemma-lora"

!{sys.executable} -m vindr_cxr_vqa.train --config {CFG_PATH} \\
    --output-dir {ADAPTER} --max-steps {MAX_STEPS}'''

MD_EVAL = """\
## 4. Evaluate the fine-tuned model

Produces the "Fine-tuned" rows: Table 4 (VQA), Table 5 (box detection at
IoU >= 0.3) and Table 6 (per-image localization quality)."""

EVAL = '''\
# Same subset as the baseline, so the two rows are comparable.
cmd = [sys.executable, "-m", "vindr_cxr_vqa.infer", "--config", CFG_PATH,
       "--split", "val", "--adapter", ADAPTER, "--batch-size", "2",
       "--out", "/kaggle/working/outputs/preds_val_ft.jsonl"]
if N_BASELINE:
    cmd += ["--limit", str(N_BASELINE)]
!{" ".join(cmd)}

!{sys.executable} -m vindr_cxr_vqa.evaluate --config {CFG_PATH} \\
    --pred /kaggle/working/outputs/preds_val_ft.jsonl --split val \\
    --out /kaggle/working/outputs/eval_val_ft.json'''

COMPARE = '''\
# Side by side with the published numbers.
import json

PAPER = {
    "vqa":        {"Pretrained": dict(accuracy=.558, precision=.421, recall=.827, f1=.558),
                   "Fine-tuned": dict(accuracy=.624, precision=.471, recall=.926, f1=.624)},
    "detection":  {"Pretrained": dict(precision=.096, recall=.068, f1=.070, miou_tp=.036),
                   "Fine-tuned": dict(precision=.120, recall=.090, f1=.103, miou_tp=.615)},
}

def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None

runs = {"Pretrained": load("/kaggle/working/outputs/eval_val_base.json"),
        "Fine-tuned": load("/kaggle/working/outputs/eval_val_ft.json")}

for table, keys in (("vqa", ("accuracy", "precision", "recall", "f1")),
                    ("detection", ("precision", "recall", "f1", "miou_tp"))):
    print(f"\\n=== {table} ===")
    print(f"{'model':<12}" + "".join(f"{k:>12}" for k in keys) + "   (paper)")
    for name, res in runs.items():
        if res is None:
            print(f"{name:<12}  not run")
            continue
        got = "".join(f"{res[table][k]:>12.3f}" for k in keys)
        want = " ".join(f"{PAPER[table][name][k]:.3f}" for k in keys)
        print(f"{name:<12}{got}   ({want})")'''

MD_VIS = """\
## 5. Qualitative results

Figure 2 of the paper: ground truth in green, zero-shot prediction in yellow,
fine-tuned prediction in red."""

VIS = '''\
!{sys.executable} -m vindr_cxr_vqa.visualize --config {CFG_PATH} \\
    --pred-base /kaggle/working/outputs/preds_val_base.jsonl \\
    --pred-ft /kaggle/working/outputs/preds_val_ft.jsonl \\
    --split val --n 6 --out /kaggle/working/outputs/figures

from IPython.display import Image, display
import glob
for p in sorted(glob.glob("/kaggle/working/outputs/figures/*.png"))[:6]:
    print(p)
    display(Image(filename=p))'''

SAVE = '''\
# Keep the adapter and the metrics as notebook output (a few tens of MB).
import shutil, os
shutil.make_archive("/kaggle/working/medgemma-lora", "zip",
                    "/kaggle/working/outputs/medgemma-lora")
for f in sorted(glob.glob("/kaggle/working/outputs/*.json")):
    print(f, os.path.getsize(f), "bytes")
print("\\nadapter archive:",
      os.path.getsize("/kaggle/working/medgemma-lora.zip") / 2**20, "MB")'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md(MD_INTRO),
        md("## 0. Setup"),
        code(SETUP),
        code(INSTALL),
        code(TOKEN),
        code(CONFIG),
        md(MD_VERIFY),
        code(VERIFY),
        md(MD_BASELINE),
        code(BASELINE),
        code(BASELINE_EVAL),
        md(MD_TRAIN),
        code(TRAIN),
        md(MD_EVAL),
        code(EVAL),
        code(COMPARE),
        md(MD_VIS),
        code(VIS),
        md("## 6. Save outputs"),
        code(SAVE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(build(), fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
