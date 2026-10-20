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
2. **Attach the model.** *Add Input → Models* → MedGemma 4B instruction-tuned
   (Transformers). Copy its path into `MODEL_PATH` a few cells down.
3. **Accelerator** → `GPU T4 x2` or `GPU P100`. *Settings → Accelerator*.

**Running with Internet OFF** — set `OFFLINE = True` in the next cell. That
skips `pip install` and every hub lookup, so the model *must* be attached
(step 2) and Kaggle's preinstalled `transformers` must be ≥ 4.52. The preflight
cell verifies both before anything expensive starts.

**Running with Internet ON** — leave `OFFLINE = False`. If no model is
attached, the weights are downloaded instead, which needs an accepted licence
for [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
and an `HF_TOKEN` under *Add-ons → Secrets*.

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

OFFLINE = '''\
# Set this to True when the notebook runs with Internet switched OFF.
# Offline mode changes three things: no pip install, no model download, and
# no hub lookups at all. See the markdown cell below for what to attach.
OFFLINE = False

if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("offline mode: nothing will be downloaded")
else:
    print("online mode: packages and weights may be fetched")'''

INSTALL = '''\
# MedGemma is a Gemma 3 model: it needs a recent transformers.
# With Internet off this is skipped — the preflight cell below checks whether
# Kaggle's preinstalled versions are new enough.
if not OFFLINE:
    %pip install -q -U "transformers>=4.52" "accelerate>=1.4" "peft>=0.14" \\
        "bitsandbytes>=0.45" pyyaml

import torch, transformers
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("cuda        ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"vram         {total/2**30:.1f} GB total, {free/2**30:.1f} GB free")
    print("bf16         ", torch.cuda.is_bf16_supported())'''

MD_MODEL = """\
### Where the model weights come from

**Attach MedGemma via *Add Input → Models*, then paste its path into
`MODEL_PATH` below.** Kaggle shows the path on the model's panel in the right
sidebar once it is attached — something like
`/kaggle/input/medgemma/transformers/4b-it/1`.

Leaving `MODEL_PATH` empty is fine too: the cell then searches every attached
input for a directory holding `config.json` plus `*.safetensors`, and falls
back to downloading from HuggingFace when the notebook is online.

With Internet **off** the weights must already be attached — a Kaggle notebook
cannot reach the hub. If no MedGemma model is available to attach, run
`python scripts/fetch_model.py` on a machine with internet (≈8 GB, gated) and
upload the result as a Dataset."""

MODEL = '''\
# ---------------------------------------------------------------------------
# Paste the attached model path here. Add Input -> Models shows it in the
# right-hand panel, e.g. "/kaggle/input/medgemma/transformers/4b-it/1".
MODEL_PATH = ""
# ---------------------------------------------------------------------------

def find_model_dir(root="/kaggle/input"):
    """Any attached directory holding config.json plus safetensors weights."""
    hits = []
    for dirpath, _, files in os.walk(root):
        if "config.json" in files and any(f.endswith(".safetensors")
                                          for f in files):
            hits.append(dirpath)
    # Prefer a path naming medgemma, then the one with the most weight shards.
    hits.sort(key=lambda p: ("medgemma" not in p.lower(),
                             -len([f for f in os.listdir(p)
                                   if f.endswith(".safetensors")])))
    return hits[0] if hits else None

if MODEL_PATH:
    if not os.path.isdir(MODEL_PATH):
        raise SystemExit(f"MODEL_PATH does not exist: {MODEL_PATH}\\n"
                         f"attached inputs: {os.listdir('/kaggle/input')}")
    print("using MODEL_PATH:", MODEL_PATH)
else:
    MODEL_PATH = find_model_dir()
    if MODEL_PATH:
        print("auto-detected model:", MODEL_PATH)
    elif OFFLINE:
        raise SystemExit(
            "Offline mode needs the weights attached, but nothing under "
            "/kaggle/input holds config.json + *.safetensors.\\n"
            "Attach MedGemma via Add Input -> Models and set MODEL_PATH.")
    else:
        MODEL_PATH = "google/medgemma-4b-it"
        print("no attached weights; will download", MODEL_PATH)

if os.path.isdir(MODEL_PATH):
    print(sorted(f for f in os.listdir(MODEL_PATH)
                 if not f.startswith("."))[:12])'''

TOKEN = '''\
# HF_TOKEN is only needed to DOWNLOAD the gated weights. With the model
# attached from Kaggle, or offline, it is not used at all.
if not OFFLINE:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle Secrets")
    except Exception as exc:
        print("No HF_TOKEN secret:", exc)
        print("Fine if you attach the model instead of downloading it.")

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

# Point the model at the attached weights (or the hub id, when online).
cfg["model"]["name"] = MODEL_PATH

# The paper uses batch 4 x grad-accum 4 on a 48 GB A6000. A 16 GB T4/P100
# holds one sample at a time, so keep the effective batch at 16 by trading
# batch size for accumulation.
cfg["train"]["batch_size"] = 1
cfg["train"]["grad_accum"] = 16
cfg["train"]["load_in_4bit"] = True
cfg["train"]["gradient_checkpointing"] = True
# T4 and P100 are pre-Ampere: fp16, as in the paper.
cfg["train"]["precision"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

# One epoch is ~935 optimizer steps. The paper saves per epoch, but a Kaggle
# session can be cut short at any point, so save by step instead — that is what
# makes `--resume auto` able to pick the run back up in a later session.
cfg["train"]["save_strategy"] = "steps"
cfg["train"]["save_steps"] = 100
cfg["train"]["save_total_limit"] = 3

with open(CFG_PATH, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)

print(json.dumps({"paths": cfg["paths"], "model": cfg["model"],
                  "train": cfg["train"]}, indent=2))
assert os.path.isdir(cfg["paths"]["dataset_dir"]), cfg["paths"]["dataset_dir"]
assert os.path.isdir(cfg["paths"]["processed_dir"]), cfg["paths"]["processed_dir"]'''

MD_PREFLIGHT = """\
### Preflight

Checks package versions, the GPU, the attached weights and the dataset files
in one pass, so a missing piece surfaces now rather than forty minutes into
training. It exits non-zero if anything is blocking."""

PREFLIGHT = '''\
!{sys.executable} -m vindr_cxr_vqa.preflight --config {CFG_PATH}'''

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

The "Pretrained" row of Tables 4-6, measured on the **test** split — 300
images / 1,207 QA pairs, held out from training. Expect roughly 40-60 minutes
for the full split on a 4-bit T4; set `N_EVAL` to a few hundred for a quick
pass first.

The same split and the same `N_EVAL` are reused for the fine-tuned model below,
so the two rows are always comparable."""

BASELINE = '''\
SPLIT  = "test"    # 300 images / 1,207 QA pairs
N_EVAL = None      # cap the QA pairs for a quick pass; None = whole split

OUT = "/kaggle/working/outputs"
PRED_BASE = f"{OUT}/preds_{SPLIT}_base.jsonl"
PRED_FT   = f"{OUT}/preds_{SPLIT}_ft.jsonl"

cmd = [sys.executable, "-m", "vindr_cxr_vqa.infer", "--config", CFG_PATH,
       "--split", SPLIT, "--batch-size", "2", "--out", PRED_BASE]
if N_EVAL:
    cmd += ["--limit", str(N_EVAL)]
!{" ".join(cmd)}'''

BASELINE_EVAL = '''\
!{sys.executable} -m vindr_cxr_vqa.evaluate --config {CFG_PATH} \\
    --pred {PRED_BASE} --split {SPLIT} --out {OUT}/eval_{SPLIT}_base.json'''

MD_TRAIN = """\
## 3. Fine-tune

LoRA r=16, alpha=32, dropout=0.1, AdamW at 2e-5, weight decay 0.01, effective
batch 16, with the multi-task objective `L = L_VQA + 0.5 * L_bbox` where the
box term is a differentiable soft-GIoU read off the coordinate digit tokens.

**Checkpoints.** A checkpoint is written every 100 optimizer steps into
`/kaggle/working/outputs/medgemma-lora/checkpoint-N`, and `--resume auto` picks
up the newest one. So if a session is cut short, re-run this cell in a new
session and training continues rather than restarting — but **only if you saved
the output**: `/kaggle/working` is wiped when a session ends. Use *Save Version
→ Save & Run All*, or download `medgemma-lora.zip` from the last cell and
re-attach it as a Dataset.

`MAX_STEPS = -1` runs the paper's full 3 epochs (~2,805 steps), which is well
beyond a single 12 h session."""

TRAIN = '''\
MAX_STEPS = 600          # -1 for the paper's full 3 epochs (~2,805 steps)
ADAPTER = f"{OUT}/medgemma-lora"

# If a previous session left checkpoints in /kaggle/input, copy them in first
# so --resume auto can find them.
for src in glob.glob("/kaggle/input/*/medgemma-lora"):
    print("found previous checkpoints:", src)
    shutil.copytree(src, ADAPTER, dirs_exist_ok=True)

!{sys.executable} -m vindr_cxr_vqa.train --config {CFG_PATH} \\
    --output-dir {ADAPTER} --max-steps {MAX_STEPS} --resume auto'''

CHECKPOINTS = '''\
# What actually got saved.
for p in sorted(glob.glob(f"{ADAPTER}/checkpoint-*"),
                key=lambda s: int(s.rsplit("-", 1)[1])):
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs) / 2**20
    print(f"{os.path.basename(p):<20} {size:7.1f} MB")

adapter_files = [f for f in os.listdir(ADAPTER) if not f.startswith("check")]
print("\\nfinal adapter:", adapter_files)
assert any(f.startswith("adapter_model") for f in adapter_files), \\
    "no adapter weights were written — training did not finish"'''

MD_EVAL = """\
## 4. Final results on the test split

Generates with the fine-tuned adapter over the same split and cap as the
baseline, then scores Table 4 (VQA), Table 5 (box detection at IoU >= 0.3) and
Table 6 (per-image localization quality)."""

EVAL = '''\
cmd = [sys.executable, "-m", "vindr_cxr_vqa.infer", "--config", CFG_PATH,
       "--split", SPLIT, "--adapter", ADAPTER, "--batch-size", "2",
       "--out", PRED_FT]
if N_EVAL:
    cmd += ["--limit", str(N_EVAL)]
!{" ".join(cmd)}

!{sys.executable} -m vindr_cxr_vqa.evaluate --config {CFG_PATH} \\
    --pred {PRED_FT} --split {SPLIT} --out {OUT}/eval_{SPLIT}_ft.json'''

COMPARE = '''\
# Final tables, baseline vs fine-tuned, with the published numbers alongside.
!{sys.executable} -m vindr_cxr_vqa.results \\
    --base {OUT}/eval_{SPLIT}_base.json \\
    --ft   {OUT}/eval_{SPLIT}_ft.json \\
    --split {SPLIT} --out {OUT}/results_{SPLIT}.md

from IPython.display import Markdown, display
display(Markdown(open(f"{OUT}/results_{SPLIT}.md").read()))'''

MD_VIS = """\
## 5. Qualitative results

Figure 2 of the paper: ground truth in green, zero-shot prediction in yellow,
fine-tuned prediction in red."""

VIS = '''\
!{sys.executable} -m vindr_cxr_vqa.visualize --config {CFG_PATH} \\
    --pred-base {PRED_BASE} --pred-ft {PRED_FT} \\
    --split {SPLIT} --n 6 --out {OUT}/figures

from IPython.display import Image, display
for p in sorted(glob.glob(f"{OUT}/figures/*.png"))[:6]:
    print(p)
    display(Image(filename=p))'''

SAVE = '''\
# /kaggle/working is wiped when the session ends — archive what matters so it
# survives as notebook output, and so a later session can resume from it.
shutil.make_archive("/kaggle/working/medgemma-lora", "zip", ADAPTER)
print("adapter archive: "
      f"{os.path.getsize('/kaggle/working/medgemma-lora.zip') / 2**20:.1f} MB")

print("\\nmetrics and predictions:")
for f in sorted(glob.glob(f"{OUT}/*.json") + glob.glob(f"{OUT}/*.md")
                + glob.glob(f"{OUT}/*.jsonl")):
    print(f"  {os.path.basename(f):<28} "
          f"{os.path.getsize(f) / 2**10:8.1f} KB")

print("\\nTo continue training in another session: Save Version -> Save & Run "
      "All, then attach this notebook's output as an input next time; the "
      "train cell copies medgemma-lora back and resumes.")'''


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
        code(OFFLINE),
        code(INSTALL),
        code(TOKEN),
        md(MD_MODEL),
        code(MODEL),
        code(CONFIG),
        md(MD_PREFLIGHT),
        code(PREFLIGHT),
        md(MD_VERIFY),
        code(VERIFY),
        md(MD_BASELINE),
        code(BASELINE),
        code(BASELINE_EVAL),
        md(MD_TRAIN),
        code(TRAIN),
        code(CHECKPOINTS),
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
