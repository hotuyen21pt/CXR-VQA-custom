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
   via *Add Input*. Paste its path into `BUNDLE` in the setup cell, e.g.
   `/kaggle/input/datasets/<user>/vqa-repo`.
2. **Attach the model.** *Add Input → Models* → MedGemma 4B instruction-tuned
   (Transformers); MedGemma **1.5** 4B-it works too — both are Gemma 3 and load
   through the same `AutoModelForImageTextToText` path. Paste its path into
   `MODEL_PATH` a few cells down.
3. **Accelerator** → `GPU T4 x2` or `GPU P100`. *Settings → Accelerator*.

Both paths can be left empty — the cells search the attached inputs — but
pasting them is faster and gives a clearer error when something is not
attached.

> **Import order matters.** Import this notebook *first*, then attach the
> inputs and set the accelerator. Kaggle applies a notebook's stored settings
> on import and will say *"your Accelerator, Data Sources and Internet
> settings have changed"* — this notebook deliberately stores none, so nothing
> of yours is overwritten, but an import after attaching can still reset the
> panel. If your inputs disappear, just add them again.

**Internet on or off is detected automatically.** With it off, the notebook
skips `pip install` and every hub lookup, so the model *must* be attached
(step 2) and Kaggle's preinstalled `transformers` must be ≥ 4.50. With it on
and no model attached, the weights are downloaded instead, which needs an
accepted licence for
[`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) and an
`HF_TOKEN` under *Add-ons → Secrets*. The preflight cell checks whichever path
you are on before anything expensive starts.

**What this produces.** Table 3 (dataset statistics, audited against the
paper), Tables 4-6 for the zero-shot baseline and the fine-tuned model on both
the validation split (what the paper reports) and the held-out test split, and
Figure 2.

**On run time.** The paper trains 3 epochs over ~21k samples in ~36 h on an
A6000 (48 GB). A Kaggle session is capped at 12 h on a 16 GB GPU, so the
defaults below run a single capped pass that finishes well inside one session.
Checkpoints are written every 100 steps, so raising `MAX_STEPS` (or setting it
to `-1` for the full 3 epochs) and resuming across sessions is possible — see
the fine-tuning section for how to carry checkpoints over."""

SETUP = '''\
import os, sys, glob, json, shutil, subprocess

# ---------------------------------------------------------------------------
# Paste the attached dataset path here, e.g.
#   "/kaggle/input/datasets/<user>/vqa-repo"
# Leave it empty to search every attached input automatically.
BUNDLE = ""
# ---------------------------------------------------------------------------

def find_bundle(root="/kaggle/input", max_depth=5):
    """Any attached directory holding the vindr_cxr_vqa package.

    Kaggle nests inputs differently depending on how they were attached
    (/kaggle/input/<slug> for some, /kaggle/input/datasets/<user>/<slug> for
    others), so this walks rather than globbing one level.
    """
    hits = []
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.count("/") - base >= max_depth:
            dirnames[:] = []
            continue
        if "vindr_cxr_vqa" in dirnames:
            hits.append(dirpath)
            dirnames[:] = []          # do not descend into a match
    return sorted(hits, key=len)[0] if hits else None

if BUNDLE:
    if not os.path.isdir(os.path.join(BUNDLE, "vindr_cxr_vqa")):
        raise SystemExit(f"{BUNDLE} does not contain vindr_cxr_vqa/\\n"
                         f"found instead: {os.listdir(BUNDLE)}")
else:
    BUNDLE = find_bundle()
    if not BUNDLE:
        raise SystemExit(
            "No attached input contains vindr_cxr_vqa/. Use 'Add Input' to "
            "attach the bundle from scripts/make_kaggle_bundle.py, then set "
            "BUNDLE above.\\n"
            f"Attached: {glob.glob('/kaggle/input/*/') + glob.glob('/kaggle/input/*/*/')}")
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
# None = detect automatically; set True/False to force it.
# Detection matters: with Internet off but OFFLINE false, pip spends minutes
# retrying DNS before giving up.
OFFLINE = None

def has_internet(host="pypi.org", port=443, timeout=4):
    import socket
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False

if OFFLINE is None:
    OFFLINE = not has_internet()
    print(f"detected: {'no internet' if OFFLINE else 'internet available'}")

if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("offline mode: no pip install, no downloads, no hub lookups")
else:
    print("online mode: packages and weights may be fetched")'''

INSTALL = '''\
# MedGemma is a Gemma 3 model: it needs a recent transformers.
# With Internet off this is skipped — the preflight cell below checks whether
# Kaggle's preinstalled versions are new enough.
# --retries/--timeout keep a misdetected network from stalling for minutes.
if not OFFLINE:
    %pip install -q -U --retries 1 --timeout 15 \\
        "transformers>=4.50" "accelerate>=1.4" "peft>=0.14" \\
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
# right-hand panel, e.g.
#   "/kaggle/input/models/<user>/googlemedgemma-1-5-4b-it/transformers/default/1/medgemma-1.5-4b-it"
# Leave empty to search the attached inputs automatically.
MODEL_PATH = ""
# ---------------------------------------------------------------------------

def find_model_dir(root="/kaggle/input"):
    """Any attached directory holding config.json plus safetensors weights."""
    hits = []
    for dirpath, dirnames, files in os.walk(root):
        # The data bundle holds thousands of PNGs; nothing to find in there.
        dirnames[:] = [d for d in dirnames if d != "train"]
        if "config.json" in files and any(f.endswith(".safetensors")
                                          for f in files):
            hits.append(dirpath)
            dirnames[:] = []
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

The "Pretrained" row of Tables 4-6.

Inference runs over the **validation** split (659 images / 2,637 QA pairs) —
the set the paper reports on. The test split is a subset of it, so the same
predictions score both: validation for the paper-comparable numbers, test for
the held-out result. One pass instead of two.

Expect roughly 1.5-2 h for the full split on a 4-bit T4. Set `N_EVAL` to a few
hundred for a quick pass first; the fine-tuned run below reuses the same cap,
so the two rows stay comparable."""

BASELINE = '''\
N_EVAL = None      # cap the QA pairs for a quick pass; None = whole split

OUT = "/kaggle/working/outputs"
PRED_BASE = f"{OUT}/preds_val_base.jsonl"
PRED_FT   = f"{OUT}/preds_val_ft.jsonl"

cmd = [sys.executable, "-m", "vindr_cxr_vqa.infer", "--config", CFG_PATH,
       "--split", "val", "--batch-size", "2", "--out", PRED_BASE]
if N_EVAL:
    cmd += ["--limit", str(N_EVAL)]
!{" ".join(cmd)}'''

BASELINE_EVAL = '''\
def score(pred, tag):
    """Score one prediction file against both splits."""
    for split in ("val", "test"):
        r = subprocess.run(
            [sys.executable, "-m", "vindr_cxr_vqa.evaluate",
             "--config", CFG_PATH, "--pred", pred, "--split", split,
             "--out", f"{OUT}/eval_{split}_{tag}.json"],
            capture_output=True, text=True)
        print(r.stdout or r.stderr)

score(PRED_BASE, "base")'''

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
## 4. Final results

Generates with the fine-tuned adapter over the same split and cap as the
baseline, then scores Table 4 (VQA), Table 5 (box detection at IoU >= 0.3) and
Table 6 (per-image localization quality) on **both** splits:

* **validation** — 659 images, directly comparable to the paper's Tables 4-6;
* **test** — 300 held-out images, the final result."""

EVAL = '''\
cmd = [sys.executable, "-m", "vindr_cxr_vqa.infer", "--config", CFG_PATH,
       "--split", "val", "--adapter", ADAPTER, "--batch-size", "2",
       "--out", PRED_FT]
if N_EVAL:
    cmd += ["--limit", str(N_EVAL)]
!{" ".join(cmd)}

score(PRED_FT, "ft")'''

COMPARE = '''\
# Final tables, baseline vs fine-tuned, with the published numbers alongside.
from IPython.display import Markdown, display

for split in ("val", "test"):
    subprocess.run([sys.executable, "-m", "vindr_cxr_vqa.results",
                    "--base", f"{OUT}/eval_{split}_base.json",
                    "--ft",   f"{OUT}/eval_{split}_ft.json",
                    "--split", split,
                    "--out",  f"{OUT}/results_{split}.md"],
                   capture_output=True, text=True)
    display(Markdown(open(f"{OUT}/results_{split}.md").read()))'''

MD_VIS = """\
## 5. Qualitative results

Figure 2 of the paper: ground truth in green, zero-shot prediction in yellow,
fine-tuned prediction in red."""

VIS = '''\
!{sys.executable} -m vindr_cxr_vqa.visualize --config {CFG_PATH} \\
    --pred-base {PRED_BASE} --pred-ft {PRED_FT} \\
    --split val --n 6 --out {OUT}/figures

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
        # Deliberately no "kaggle" metadata block and no "accelerator" key.
        # Kaggle applies those on import and warns that "your Accelerator,
        # Data Sources and Internet settings have changed" -- an empty
        # dataSources list would detach the inputs the user had just added.
        # Leaving them out keeps whatever the notebook is already configured
        # with; the intro cell says which settings to pick by hand.
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(build(), fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
