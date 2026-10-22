"""Cell sources shared by the Kaggle notebooks, plus the writers that build them.

``make_notebook.py`` (training) and ``make_infer_notebook.py`` (inference) both
have to find the attached inputs, reject a stale code bundle, decide whether
the session has internet and locate the weights.  Keeping one copy here means
a fix to any of that reaches both notebooks the next time they are generated,
instead of only the one that happened to be edited.

Nothing here is executed by the generators; these are strings that become
notebook cells.
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Setup: finding what is attached
# --------------------------------------------------------------------------
FINDERS = '''\
def find_inputs(marker, root="/kaggle/input", max_depth=6):
    """Every attached directory containing `marker`, shallowest first.

    Kaggle nests inputs differently depending on how they were attached
    (/kaggle/input/<slug> for some, /kaggle/input/datasets/<user>/<slug> or
    /kaggle/input/models/<user>/<model>/<fw>/<var>/<ver>/ for others), so this
    walks rather than globbing one level.
    """
    hits, base = [], root.rstrip("/").count("/")
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.count("/") - base >= max_depth:
            dirnames[:] = []
            continue
        # Never descend into the image tree; there are thousands of files.
        dirnames[:] = [d for d in dirnames if d != "train"]
        if os.path.exists(os.path.join(dirpath, marker)):
            hits.append(dirpath)
            dirnames[:] = []
    return sorted(hits, key=len)

def tree(root="/kaggle/input", max_depth=4, limit=40):
    """What is actually attached, files included.

    Two levels of directory names say nothing about why a search failed --
    an un-extracted zip and a dataset that was never attached look identical.
    Showing the files tells the two apart at a glance.
    """
    lines, seen, base = [], 0, root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count("/") - base
        if depth >= max_depth:
            dirnames[:] = []
        # Never list the image tree; there are thousands of files.
        dirnames[:] = [d for d in dirnames if d != "train"]
        lines.append("  " * depth + (os.path.basename(dirpath) or root) + "/")
        for name in sorted(filenames)[:5]:
            lines.append("  " * (depth + 1) + name)
        if len(filenames) > 5:
            lines.append("  " * (depth + 1) + f"… {len(filenames) - 5} file khác")
        seen += 1
        if seen >= limit:
            lines.append("  …")
            break
    return "\\n".join(lines)

def unpack(marker, dest="/kaggle/working/unpacked"):
    """Extract an attached .zip that carries `marker`, if one is still zipped.

    Uploading a bundle through the Kaggle UI usually extracts it, but not
    always -- and a dataset left as a single .zip is indistinguishable from
    one that was never attached. Rather than stop on a zip holding exactly
    what is being looked for, unpack it into the writable tree and go on.
    """
    import zipfile
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        dirnames[:] = [d for d in dirnames if d != "train"]
        for name in filenames:
            if not name.endswith(".zip"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with zipfile.ZipFile(path) as z:
                    inside = z.namelist()
                    if not any(n.startswith(marker) for n in inside):
                        continue
                    out = os.path.join(dest, os.path.splitext(name)[0])
                    # Code and JSONL ride in the same bundle, so this is
                    # reached twice; unpack it once.
                    if os.path.exists(os.path.join(out, marker)):
                        return out
                    print(f"{path} chưa được giải nén; đang bung "
                          f"{len(inside)} file -> {out}")
                    z.extractall(out)
                    return out
            except zipfile.BadZipFile:
                continue
    return None

def locate(marker, what):
    """Every input holding `marker`, unpacking a stray zip if that is all there is."""
    hits = find_inputs(marker)
    if hits:
        return hits
    out = unpack(marker)
    if out:
        return find_inputs(marker, root=out)
    raise SystemExit(
        f"Không input nào chứa {marker} ({what}).\\n\\n"
        f"Đang attach:\\n{tree()}\\n\\n"
        "Kiểm tra: dataset đã attach chưa, đã trỏ sang version mới nhất chưa, "
        "và Kaggle đã giải nén file .zip chưa.")

'''

# --------------------------------------------------------------------------
# Setup: refusing a stale bundle
# --------------------------------------------------------------------------
CODE_CHECK = '''\
# Checking that the .py files merely exist is not enough: an attached dataset
# can carry an older copy of a module that is present but missing a fix, and
# the failure then surfaces half an hour later as a traceback from deep inside
# a library. So look for a snippet each fix introduced.
#
# Snippets, not just function names: some fixes add no new function -- the one
# that moves the box loss onto the GPU is a single line inside an existing
# method -- and a names-only check waves those through.
NEEDS = {
    "modeling":  ["def load_base_model", "def disable_broken_peft_dispatchers",
                  "def adapt_training_kwargs", "def language_side_targets"],
    "evaluate":  ["def diagnose_predictions", "def load_predictions",
                  "def eval_boxes"],
    "train":     ["def latest_checkpoint", "def warn_if_nothing_will_be_saved",
                  "def build_training_arguments"],
    "losses":    ["self.to(logits.device)"],
    "results":   ["def render"],
    "preflight": ["def check_model"],
    "infer":     ["def run"],
    "boxes":     ["def to_canvas", "def match_any", "def greedy_match"],
}

def missing_markers(root):
    """Which of NEEDS a copy of the package under `root` does not have."""
    gone = []
    for mod, markers in NEEDS.items():
        path = os.path.join(root, "vindr_cxr_vqa", mod + ".py")
        if not os.path.exists(path):
            gone.append(f"{mod}.py is missing entirely")
            continue
        src = open(path, encoding="utf-8").read()
        gone += [f"{mod}.py: {m}" for m in markers if m not in src]
    return gone

'''

# --------------------------------------------------------------------------
# Setup: picking the inputs and copying the code somewhere writable
# --------------------------------------------------------------------------
RESOLVE = '''\
# The code, the built JSONL and the images may live in one input or in two.
#
# Two attached inputs can both carry vindr_cxr_vqa/ -- an older full bundle
# next to a freshly uploaded code-only dataset -- and picking by path length
# alone is a coin flip between them. So prefer whichever copy actually has the
# functions this notebook needs. When none does, keep the shallowest, so the
# report below names what is missing rather than running yesterday's code.
if BUNDLE:
    CODE_ROOT = BUNDLE
else:
    roots = locate("vindr_cxr_vqa", "code")
    CODE_ROOT = next((r for r in roots if not missing_markers(r)), roots[0])
    if len(roots) > 1:
        print(f"{len(roots)} attached inputs carry vindr_cxr_vqa/; "
              f"chose {CODE_ROOT}")

PROCESSED_ROOT = locate("data/processed/train.jsonl", "JSONL đã build")[0]
IMAGES_ROOT = IMAGES or locate("dataset/train_data/train", "ảnh X-quang")[0]

print("code     :", CODE_ROOT)
print("jsonl    :", PROCESSED_ROOT)
print("images   :", IMAGES_ROOT)

# /kaggle/input is read-only, so work from a writable copy of the code.
WORK = "/kaggle/working/vindr"
os.makedirs(WORK, exist_ok=True)
for item in ("vindr_cxr_vqa", "configs", "requirements.txt", "docs"):
    src = os.path.join(CODE_ROOT, item)
    if not os.path.exists(src):
        continue
    dst = os.path.join(WORK, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

os.chdir(WORK)
sys.path.insert(0, WORK)
print("\\ncwd:", os.getcwd())
print(sorted(os.listdir(WORK)))

# Fail now, clearly, if the attached code predates what this notebook needs.
# This re-checks the writable copy rather than CODE_ROOT, so a half-finished
# copy is caught too -- and it is the only report when BUNDLE was set by hand
# or when no attached input is current.
stale = missing_markers(WORK)
if stale:
    raise SystemExit(
        "The attached code is OUT OF DATE. Missing:\\n  "
        + "\\n  ".join(stale)
        + f"\\n\\nIt was read from: {CODE_ROOT}\\n"
        "Rebuild with `python scripts/make_kaggle_bundle.py --part code`, "
        "upload it as a new version of that dataset, and re-run.")
print("code check: up to date")'''

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
PREFLIGHT = '''\
# Stop here on a blocking problem rather than failing hours later.
r = subprocess.run([sys.executable, "-m", "vindr_cxr_vqa.preflight",
                    "--config", CFG_PATH], capture_output=True, text=True)
print(r.stdout or r.stderr)
if r.returncode != 0:
    raise SystemExit("Preflight found blocking problems — fix them above "
                     "before running the rest of the notebook.")'''


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
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


def write(path: str, cells: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(notebook(cells), fh, indent=1, ensure_ascii=False)
    print(f"wrote {path}")
