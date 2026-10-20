"""Check that a machine can actually run training before it starts.

Written for Kaggle's no-internet mode, where a missing package or an
unattached model shows up as a confusing traceback forty minutes in.  Every
check reports what is wrong and what to do about it.

    python -m vindr_cxr_vqa.preflight --config configs/default.yaml

Standard library only; packages are probed by import, never required.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from .config import load_config

#: Minimum versions. MedGemma is a Gemma 3 model, which transformers only
#: learned to load in 4.50; 4.52 is the first release with a stable
#: AutoModelForImageTextToText path for it.
REQUIRED = {
    "torch": "2.4",
    "transformers": "4.52",
    "accelerate": "1.4",
    "peft": "0.14",
    "PIL": None,            # Pillow, imported as PIL
}
OPTIONAL = {
    "bitsandbytes": "0.45",  # only when train.load_in_4bit is true
    "yaml": None,            # configs fall back to a built-in parser
}

#: Files `from_pretrained` needs in a local model directory.
MODEL_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json")


def _version(mod) -> str:
    return getattr(mod, "__version__", None) or getattr(mod, "VERSION", "?")


def _as_tuple(v: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in str(v).split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = 0

    def add(self, ok: bool | None, name: str, detail: str) -> None:
        mark = "OK  " if ok else ("WARN" if ok is None else "FAIL")
        if ok is False:
            self.failed += 1
        self.rows.append((mark, name, detail))

    def show(self) -> int:
        width = max(len(n) for _, n, _ in self.rows)
        for mark, name, detail in self.rows:
            print(f"[{mark}] {name:<{width}}  {detail}")
        print()
        if self.failed:
            print(f"{self.failed} blocking problem(s) — fix these before "
                  "training.")
        else:
            print("Ready to train.")
        return 1 if self.failed else 0


# --------------------------------------------------------------------------
def check_packages(rep: Report, need_4bit: bool) -> None:
    for name, minimum in REQUIRED.items():
        try:
            mod = importlib.import_module(name)
        except ImportError:
            rep.add(False, name, f"not installed (need >= {minimum or 'any'})")
            continue
        got = _version(mod)
        if minimum and got != "?" and _as_tuple(got) < _as_tuple(minimum):
            rep.add(False, name, f"{got} is too old, need >= {minimum}")
        else:
            rep.add(True, name, str(got))

    for name, minimum in OPTIONAL.items():
        required_here = need_4bit and name == "bitsandbytes"
        try:
            mod = importlib.import_module(name)
        except ImportError:
            rep.add(False if required_here else None, name,
                    "not installed"
                    + (" but train.load_in_4bit is true" if required_here
                       else " (optional)"))
            continue
        rep.add(True, name, str(_version(mod)))


def check_gpu(rep: Report) -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        rep.add(False, "gpu", "no CUDA device — training needs one")
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    bf16 = torch.cuda.is_bf16_supported()
    detail = f"{name}, {total:.1f} GB, bf16={'yes' if bf16 else 'no'}"
    # 4-bit MedGemma-4B plus activations needs roughly 12 GB.
    rep.add(True if total >= 12 else None, "gpu", detail
            + ("" if total >= 12 else " — tight for a 4B model"))


def check_model(rep: Report, name: str, offline: bool) -> None:
    if os.path.isdir(name):
        missing = [f for f in MODEL_FILES
                   if not os.path.exists(os.path.join(name, f))]
        weights = [f for f in os.listdir(name) if f.endswith(".safetensors")]
        if missing:
            rep.add(False, "model", f"{name} is missing {missing}")
        elif not weights:
            rep.add(False, "model", f"{name} has no .safetensors weights")
        else:
            size = sum(os.path.getsize(os.path.join(name, f))
                       for f in weights) / 2**30
            rep.add(True, "model", f"{name} ({len(weights)} shard(s), "
                                   f"{size:.1f} GB)")
        return

    # Not a local path: it is a hub id, which needs either network or a cache.
    cache = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub", "models--" + name.replace("/", "--"))
    if os.path.isdir(cache):
        rep.add(True, "model", f"{name} found in the local HF cache")
    elif offline:
        rep.add(False, "model",
                f"{name} is a hub id, but offline mode is on and it is not "
                "cached — attach the weights and set model.name to that path")
    else:
        rep.add(None, "model", f"{name} will be downloaded from HuggingFace "
                               "(needs internet and an accepted licence)")


def check_dataset(rep: Report, cfg) -> None:
    pdir = cfg.paths.processed_dir
    for name in ("train.jsonl", "val.jsonl", "images.json"):
        path = os.path.join(pdir, name)
        if os.path.exists(path):
            rep.add(True, f"data/{name}",
                    f"{os.path.getsize(path) / 2**20:.1f} MB")
        else:
            rep.add(False, f"data/{name}",
                    f"missing — run python -m vindr_cxr_vqa.build_dataset")

    train = os.path.join(pdir, "train.jsonl")
    if not os.path.exists(train):
        return
    with open(train, encoding="utf-8") as fh:
        sample = [json.loads(next(fh)) for _ in range(min(50, 50))]
    missing = [r["image_path"] for r in sample
               if not os.path.exists(os.path.join(cfg.paths.dataset_dir,
                                                  r["image_path"]))]
    if missing:
        rep.add(False, "images",
                f"{len(missing)}/{len(sample)} sampled images not found under "
                f"{cfg.paths.dataset_dir}")
    else:
        rep.add(True, "images",
                f"{len(sample)} sampled paths resolve under "
                f"{cfg.paths.dataset_dir}")


def check_offline_env(rep: Report) -> bool:
    flags = {k: os.environ.get(k) for k in
             ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    on = any(v == "1" for v in flags.values())
    rep.add(True, "offline mode",
            "on — nothing will be downloaded" if on
            else "off — the hub may be contacted")
    return on


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    print(f"python {sys.version.split()[0]}\nconfig {args.config}\n")
    rep = Report()
    offline = check_offline_env(rep)
    check_packages(rep, need_4bit=bool(cfg.train.load_in_4bit))
    check_gpu(rep)
    check_model(rep, str(cfg.model.name), offline)
    check_dataset(rep, cfg)
    raise SystemExit(rep.show())


if __name__ == "__main__":
    main()
