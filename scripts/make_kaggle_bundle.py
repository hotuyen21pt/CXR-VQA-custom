"""Package the repo, the built dataset and the images it uses into one zip.

Upload the result as a Kaggle Dataset, attach it to a notebook, and
``notebooks/kaggle_train.ipynb`` runs end to end with no further downloads.

Only the 4,394 chest X-rays that VinDr-CXR-VQA actually references are copied,
which keeps the bundle near 400 MB instead of the 1.3 GB of the full VinDr
train partition.

    python scripts/make_kaggle_bundle.py              # one zip, ~362 MB
    python scripts/make_kaggle_bundle.py --part code  # code + JSONL, ~9 MB
    python scripts/make_kaggle_bundle.py --part images

Splitting is worth it once the repo is already on Kaggle: the images never
change, but the code does, so uploading two datasets means later fixes cost a
9 MB re-upload instead of 362 MB. The notebook finds the code and the images
wherever they are attached, together or apart.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

#: Copied verbatim into the bundle root.
CODE_ITEMS = [
    "scripts",
    "vindr_cxr_vqa",
    "configs",
    "requirements.txt",
    "README.md",
    "docs",
]

#: VinDr CSVs the builder needs if the dataset is ever rebuilt on Kaggle.
DATASET_FILES = [
    "annotations_train.csv",
    "image_labels_train.csv",
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def copy_code(repo: str, stage: str) -> None:
    for item in CODE_ITEMS:
        src = os.path.join(repo, item)
        if not os.path.exists(src):
            continue
        dst = os.path.join(stage, item)
        if os.path.isdir(src):
            shutil.copytree(
                src, dst, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[bundle] code  -> {stage}")


def copy_processed(repo: str, stage: str) -> list[str]:
    """Copy the built JSONL + image index; return the image ids referenced."""
    src = os.path.join(repo, "data", "processed")
    if not os.path.isdir(src):
        raise SystemExit(
            "data/processed not found — run "
            "`python -m vindr_cxr_vqa.build_dataset` first")
    dst = os.path.join(stage, "data", "processed")
    os.makedirs(dst, exist_ok=True)

    ids: set[str] = set()
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "images.json",
                 "build_summary.json"):
        s = os.path.join(src, name)
        if not os.path.exists(s):
            continue
        shutil.copy2(s, os.path.join(dst, name))
        if name.endswith(".jsonl"):
            with open(s, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        ids.add(json.loads(line)["image_id"])
    print(f"[bundle] data  -> {dst}  ({len(ids)} distinct images referenced)")
    return sorted(ids)


def copy_images(stage: str, dataset_dir: str, image_ids: list[str]) -> int:
    src_dir = os.path.join(dataset_dir, "train_data", "train")
    dst_dir = os.path.join(stage, "dataset", "train_data", "train")
    os.makedirs(dst_dir, exist_ok=True)

    total = 0
    missing: list[str] = []
    for i, iid in enumerate(image_ids, 1):
        s = os.path.join(src_dir, f"{iid}.png")
        if not os.path.exists(s):
            missing.append(iid)
            continue
        shutil.copy2(s, os.path.join(dst_dir, f"{iid}.png"))
        total += os.path.getsize(s)
        if i % 500 == 0:
            print(f"[bundle] images {i}/{len(image_ids)}", end="\r", flush=True)

    rf = os.path.join(src_dir, "resize_factors.csv")
    if os.path.exists(rf):
        shutil.copy2(rf, os.path.join(dst_dir, "resize_factors.csv"))
    for name in DATASET_FILES:
        s = os.path.join(dataset_dir, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(stage, "dataset", name))
            total += os.path.getsize(s)

    if missing:
        print(f"\n[bundle] WARNING: {len(missing)} image(s) not found under "
              f"{src_dir}; first few: {missing[:5]}")
    print(f"\n[bundle] images-> {dst_dir}  ({human(total)})")
    return total


def write_manifest(stage: str, n_images: int, part: str) -> None:
    contents = {
        "all": f"the repo, the built dataset and the {n_images} chest X-rays "
               "it references — everything needed in one input",
        "code": "the repo and the built dataset (JSONL + index). **No "
                "images** — attach the images part as a second input",
        "images": f"the {n_images} chest X-rays and the VinDr CSVs. **No "
                  "code** — attach the code part as a second input",
    }[part]
    with open(os.path.join(stage, "KAGGLE_README.md"), "w",
              encoding="utf-8") as fh:
        fh.write(f"""# VinDr-CXR-VQA — Kaggle bundle ({part})

Contains {contents}.

## How to use

1. Kaggle → *Datasets* → *New Dataset* → upload this zip. Kaggle unpacks it,
   so the contents land under `/kaggle/input/…/<your-dataset-slug>/`.
2. Import `notebooks/kaggle_train.ipynb` **before** attaching anything
   (Kaggle applies a notebook's stored settings on import), then attach this
   dataset and set the accelerator to **GPU T4 x2** or **P100**.
3. *Add Input → Models* → attach MedGemma 4B-it or MedGemma 1.5 4B-it, and
   paste its path into `MODEL_PATH` in the model cell.
4. Run all cells.

Works with Internet **off** — the notebook detects it and skips every
download. No `HF_TOKEN` is needed once the model is attached; it only ever
authenticates a download. The dataset is already built inside this bundle.

## Layout

```
vindr_cxr_vqa/          package (build, train, infer, evaluate, preflight)
scripts/reproduce.py    run every result the paper reports, end to end
configs/default.yaml    hyper-parameters from the paper
data/processed/         train/val/test JSONL + images.json
dataset/train_data/train/*.png
dataset/annotations_train.csv, image_labels_train.csv
```

Code and images can live in two separate datasets (`--part code` /
`--part images`); the notebook finds each wherever it is attached.
""")


def zip_stage(stage: str, out_zip: str) -> None:
    print(f"[bundle] zipping -> {out_zip}")
    n = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=1) as z:
        for root, _, files in os.walk(stage):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, stage)
                # PNGs are already compressed; storing them is much faster.
                z.write(full, arc,
                        compress_type=(zipfile.ZIP_STORED
                                       if name.endswith(".png")
                                       else zipfile.ZIP_DEFLATED))
                n += 1
                if n % 1000 == 0:
                    print(f"[bundle] zipped {n} files", end="\r", flush=True)
    print(f"\n[bundle] {n} files, {human(os.path.getsize(out_zip))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--part", choices=("all", "code", "images"), default="all",
                    help="all: one zip; code: package + JSONL; images: PNGs "
                         "and the VinDr CSVs")
    ap.add_argument("--no-images", action="store_true",
                    help="deprecated alias for --part code")
    ap.add_argument("--keep-stage", action="store_true")
    args = ap.parse_args()
    if args.no_images:
        args.part = "code"

    from vindr_cxr_vqa.config import load_config
    cfg = load_config(os.path.join(args.repo, "configs", "default.yaml"))
    dataset_dir = args.dataset_dir or cfg.paths.dataset_dir

    suffix = {"all": "", "code": "-code", "images": "-images"}[args.part]
    stage = os.path.join(args.repo, "outputs", "kaggle_bundle")
    out_zip = args.out or os.path.join(
        args.repo, "outputs", f"vindr-cxr-vqa-kaggle{suffix}.zip")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)

    # The image ids always come from the built JSONL, even when only the
    # images are being packaged.
    ids = copy_processed(args.repo, stage)
    if args.part == "images":
        shutil.rmtree(os.path.join(stage, "data"))
    else:
        copy_code(args.repo, stage)
        nb_src = os.path.join(args.repo, "notebooks")
        if os.path.isdir(nb_src):
            shutil.copytree(nb_src, os.path.join(stage, "notebooks"),
                            dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
    if args.part in ("all", "images"):
        copy_images(stage, dataset_dir, ids)
    write_manifest(stage, len(ids), args.part)

    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    if os.path.exists(out_zip):
        os.remove(out_zip)
    zip_stage(stage, out_zip)

    if not args.keep_stage:
        shutil.rmtree(stage)
    print(f"\n[bundle] done: {out_zip}")
    print("[bundle] upload it at kaggle.com/datasets -> New Dataset")


if __name__ == "__main__":
    main()
