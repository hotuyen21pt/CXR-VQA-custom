"""Run everything the paper reports, end to end.

Produces Table 3 (dataset statistics), Tables 4-6 for both the zero-shot
baseline and the fine-tuned model, Figure 2, and a consolidated
`outputs/REPRODUCTION.md`.

Inference runs once per model over the **validation** split. The test split is
a subset of validation, so the same predictions score both: the validation
numbers are the ones comparable to the paper, the test numbers are the
held-out result.

    python scripts/reproduce.py                    # the whole thing
    python scripts/reproduce.py --skip-train       # score an existing adapter
    python scripts/reproduce.py --limit 200        # quick smoke run
    python scripts/reproduce.py --stages data      # dataset only, no GPU

Every stage is skipped when its output already exists, so an interrupted run
can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

STAGES = ("data", "baseline", "train", "finetuned", "report")


def run(args: list[str], label: str, optional: bool = False) -> bool:
    """Run one stage. ``optional`` stages warn instead of aborting the run."""
    print(f"\n{'=' * 72}\n== {label}\n{'=' * 72}")
    print("$ " + " ".join(args[1:]), flush=True)
    start = time.time()
    proc = subprocess.run([sys.executable, "-m", *args])
    if proc.returncode != 0:
        if optional:
            print(f"[reproduce] skipping {label} (exit {proc.returncode})")
            return False
        raise SystemExit(f"\n[reproduce] stage failed: {label}")
    print(f"[reproduce] {label} took {time.time() - start:.0f}s")
    return True


def exists(path: str, force: bool) -> bool:
    if os.path.exists(path) and not force:
        print(f"[reproduce] skipping, {path} already exists")
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO, "configs",
                                                     "default.yaml"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap QA pairs per inference pass (smoke runs)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--stages", nargs="*", default=list(STAGES),
                    choices=STAGES)
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even when their output exists")
    args = ap.parse_args()

    from vindr_cxr_vqa.config import load_config
    cfg = load_config(args.config)
    out = cfg.paths.output_dir
    pdir = cfg.paths.processed_dir
    adapter = args.adapter or os.path.join(out, "medgemma-lora")
    os.makedirs(out, exist_ok=True)

    cfgarg = ["--config", args.config]
    base_pred = os.path.join(out, "preds_val_base.jsonl")
    ft_pred = os.path.join(out, "preds_val_ft.jsonl")
    stages = set(args.stages)

    # ---- 1. dataset + Table 3 ---------------------------------------------
    if "data" in stages:
        if not exists(os.path.join(pdir, "val.jsonl"), args.force):
            run(["vindr_cxr_vqa.build_dataset", *cfgarg], "Build dataset")
        run(["vindr_cxr_vqa.stats", *cfgarg], "Table 3 + audit vs. the paper")
        run(["vindr_cxr_vqa.evaluate", *cfgarg, "--oracle", "--split", "val",
             "--out", os.path.join(out, "eval_val_oracle.json")],
            "Scorer self-test (ground truth replayed as predictions)")

    limit = ["--limit", str(args.limit)] if args.limit else []

    # ---- 2. zero-shot baseline --------------------------------------------
    if "baseline" in stages and not exists(base_pred, args.force):
        run(["vindr_cxr_vqa.infer", *cfgarg, "--split", "val",
             "--batch-size", str(args.batch_size), "--out", base_pred, *limit],
            "Inference: pretrained baseline (validation)")

    # ---- 3. fine-tune ------------------------------------------------------
    if "train" in stages and not args.skip_train:
        if not exists(os.path.join(adapter, "adapter_config.json"), args.force):
            run(["vindr_cxr_vqa.train", *cfgarg, "--output-dir", adapter,
                 "--max-steps", str(args.max_steps), "--resume", "auto"],
                "Fine-tune MedGemma-4B-it with LoRA")

    # ---- 4. fine-tuned inference ------------------------------------------
    if "finetuned" in stages and not exists(ft_pred, args.force):
        run(["vindr_cxr_vqa.infer", *cfgarg, "--split", "val",
             "--adapter", adapter, "--batch-size", str(args.batch_size),
             "--out", ft_pred, *limit],
            "Inference: fine-tuned model (validation)")

    # ---- 5. score both splits and render -----------------------------------
    if "report" not in stages:
        return

    produced: dict[str, dict[str, str]] = {}
    for split in ("val", "test"):
        for tag, pred in (("base", base_pred), ("ft", ft_pred)):
            if not os.path.exists(pred):
                continue
            dst = os.path.join(out, f"eval_{split}_{tag}.json")
            run(["vindr_cxr_vqa.evaluate", *cfgarg, "--pred", pred,
                 "--split", split, "--out", dst],
                f"Tables 4-6: {tag} on {split}")
            produced.setdefault(split, {})[tag] = dst

    for split, tags in produced.items():
        run(["vindr_cxr_vqa.results",
             *(["--base", tags["base"]] if "base" in tags else []),
             *(["--ft", tags["ft"]] if "ft" in tags else []),
             "--split", split,
             "--out", os.path.join(out, f"results_{split}.md")],
            f"Final tables for {split}")

    if os.path.exists(ft_pred):
        # Optional: the figures need Pillow, which the scoring path does not.
        run(["vindr_cxr_vqa.visualize", *cfgarg, "--split", "val", "--n", "6",
             "--pred-base", base_pred, "--pred-ft", ft_pred,
             "--out", os.path.join(out, "figures")],
            "Figure 2: qualitative overlays", optional=True)

    write_summary(out, produced)


def write_summary(out: str, produced: dict[str, dict[str, str]]) -> None:
    """Stitch the dataset audit and the result tables into one document."""
    parts = ["# VinDr-CXR-VQA — reproduction report", "",
             "Generated by `scripts/reproduce.py`. Every number is recomputed "
             "from this repo; nothing is copied from the paper except the "
             "reference columns.", ""]

    report = os.path.join(out, "dataset_report.md")
    if os.path.exists(report):
        with open(report, encoding="utf-8") as fh:
            body = fh.read().split("\n", 1)[1]
        parts += ["## Dataset (Table 3)", body, ""]

    for split in ("val", "test"):
        path = os.path.join(out, f"results_{split}.md")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            body = fh.read().split("\n", 1)[1]
        title = ("Validation — directly comparable to the paper"
                 if split == "val" else "Test — held-out result")
        parts += [f"## {title}", body, ""]

    figures = os.path.join(out, "figures")
    if os.path.isdir(figures):
        pngs = sorted(f for f in os.listdir(figures) if f.endswith(".png"))
        parts += ["## Figure 2 — qualitative localization", "",
                  f"{len(pngs)} overlay(s) in `outputs/figures/`: "
                  "ground truth green, pretrained yellow, fine-tuned red.", ""]

    dst = os.path.join(out, "REPRODUCTION.md")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))

    print(f"\n{'=' * 72}")
    print(f"[reproduce] wrote {dst}")
    for split, tags in produced.items():
        for tag, path in tags.items():
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            print(f"[reproduce] {split:<5} {tag:<5} "
                  f"VQA F1 {r['vqa']['f1']:.3f}   "
                  f"bbox F1 {r['detection']['f1']:.3f}   "
                  f"mIoU(TP) {r['detection']['miou_tp']:.3f}")


if __name__ == "__main__":
    main()
