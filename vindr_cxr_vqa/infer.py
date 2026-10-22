"""Generate answers for a split and write a predictions JSONL.

Works both for the zero-shot baseline ("Pretrained" in Tables 4-6) and for a
LoRA-adapted checkpoint ("Fine-tuned"), so the two rows of each table come from
the same code path.

    # zero-shot baseline
    python -m vindr_cxr_vqa.infer --split val --out outputs/preds_val_base.jsonl
    # fine-tuned
    python -m vindr_cxr_vqa.infer --split val --adapter outputs/medgemma-lora \
        --out outputs/preds_val_ft.jsonl

:func:`load_model`, :func:`load_processor` and :func:`answer` are the same
three steps split apart, so that a caller holding one loaded model can keep
answering -- the notebook and :mod:`vindr_cxr_vqa.serve` both do, since
loading MedGemma costs minutes and a single answer costs seconds.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor

from .config import load_config
from .modeling import disable_broken_peft_dispatchers, load_base_model
from .prompts import build_messages


def load_model(cfg, adapter: str | None, load_in_4bit: bool | None):
    model, _ = load_base_model(cfg, load_in_4bit=load_in_4bit)
    if adapter:
        from peft import PeftModel
        disable_broken_peft_dispatchers()
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[infer] loaded LoRA adapter from {adapter}")
    return model.eval()


def load_processor(cfg, adapter: str | None = None):
    """The tokenizer + image preprocessor the prompts are built with.

    A LoRA run usually saves the processor next to the adapter; when it did,
    that copy wins, since it is the one the checkpoint was trained against.
    """
    source = cfg.model.name
    if adapter and os.path.exists(
            os.path.join(adapter, "preprocessor_config.json")):
        source = adapter
    processor = AutoProcessor.from_pretrained(source)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Left padding keeps every sequence's continuation at the same offset.
    processor.tokenizer.padding_side = "left"
    return processor


@torch.no_grad()
def answer(model, processor, cfg, records: list[dict], images: list,
           max_new_tokens: int | None = None) -> list[str]:
    """Answer one batch: ``records[i]["question"]`` asked about ``images[i]``.

    Only the ``question`` key is read, so a hand-typed question about an
    uploaded X-ray goes through exactly the path a dataset record does.
    """
    texts = [
        processor.apply_chat_template(
            build_messages(r, img, cfg.data.loc_format),
            add_generation_prompt=True, tokenize=False)
        for r, img in zip(records, images)
    ]
    enc = processor(text=texts, images=[[i] for i in images],
                    return_tensors="pt", padding=True)
    enc = enc.to(next(model.parameters()).device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens or cfg.eval.max_new_tokens,
        do_sample=cfg.generate.do_sample,
        num_beams=cfg.generate.num_beams,
        pad_token_id=processor.tokenizer.pad_token_id)
    # Keep only the continuation, not the echoed prompt.
    new_tokens = gen[:, enc["input_ids"].shape[1]:]
    decoded = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True)
    return [t.strip() for t in decoded]


def run(cfg, split: str, adapter: str | None, out_path: str,
        limit: int | None, batch_size: int,
        load_in_4bit: bool | None = None) -> str:
    processor = load_processor(cfg, adapter)
    model = load_model(cfg, adapter, load_in_4bit)

    path = os.path.join(cfg.paths.processed_dir, f"{split}.jsonl")
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if limit:
        records = records[:limit]
    print(f"[infer] {len(records)} QA pairs from {path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            images = [
                Image.open(os.path.join(cfg.paths.dataset_dir, r["image_path"])
                           ).convert("RGB")
                for r in chunk
            ]
            for r, text in zip(chunk, answer(model, processor, cfg, chunk,
                                             images)):
                out.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "image_id": r["image_id"],
                    "question": r["question"],
                    "prediction": text,
                    "gt_finding": r["gt_finding"],
                    "gt_location": r["gt_location"],
                }, ensure_ascii=False) + "\n")
                written += 1
            print(f"[infer] {written}/{len(records)}", end="\r", flush=True)

    print(f"\n[infer] wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--adapter", default=None,
                    help="LoRA directory; omit for the zero-shot baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-4bit", action="store_true",
                    help="force full-precision weights, overriding the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = "ft" if args.adapter else "base"
    out = args.out or os.path.join(cfg.paths.output_dir,
                                   f"preds_{args.split}_{tag}.jsonl")
    # None means "whatever train.load_in_4bit says"; --no-4bit forces it off.
    run(cfg, args.split, args.adapter, out, args.limit, args.batch_size,
        load_in_4bit=False if args.no_4bit else None)
    print("[infer] next: python -m vindr_cxr_vqa.evaluate "
          f"--pred {out} --split {args.split}")


if __name__ == "__main__":
    main()
