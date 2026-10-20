"""Generate answers for a split and write a predictions JSONL.

Works both for the zero-shot baseline ("Pretrained" in Tables 4-6) and for a
LoRA-adapted checkpoint ("Fine-tuned"), so the two rows of each table come from
the same code path.

    # zero-shot baseline
    python -m vindr_cxr_vqa.infer --split val --out outputs/preds_val_base.jsonl
    # fine-tuned
    python -m vindr_cxr_vqa.infer --split val --adapter outputs/medgemma-lora \
        --out outputs/preds_val_ft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import load_config
from .prompts import build_messages


def load_model(cfg, adapter: str | None, load_in_4bit: bool):
    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_available()
                and torch.cuda.is_bf16_supported() else torch.float16))

    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.name, torch_dtype=dtype, quantization_config=quant,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[infer] loaded LoRA adapter from {adapter}")
    return model.eval()


@torch.no_grad()
def run(cfg, split: str, adapter: str | None, out_path: str,
        limit: int | None, batch_size: int, load_in_4bit: bool) -> str:
    processor = AutoProcessor.from_pretrained(
        adapter if adapter and os.path.exists(
            os.path.join(adapter, "preprocessor_config.json")) else cfg.model.name)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Left padding keeps every sequence's continuation at the same offset.
    processor.tokenizer.padding_side = "left"

    model = load_model(cfg, adapter, load_in_4bit)

    path = os.path.join(cfg.paths.processed_dir, f"{split}.jsonl")
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if limit:
        records = records[:limit]
    print(f"[infer] {len(records)} QA pairs from {path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    device = next(model.parameters()).device
    written = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            images, texts = [], []
            for r in chunk:
                img = Image.open(
                    os.path.join(cfg.paths.dataset_dir, r["image_path"])
                ).convert("RGB")
                images.append(img)
                texts.append(processor.apply_chat_template(
                    build_messages(r, img, cfg.data.loc_format),
                    add_generation_prompt=True, tokenize=False))

            enc = processor(text=texts, images=[[i] for i in images],
                            return_tensors="pt", padding=True).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=cfg.eval.max_new_tokens,
                do_sample=cfg.generate.do_sample,
                num_beams=cfg.generate.num_beams,
                pad_token_id=processor.tokenizer.pad_token_id)
            # Keep only the continuation, not the echoed prompt.
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True)

            for r, text in zip(chunk, decoded):
                out.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "image_id": r["image_id"],
                    "question": r["question"],
                    "prediction": text.strip(),
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
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = "ft" if args.adapter else "base"
    out = args.out or os.path.join(cfg.paths.output_dir,
                                   f"preds_{args.split}_{tag}.jsonl")
    run(cfg, args.split, args.adapter, out, args.limit, args.batch_size,
        load_in_4bit=not args.no_4bit)
    print("[infer] next: python -m vindr_cxr_vqa.evaluate "
          f"--pred {out} --split {args.split}")


if __name__ == "__main__":
    main()
