"""LoRA fine-tuning of MedGemma-4B-it on VinDr-CXR-VQA (Section 3.1).

    L_total = L_VQA + 0.5 * L_bbox

with LoRA r=16, alpha=32, dropout=0.1, AdamW at 2e-5, weight decay 0.01,
batch 4 x grad-accum 4 (effective 16), 3 epochs, fp16 mixed precision.
Checkpoints are written at the end of every epoch and the best one is chosen by
validation F1, as the paper describes.

    python -m vindr_cxr_vqa.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          Trainer, TrainingArguments)

from .config import load_config
from .data import MedGemmaCollator, VinDrCXRVQADataset, digit_token_ids
from .losses import SoftBoxGIoULoss, gather_digit_logits, multitask_loss


class MultiTaskTrainer(Trainer):
    """Adds the IoU-based box term to the usual language-modelling loss."""

    def __init__(self, *args, bbox_loss_fn=None, lambda_bbox: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.bbox_loss_fn = bbox_loss_fn
        self.lambda_bbox = lambda_bbox
        self._bbox_running = 0.0
        self._bbox_steps = 0

    def compute_loss(self, model, inputs, return_outputs: bool = False,
                     **kwargs):
        index = inputs.pop("bbox_sample_index", None)
        positions = inputs.pop("bbox_positions", None)
        targets = inputs.pop("bbox_targets", None)

        outputs = model(**inputs)
        loss = outputs.loss

        bbox = None
        if (self.bbox_loss_fn is not None and positions is not None
                and positions.numel() > 0):
            logits = outputs.logits[index]
            digits = gather_digit_logits(logits, positions.to(logits.device))
            bbox = self.bbox_loss_fn(digits, targets.to(logits.device))
            self._bbox_running += float(bbox.detach())
            self._bbox_steps += 1

        total = multitask_loss(loss, bbox, self.lambda_bbox)
        return (total, outputs) if return_outputs else total

    def log(self, logs: dict, *args, **kwargs) -> None:
        if self._bbox_steps:
            logs["bbox_loss"] = round(self._bbox_running / self._bbox_steps, 4)
            self._bbox_running = 0.0
            self._bbox_steps = 0
        super().log(logs, *args, **kwargs)


# --------------------------------------------------------------------------
def load_model(cfg):
    """Load MedGemma and wrap the language side in LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = None
    if cfg.train.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported()
                else torch.float16),
        )

    dtype = cfg.model.torch_dtype
    if dtype == "auto":
        dtype = (torch.bfloat16 if torch.cuda.is_available()
                 and torch.cuda.is_bf16_supported() else torch.float16)

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        quantization_config=quant,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)

    if cfg.lora.freeze_vision_tower:
        for name, param in model.named_parameters():
            if "vision_tower" in name or "multi_modal_projector" in name:
                param.requires_grad = False

    peft_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        # Keep the adapters off the vision tower, matching the frozen encoder.
        modules_to_save=None,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap the run; useful for a smoke test")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output_dir or os.path.join(cfg.paths.output_dir, "medgemma-lora")
    os.makedirs(out_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model.name)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_ds = VinDrCXRVQADataset(
        os.path.join(cfg.paths.processed_dir, "train.jsonl"),
        cfg.paths.dataset_dir)
    print(f"[train] {len(train_ds)} training QA pairs")

    collator = MedGemmaCollator(processor, cfg)

    bbox_loss_fn = None
    if cfg.loss.bbox_loss == "giou":
        if cfg.data.loc_format != "norm1000":
            print("[train] loc_format is not norm1000; the soft-IoU term "
                  "needs fixed-width coordinates and is disabled")
        else:
            bbox_loss_fn = SoftBoxGIoULoss(
                digit_token_ids(processor.tokenizer),
                digits_per_coord=collator.digits_per_coord)

    model = load_model(cfg)

    bf16 = (cfg.train.precision == "bf16"
            or (cfg.train.precision == "auto" and torch.cuda.is_bf16_supported()))
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.train.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=float(cfg.train.lr),
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler,
        max_grad_norm=cfg.train.max_grad_norm,
        optim="paged_adamw_8bit" if cfg.train.load_in_4bit else "adamw_torch",
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        save_total_limit=cfg.train.epochs,
        seed=cfg.train.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        bbox_loss_fn=bbox_loss_fn,
        lambda_bbox=cfg.loss.lambda_bbox,
    )
    trainer.train(resume_from_checkpoint=args.resume)

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    if collator.n_bbox_skipped:
        print(f"[train] bbox loss skipped on {collator.n_bbox_skipped} samples "
              "whose coordinate span could not be located")
    print(f"[train] adapters saved to {out_dir}")
    print("[train] next: python -m vindr_cxr_vqa.infer --adapter "
          f"{out_dir} --split val")


if __name__ == "__main__":
    main()
