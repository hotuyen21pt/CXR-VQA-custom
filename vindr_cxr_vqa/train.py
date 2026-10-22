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
from transformers import AutoProcessor, Trainer, TrainingArguments

from .config import load_config
from .data import MedGemmaCollator, VinDrCXRVQADataset, digit_token_ids
from .losses import SoftBoxGIoULoss, gather_digit_logits, multitask_loss
from .modeling import (VISION_PREFIXES, accepted_arguments,
                       adapt_training_kwargs, disable_broken_peft_dispatchers,
                       language_side_targets, load_base_model)


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
def load_model(cfg) -> tuple[object, bool]:
    """Load MedGemma and wrap the language side in LoRA adapters.

    Returns ``(model, used_4bit)``; the caller needs the flag to choose
    between a paged 8-bit optimizer and plain AdamW.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, used_4bit = load_base_model(cfg)
    if used_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)

    if cfg.lora.freeze_vision_tower:
        for name, param in model.named_parameters():
            if "vision_tower" in name or "multi_modal_projector" in name:
                param.requires_grad = False

    # Keep the adapters off the vision tower, matching the frozen encoder.
    # Passing the bare suffixes straight through would not do that: peft
    # matches by suffix, so they also wrap the vision tower and undo the freeze
    # above. See modeling.language_side_targets.
    targets = list(cfg.lora.target_modules)
    if cfg.lora.freeze_vision_tower:
        names = [n for n, _ in model.named_modules()]
        targets = language_side_targets(names, targets)
        if not targets:
            raise SystemExit(
                "[train] no language-side module matched "
                f"lora.target_modules={list(cfg.lora.target_modules)}.\n"
                "Check the names against the model: every match was inside "
                f"{' / '.join(VISION_PREFIXES)}.")
        skipped = len(language_side_targets(
            names, cfg.lora.target_modules, exclude=())) - len(targets)
        print(f"[train] LoRA on {len(targets)} language-side modules; "
              f"{skipped} vision-tower modules left frozen")

    peft_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    disable_broken_peft_dispatchers()
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, used_4bit


def build_training_arguments(**kwargs) -> TrainingArguments:
    """``TrainingArguments(**kwargs)``, reconciled with this transformers.

    The 5.x line kept renaming and retiring arguments -- 5.15 dropped
    ``warmup_ratio`` outright -- and a keyword this version has never heard of
    is a TypeError, raised only once the weights are already on the GPU. So
    the names are checked against the real signature first, and every
    substitution is printed.
    """
    kwargs, notes = adapt_training_kwargs(
        kwargs, accepted_arguments(TrainingArguments.__init__))
    for note in notes:
        print(f"[train] {note}")
    return TrainingArguments(**kwargs)


def latest_checkpoint(out_dir: str) -> str | None:
    """Newest ``checkpoint-N`` directory under ``out_dir``, by step number."""
    if not os.path.isdir(out_dir):
        return None
    ckpts = [(int(n.split("-")[1]), os.path.join(out_dir, n))
             for n in os.listdir(out_dir)
             if n.startswith("checkpoint-") and n.split("-")[-1].isdigit()]
    return max(ckpts)[1] if ckpts else None


def warn_if_nothing_will_be_saved(cfg, n_samples: int, max_steps: int) -> None:
    """A capped run with per-epoch saving writes no intermediate checkpoint.

    On a time-limited machine that means an interrupted session loses
    everything, so say so loudly before the hours are spent.
    """
    if cfg.train.save_strategy != "epoch" or max_steps <= 0:
        return
    per_epoch = max(1, n_samples // (cfg.train.batch_size * cfg.train.grad_accum))
    if max_steps < per_epoch:
        print(f"[train] WARNING: --max-steps {max_steps} ends before the first "
              f"epoch ({per_epoch} steps) and save_strategy is 'epoch', so no "
              "checkpoint will be written until the run finishes.\n"
              "[train]          Set train.save_strategy: steps (with "
              "save_steps) to keep the run resumable.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap the run; useful for a smoke test")
    ap.add_argument("--resume", default=None,
                    help="checkpoint directory, 'auto' for the newest one, or "
                         "'none' to start fresh even when checkpoints exist")
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
    warn_if_nothing_will_be_saved(cfg, len(train_ds), args.max_steps)

    resume = args.resume
    if resume in ("none", "no", "false", ""):
        # A caller that decided to start over -- a changed learning rate or a
        # different set of LoRA targets makes a saved optimizer state wrong --
        # needs a way to say so that is not "delete the checkpoints".
        resume = None
        print("[train] --resume none: starting fresh, existing checkpoints "
              "are left alone")
    elif resume == "auto":
        resume = latest_checkpoint(out_dir)
        print(f"[train] resuming from {resume}" if resume
              else "[train] no checkpoint to resume from; starting fresh")

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

    model, used_4bit = load_model(cfg)

    bf16 = (cfg.train.precision == "bf16"
            or (cfg.train.precision == "auto" and torch.cuda.is_bf16_supported()))
    targs = build_training_arguments(
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
        # The paged optimizer is a bitsandbytes feature, so it is only
        # available when the base weights were actually quantized.
        optim="paged_adamw_8bit" if used_4bit else "adamw_torch",
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        # Only meaningful for save_strategy="steps"; harmless otherwise.
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
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
    trainer.train(resume_from_checkpoint=resume)

    # The adapter lands in out_dir itself, next to the checkpoint-N folders,
    # so --adapter <out_dir> is what inference loads.
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
