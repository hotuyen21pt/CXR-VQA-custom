"""Torch dataset and collator for MedGemma fine-tuning.

Requires torch / transformers / Pillow.  Everything that only builds or scores
the dataset lives in modules that import none of those, so a CPU-only machine
can still run the full data pipeline.
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import LOC_NORM_RANGE
from .prompts import build_messages, build_target


class VinDrCXRVQADataset(Dataset):
    """One item per QA pair, paired with its chest X-ray."""

    def __init__(self, jsonl_path: str, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(jsonl_path, encoding="utf-8") as fh:
            self.records = [json.loads(line) for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        path = os.path.join(self.dataset_dir, r["image_path"])
        # Decode here rather than in the collator so DataLoader workers do it.
        r = dict(r)
        r["_image"] = Image.open(path).convert("RGB")
        return r


class MedGemmaCollator:
    """Build padded model inputs with the prompt masked out of the labels.

    Also locates the coordinate digit tokens inside each target so the
    soft-IoU term of the multi-task loss knows where to read.  When the span
    cannot be located -- a malformed record, or ``loc_format: pixel``, whose
    coordinates are not fixed width -- that sample simply contributes no bbox
    loss, and the count is exposed as :attr:`n_bbox_skipped`.
    """

    def __init__(self, processor, cfg):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.cfg = cfg
        self.box_source = cfg.data.box_source
        self.loc_format = cfg.data.loc_format
        self.include_reason = cfg.data.include_reason
        self.max_len = cfg.data.max_seq_length
        self.digits_per_coord = len(str(LOC_NORM_RANGE - 1))
        self.n_coord_tokens = 4 * self.digits_per_coord
        self.n_bbox_skipped = 0

        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.image_token_ids = _image_token_ids(processor)

    # ------------------------------------------------------------------
    def __call__(self, batch: list[dict]) -> dict:
        encoded = [self._encode(r) for r in batch]
        encoded = [e for e in encoded if e is not None]
        if not encoded:
            raise RuntimeError("every sample in the batch failed to encode")

        width = max(e["input_ids"].size(0) for e in encoded)
        input_ids, labels, attn, pos = [], [], [], []
        for e in encoded:
            pad = width - e["input_ids"].size(0)
            input_ids.append(torch.cat(
                [e["input_ids"], e["input_ids"].new_full((pad,), self.pad_id)]))
            labels.append(torch.cat(
                [e["labels"], e["labels"].new_full((pad,), -100)]))
            attn.append(torch.cat(
                [torch.ones_like(e["input_ids"]),
                 e["input_ids"].new_zeros(pad)]))
            pos.append(e["loc_positions"])

        out = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": torch.cat([e["pixel_values"] for e in encoded]),
        }
        # Boxes are supervised only for samples whose digit span was found.
        has_box = [i for i, p in enumerate(pos) if p is not None]
        if has_box:
            out["bbox_sample_index"] = torch.tensor(has_box, dtype=torch.long)
            out["bbox_positions"] = torch.stack([pos[i] for i in has_box])
            out["bbox_targets"] = torch.stack(
                [encoded[i]["bbox_target"] for i in has_box])
        return out

    # ------------------------------------------------------------------
    def _encode(self, record: dict) -> dict | None:
        image = record["_image"]
        messages = build_messages(record, image, self.loc_format)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        target_text, loc_span = build_target(
            record, self.box_source, self.loc_format, self.include_reason)
        target_text += self.tokenizer.eos_token

        full = self.processor(text=prompt_text + target_text, images=[image],
                              return_tensors="pt")
        prompt = self.processor(text=prompt_text, images=[image],
                                return_tensors="pt")
        ids = full["input_ids"][0]
        n_prompt = prompt["input_ids"].shape[1]
        if ids.size(0) > self.max_len:
            ids = ids[: self.max_len]
        if n_prompt >= ids.size(0):
            return None  # the answer was truncated away entirely

        labels = ids.clone()
        labels[:n_prompt] = -100
        labels[ids == self.pad_id] = -100
        for tid in self.image_token_ids:
            labels[ids == tid] = -100

        positions = self._locate_digits(target_text, loc_span, n_prompt,
                                        ids.size(0))
        if positions is None:
            self.n_bbox_skipped += 1

        return {
            "input_ids": ids,
            "labels": labels,
            "pixel_values": full["pixel_values"],
            "loc_positions": positions,
            "bbox_target": _normalized_target(record, self.box_source),
        }

    # ------------------------------------------------------------------
    def _locate_digits(self, target_text: str, loc_span: tuple[int, int],
                       offset: int, seq_len: int) -> torch.Tensor | None:
        """Label indices of the coordinate digit tokens, or ``None``.

        Offsets come from tokenising the target on its own; that is safe here
        because the target always begins at a turn boundary, and the result is
        checked -- the span must contain exactly ``4 * digits_per_coord``
        single-digit tokens, otherwise the sample is skipped rather than
        supervised against the wrong positions.
        """
        if self.loc_format != "norm1000":
            return None
        try:
            enc = self.tokenizer(target_text, add_special_tokens=False,
                                 return_offsets_mapping=True)
        except (NotImplementedError, ValueError):
            return None  # slow tokenizer without offset support

        lo, hi = loc_span
        found = [
            i for i, (s, e) in enumerate(enc["offset_mapping"])
            if s >= lo and e <= hi and target_text[s:e].strip().isdigit()
            and len(target_text[s:e].strip()) == 1
        ]
        if len(found) != self.n_coord_tokens:
            return None
        pos = [offset + i for i in found]
        if max(pos) >= seq_len:
            return None  # truncated out of the sequence
        return torch.tensor(pos, dtype=torch.long)


# --------------------------------------------------------------------------
def _normalized_target(record: dict, box_source: str) -> torch.Tensor:
    """Ground-truth box on the ``[0, 1]`` scale the soft-IoU loss works in."""
    key = "bbox_fixed_norm" if box_source == "fixed" else "bbox_norm"
    scale = float(LOC_NORM_RANGE - 1)
    return torch.tensor([v / scale for v in record[key]], dtype=torch.float32)


def _image_token_ids(processor) -> list[int]:
    """Token ids that stand in for image patches, which must not be learned."""
    ids: set[int] = set()
    tok = processor.tokenizer
    for attr in ("image_token", "boi_token", "eoi_token"):
        text = getattr(processor, attr, None) or getattr(tok, attr, None)
        if isinstance(text, str):
            tid = tok.convert_tokens_to_ids(text)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
    for attr in ("image_token_id", "boi_token_id", "eoi_token_id"):
        tid = getattr(processor, attr, None)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return sorted(ids)


def digit_token_ids(tokenizer) -> list[int]:
    """Token id of each digit 0-9 as it appears mid-string.

    Gemma tokenises digits individually, so a coordinate is a run of
    single-digit tokens -- the property the soft-IoU decoder relies on.
    """
    ids: list[int] = []
    for d in "0123456789":
        enc = tokenizer(d, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise RuntimeError(
                f"tokenizer does not emit digit {d!r} as a single token "
                f"(got {enc}); set loss.bbox_loss: none in the config")
        ids.append(enc[0])
    return ids
