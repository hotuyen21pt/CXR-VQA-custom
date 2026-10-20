# VinDr-CXR-VQA

Reproduction of **"VinDr-CXR-VQA: A Visual Question Answering Dataset for
Explainable Chest X-ray Analysis with Multi-Task Learning"**
([arXiv:2511.00504v2](https://arxiv.org/abs/2511.00504)).

Joins the released QA annotations
([`Dangindev/VinDR-CXR-VQA`](https://huggingface.co/datasets/Dangindev/VinDR-CXR-VQA))
with a local VinDr-CXR image copy, fine-tunes MedGemma-4B-it with LoRA under
the paper's multi-task objective, and scores Tables 4, 5 and 6.

> **Read [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) before training.** The
> released data departs from the paper in four ways, one of them consequential:
> in **60% of QA pairs the ground-truth box belongs to a different pathology
> than the label it is paired with**. The repo ships both the published box and
> a repaired one, and the choice is a config switch.

---

## Layout

```
configs/default.yaml        every hyper-parameter, taken from Section 3.1
vindr_cxr_vqa/
  constants.py              14 pathologies, 6 question types, synonyms
  boxes.py                  <loc_...> parsing, coordinate spaces, IoU, matching
  vindr.py                  VinDr-CXR CSV + PNG readers, consensus box fusion
  build_dataset.py          join + split + repair  ->  data/processed/*.jsonl
  stats.py                  recompute Table 3 and audit it against the paper
  prompts.py                prompt and supervision-target construction
  losses.py                 differentiable soft-GIoU over coordinate tokens
  data.py                   torch Dataset + MedGemma collator
  train.py                  LoRA fine-tuning with L_VQA + 0.5 * L_bbox
  infer.py                  generation -> predictions JSONL
  evaluate.py               Tables 4, 5 and 6
  visualize.py              Figure 2 overlays
scripts/
  make_kaggle_bundle.py     package repo + data + images into one zip
  make_notebook.py          regenerate the Kaggle notebook
notebooks/kaggle_train.ipynb
tests/                      pytest, no GPU needed
```

**Nothing in the data path needs a GPU, torch, or even pandas.**
`build_dataset`, `stats` and `evaluate` are standard library only, so the
dataset can be assembled and scored on any machine; only `train` / `infer` /
`visualize` need the stack in `requirements.txt`.

---

## Expected input layout

```
dataset/
  annotations_train.csv        image_id, rad_id, class_name, x_min…y_max
  annotations_test.csv
  image_labels_train.csv
  image_labels_test.csv
  train_data/train/*.png       15,000 images, pre-resized to 224×224
  train_data/train/resize_factors.csv
  test_data/test/*.png         3,000 images
```

All 4,394 images referenced by `data_v1.json` live in `train_data/train/`.
`data_v1.json` is downloaded from HuggingFace on first build.

---

## Quick start

```bash
# 1. Build the dataset (~30 s, no dependencies)
python -m vindr_cxr_vqa.build_dataset --config configs/default.yaml

# 2. Verify it against the paper
python -m vindr_cxr_vqa.stats
#    -> outputs/dataset_report.md   (25/29 checks match; see docs/DATA_NOTES.md)

# 3. Self-test the scorer: replay the ground truth as predictions
python -m vindr_cxr_vqa.evaluate --oracle --split val
#    -> VQA precision 1.000, localization 100% at IoU>=0.5
```

Training needs a GPU:

```bash
pip install -r requirements.txt
huggingface-cli login            # google/medgemma-4b-it is gated

python -m vindr_cxr_vqa.train  --config configs/default.yaml
python -m vindr_cxr_vqa.infer  --split val --out outputs/preds_val_base.jsonl
python -m vindr_cxr_vqa.infer  --split val --adapter outputs/medgemma-lora \
                               --out outputs/preds_val_ft.jsonl
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_ft.jsonl --split val
```

---

## Running on Kaggle

```bash
python -m vindr_cxr_vqa.build_dataset
python scripts/make_kaggle_bundle.py
#    -> outputs/vindr-cxr-vqa-kaggle.zip   (code + JSONL + the 4,394 PNGs)
```

1. **kaggle.com/datasets → New Dataset** → upload the zip. Kaggle unpacks it to
   `/kaggle/input/<slug>/`.
2. **Code → New Notebook → File → Import Notebook** →
   `notebooks/kaggle_train.ipynb`.
3. *Add Input* → attach the dataset you just created.
4. *Settings* → **Accelerator: GPU T4 x2** (or P100), **Internet: On**.
5. *Add-ons → Secrets* → add `HF_TOKEN` (accept the
   [MedGemma licence](https://huggingface.co/google/medgemma-4b-it) first).
6. Run all.

The notebook rewrites the config for a 16 GB GPU: 4-bit base weights, batch 1
with gradient accumulation 16 to keep the paper's effective batch of 16, and
fp16 on pre-Ampere cards. A Kaggle session caps at 12 h against the paper's
~36 h on an A6000, so `MAX_STEPS` defaults to a capped pass; set it to `-1`
and resume from the checkpoint across sessions for the full 3 epochs.

Use `--no-images` if you would rather attach the X-rays as a separate dataset:

```bash
python scripts/make_kaggle_bundle.py --no-images   # a few MB
```

---

## What the built dataset looks like

One JSON object per QA pair in `data/processed/{train,val,test}.jsonl`:

```json
{
  "sample_id": "9a5094b2563a1ef3ff50dc5c7ff71345_0",
  "image_id":  "9a5094b2563a1ef3ff50dc5c7ff71345",
  "image_path": "train_data/train/9a5094b2563a1ef3ff50dc5c7ff71345.png",
  "split": "train",
  "question": "Where is the Cardiomegaly?",
  "answer":   "The Cardiomegaly finding refers to … <loc_691_1375_1653_1831>.",
  "reason":   "Cardiomegaly is a nonspecific finding often indicative of …",
  "type": "Where", "difficulty": "Medium", "gt_finding": "Cardiomegaly",

  "gt_location": "<loc_691_1375_1653_1831>",
  "bbox_orig":    [691, 1375, 1653, 1831],
  "bbox_resized": [66.3, 108.6, 158.6, 144.6],
  "bbox_norm":    [330, 442, 790, 589],
  "loc_norm":     "<loc_330_442_790_589>",

  "box_label_consistent": true,
  "repair_method": "published",
  "gt_location_fixed": "<loc_691_1375_1653_1831>",
  "bbox_fixed_orig": [691, 1375, 1653, 1831],

  "image_width": 2084, "image_height": 2048,
  "resized_width": 224, "resized_height": 224
}
```

`data/processed/images.json` indexes all 4,394 images with their per-image
multi-label annotations, every raw radiologist box (`boxes_raw`, the ground
truth for Table 5) and the fused consensus boxes (`boxes_fused`).

---

## Rebuilt statistics

| Attribute | Paper | Rebuilt |
| --- | --- | --- |
| Source images | 4,394 | 4,394 |
| QA pairs | 17,597 | 17,597 |
| QA per image | 4.0 | 4.0 |
| Pathology classes | 14 | 14 |
| Question types | 6, balanced 16.4–17.0% | 6, 16.4–17.0% |
| Train / val / test images | 3,735 / 659 / 300 | 3,735 / 659 / 300 |
| Validation ground-truth boxes | 5,471 | 5,561 |
| Boxes per validation image | 8.3 | 8.44 |
| `gt_location` matches `gt_finding` | 100% | **40.0%** |
| `reason` length | 150–250 words | **4–40 (mean 17.8)** |
| Negative samples | 58.3% | **0%** |
| Validation images with ≥2 pathologies | 54.8% | **92.7%** |

Regenerate with `python -m vindr_cxr_vqa.stats`.

Table 3 lists 3,735 + 659 = 4,394 images *and* a 300-image test split, so test
cannot be a third partition — it is a stratified subset of validation, which is
also the set Tables 4–6 are computed on. Pass `--disjoint-test` to carve it out
instead.

---

## Method notes

**Multi-task loss.** Section 3.1 specifies
`L_total = L_VQA + λ·L_bbox` with `λ = 0.5`, cross-entropy on answer tokens and
"IoU-based regression" on boxes "predicted via special tokens embedded in the
generated answer" — with no architectural change. `losses.py` realises that
directly: at each coordinate digit position the model emits a distribution over
the ten digit tokens, so the *expected* digit is differentiable in the logits;
three expected digits assemble a coordinate, four coordinates a soft box, and
GIoU against the ground truth back-propagates into the LM head. No regression
head is added.

This requires fixed-width coordinates, which is why `data.loc_format` defaults
to `norm1000` (three digits, 0–999). Set it to `pixel` to reproduce the
released answer strings verbatim; the soft-IoU term then disables itself and
training falls back to plain cross-entropy.

**Box matching.** Section 3.2 calls a prediction a true positive if it overlaps
*any* ground-truth box at IoU ≥ 0.3, and counts unmatched ground truth as false
negatives. Those two statements are only consistent under one-to-one matching,
so `eval.match` defaults to `greedy` (descending IoU). `--match any` gives the
literal reading.

**Table 4's Accuracy column.** The paper prints the same value for Accuracy and
F1 in both rows (0.558 and 0.624). Those are not equal in general, and the
printed F1 is exactly `2PR/(P+R)` of the printed precision and recall — so the
Accuracy column looks like a duplicate of F1. `evaluate.py` reports both: `f1`
is the comparable number and `accuracy` is the genuine `(TP+TN)/total` over all
image × class decisions.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers box parsing and coordinate round-trips, IoU and greedy matching, the
split arithmetic, the config parser, the negation-aware answer parser, and the
soft-box decoder (torch-dependent tests skip automatically when torch is
absent).

---

## Citation

```bibtex
@misc{nguyen2025vindrcxrvqa,
  title  = {VinDr-CXR-VQA: A Visual Question Answering Dataset for Explainable
            Chest X-Ray Analysis with Multi-Task Learning},
  author = {Dang H. Nguyen and Hieu H. Pham and Hao T. Nguyen and Hieu H. Pham},
  year   = {2025},
  eprint = {2511.00504},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

VinDr-CXR and VinDr-CXR-VQA are both released under CC BY 4.0.
