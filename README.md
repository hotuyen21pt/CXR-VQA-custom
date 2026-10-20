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
  results.py                baseline vs fine-tuned, rendered as final tables
  visualize.py              Figure 2 overlays
  preflight.py              check packages, GPU, weights and data before training
scripts/
  reproduce.py              run every result the paper reports, end to end
  make_kaggle_bundle.py     package repo + data + images into one zip
  make_notebook.py          regenerate the Kaggle notebook
  fetch_model.py            download MedGemma for offline/air-gapped use
notebooks/kaggle_train.ipynb
docs/  (in Vietnamese)
  PIPELINE.md               one record traced through every stage
  DATA_NOTES.md             what the released data actually contains
  EVALUATION.md             the metrics explained from scratch
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
#    -> outputs/dataset_report.md   (25/30 checks match; see docs/DATA_NOTES.md)

# 3. Self-test the scorer: replay the ground truth as predictions
python -m vindr_cxr_vqa.evaluate --oracle --split val
#    -> VQA precision 1.000, localization 100% at IoU>=0.5
```

### Reproducing every result in the paper

One command runs the lot — Table 3, Tables 4–6 for both the baseline and the
fine-tuned model on both splits, Figure 2, and a consolidated
`outputs/REPRODUCTION.md`:

```bash
pip install -r requirements.txt
huggingface-cli login            # google/medgemma-4b-it is gated

python scripts/reproduce.py                 # everything
python scripts/reproduce.py --stages data   # dataset + audit only, no GPU
python scripts/reproduce.py --limit 200     # quick smoke run
python scripts/reproduce.py --skip-train    # score an existing adapter
```

Each stage is skipped when its output already exists, so an interrupted run can
just be restarted. Inference runs **once per model, over validation** — the
test split is a subset of it, so the same predictions score both: validation
for the paper-comparable numbers, test for the held-out result.

The individual steps, if you want them separately:

```bash
python -m vindr_cxr_vqa.preflight --config configs/default.yaml
python -m vindr_cxr_vqa.train     --config configs/default.yaml
python -m vindr_cxr_vqa.infer --split val --out outputs/preds_val_base.jsonl
python -m vindr_cxr_vqa.infer --split val --adapter outputs/medgemma-lora \
                              --out outputs/preds_val_ft.jsonl
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_ft.jsonl --split test
python -m vindr_cxr_vqa.results --base outputs/eval_test_base.json \
    --ft outputs/eval_test_ft.json --split test
```

### Checkpoints and resuming

`train.save_strategy` defaults to `epoch`, matching the paper. One epoch is
~935 optimizer steps, so on a time-limited machine switch to:

```yaml
train:
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3
```

Checkpoints land in `<output-dir>/checkpoint-N` and `--resume auto` picks the
newest one:

```bash
python -m vindr_cxr_vqa.train --max-steps 600 --resume auto
```

`train.py` warns if `--max-steps` would end before the first epoch while
`save_strategy` is still `epoch` — that combination writes no intermediate
checkpoint, so an interrupted run loses everything. The Kaggle notebook sets
the step-based strategy automatically.

---

## Running on Kaggle

```bash
python -m vindr_cxr_vqa.build_dataset
python scripts/make_kaggle_bundle.py
#    -> outputs/vindr-cxr-vqa-kaggle.zip   (code + JSONL + the 4,394 PNGs)
```

1. **kaggle.com/datasets → New Dataset** → upload the zip. Kaggle unpacks it,
   so the contents land under `/kaggle/input/…/<slug>/`.
2. **Code → New Notebook → File → Import Notebook** →
   `notebooks/kaggle_train.ipynb`. **Import before attaching anything** —
   Kaggle applies a notebook's stored settings on import. This notebook ships
   no accelerator, data-source or internet metadata precisely so nothing of
   yours is overwritten, but importing on top of a configured notebook can
   still reset the panel.
3. *Add Input* → attach the dataset you created, and paste its path into
   `BUNDLE` in the setup cell.
4. *Add Input → Models* → attach MedGemma 4B-it or MedGemma 1.5 4B-it
   (Transformers), and paste its path into `MODEL_PATH` in the model cell.
5. *Settings* → **Accelerator: GPU T4 x2** (or P100).
6. Run all.

Kaggle nests inputs differently depending on how they were attached —
`/kaggle/input/<slug>` for some, `/kaggle/input/datasets/<user>/<slug>` and
`/kaggle/input/models/<user>/<model>/<framework>/<variation>/<version>/` for
others. Both cells walk the tree, so leaving `BUNDLE` and `MODEL_PATH` empty
works too; pasting them just gives a clearer error when something is missing.

### Running with Internet off

Detected automatically — the second cell opens a socket to pypi.org with a
short timeout and sets `OFFLINE` from the result (override it by assigning
`True`/`False` instead of `None`). This matters: with Internet off but the flag
left false, `pip install` spends several minutes retrying DNS before failing.

Offline skips `pip install` and disables every hub lookup
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), so two things must hold:

- **The weights are attached** (step 4). Nothing can be downloaded. If no
  MedGemma model is available to attach, run `python scripts/fetch_model.py`
  on a machine with internet — ≈8 GB, gated, so accept the licence first — and
  upload the resulting folder as a Kaggle Dataset.
- **Kaggle's preinstalled `transformers` is ≥ 4.50.** MedGemma — both 4B-it and
  1.5 4B-it — is a Gemma 3 model, and Google's model card gives 4.50.0 as the
  first release supporting it. If Kaggle's is older you need internet once, or
  a Dataset of wheels to `pip install --no-index` from.

No `HF_TOKEN` is needed in this mode — it only ever authenticates a download.
The built dataset ships inside the bundle, so `data_v1.json` is never fetched
either.

Check everything at once before committing a GPU session to it:

```bash
python -m vindr_cxr_vqa.preflight --config configs/kaggle.yaml
```

It reports package versions, GPU and VRAM, whether the model directory really
holds `config.json` + `*.safetensors`, and whether the dataset's image paths
resolve — exiting non-zero if anything is blocking. The notebook runs it
automatically right after setup.

The notebook rewrites the config for a 16 GB GPU: 4-bit base weights, batch 1
with gradient accumulation 16 to keep the paper's effective batch of 16, and
fp16 on pre-Ampere cards. It trains, then runs inference and scoring on the
**test** split and prints the final Tables 4–6 next to the published numbers.

A Kaggle session caps at 12 h against the paper's ~36 h on an A6000, so
`MAX_STEPS` defaults to a capped pass. Checkpoints are written every 100 steps
and the train cell resumes from the newest one, so the full 3 epochs can be
spread over several sessions — but `/kaggle/working` is wiped at the end of
each, so use *Save Version → Save & Run All* and attach the previous run's
output as an input. The train cell copies any `medgemma-lora` found under
`/kaggle/input` back into place before resuming.

### Updating the code on Kaggle without re-uploading 362 MB

Of that bundle, 356 MB is images that never change. Once the data is on
Kaggle, split it in two and the images stay put:

```bash
python scripts/make_kaggle_bundle.py --part code     # 6.3 MB — the package + JSONL
python scripts/make_kaggle_bundle.py --part images   # 356 MB — upload once
```

Attach both as inputs. The setup cell locates the code, the JSONL and the
images independently, so they can share one dataset or sit in two. Later code
fixes are then a 6 MB new version of the code dataset.

The setup cell also fails immediately if the attached code is missing a module
the notebook needs, rather than halfway through a GPU session.

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
| Training samples (Section 3.1) | 20,880 | **14,960** |

25 of 30 checks match. Regenerate with `python -m vindr_cxr_vqa.stats`.

The last row is worth flagging separately: Section 3.1 trains on 20,880
samples, but the entire release holds 17,597 QA pairs, so the stated training
set is larger than all the data there is. 85% of 17,597 is the 14,960 here.

Table 3 lists 3,735 + 659 = 4,394 images *and* a 300-image test split, so test
cannot be a third partition — it is a stratified subset of validation, which is
also the set Tables 4–6 are computed on. Pass `--disjoint-test` to carve it out
instead.

---

## Evaluation, step by step

New to computer vision? **[docs/EVALUATION.md](docs/EVALUATION.md)** explains
all of this from scratch — what IoU is, what precision and recall mean, why
accuracy is misleading here, and how to read the output — with worked
arithmetic on real records. **[docs/PIPELINE.md](docs/PIPELINE.md)** traces a
single record through every stage, from the raw JSON to its row in the table.
*(Both are in Vietnamese.)*

The short version. Three commands, in order:

```bash
# 1. the model answers every question on the split, writing what it said
python -m vindr_cxr_vqa.infer --split val --out outputs/preds_val_ft.jsonl

# 2. grade those answers -> the three tables
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_ft.jsonl --split val

# 3. put baseline and fine-tuned side by side
python -m vindr_cxr_vqa.results --base outputs/eval_val_base.json \
    --ft outputs/eval_val_ft.json --split val
```

### What is actually graded

Each image carries 3–5 questions across six types — `Yes_No`, `Is_there`,
`What`, `Where`, `How_many`, `Which`. **The metrics do not grade whether a
Yes/No question got "yes" or "no" right**, for two reasons: every question in
the release is about a finding that is genuinely present, so "yes" is always
correct and grading it measures nothing; and the reference answers do not
respect their own question type (`How many findings are there?` is answered
with `Yes, Pulmonary fibrosis <loc_…>`).

So evaluation ignores question type and asks two things per **image**:

| Table | Question | Metrics |
| --- | --- | --- |
| **4** | Which of the 14 pathologies did the model name, anywhere in its answers? | Accuracy, Precision, Recall, **F1** |
| **5** | Do its rectangles overlap real lesions? | Precision, Recall, F1, mIoU over true positives |
| **6** | On what share of images is at least one rectangle usable? | % at IoU≥0.5, % at IoU≥0.3, mean IoU |

### The metrics in one paragraph each

**Precision** — of the pathologies the model claimed, how many were real.
Low precision means it hallucinates findings. **Recall** — of the pathologies
really present, how many it caught. Low recall means it misses disease.
**F1** — the harmonic mean of the two, which punishes lopsidedness: a model
with precision 1.00 and recall 0.02 has F1 0.04, not 0.51. F1 is the paper's
headline number (0.558 → 0.624).

**Accuracy is reported but should not be trusted here.** Most of the 14
pathologies are absent from any given image, so a model that says nothing at
all still scores about 0.75 — accuracy rewards silence.

**IoU** (intersection over union) is overlap area ÷ combined area of two
rectangles: 0.0 is no overlap, 1.0 is pixel-perfect. A prediction counts as
correct at **IoU ≥ 0.3**, which the paper calls a clinically meaningful
minimum — enough to point a radiologist at the right region. Matching is
one-to-one, highest IoU first, so a single lucky rectangle cannot claim
several ground-truth boxes. **mIoU** averages only the successful matches, so
it answers a separate question: when the model does find a lesion, how tightly
does it box it?

### Check the scorer before trusting a model number

```bash
python -m vindr_cxr_vqa.evaluate --oracle --split val   # no GPU needed
```

This replays the answer key as if it were the model's output. It must give
VQA precision **1.000** and localization **100%** at IoU ≥ 0.5 — if it does
not, the grader is broken, not the model. (Recall is ~0.29 there, correctly:
the dataset only asks about ~4 of the ~8.3 lesions per image, which also caps
box recall at ~0.48 for any model.)

### If the box numbers are all zero

The scorer prints what share of predictions contained a `<loc_…>` span. At
**0.0%** the model emitted no rectangles at all, so Tables 5 and 6 are
*vacuous rather than poor* — a formatting failure, not a localization one. A
zero-shot model usually lands there; learning the `<loc_…>` convention is part
of what fine-tuning does.

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
