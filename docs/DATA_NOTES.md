# What the released data actually contains

Everything here is reproducible with:

```bash
python -m vindr_cxr_vqa.build_dataset     # writes data/processed/
python -m vindr_cxr_vqa.stats             # writes outputs/dataset_report.md
```

`outputs/dataset_report.md` is the machine-generated version of this page: it
recomputes every statistic from `data/processed/*.jsonl` and marks each one
`match` or `MISMATCH` against the paper. **25 of 30 checks match.** This page
explains the five that do not, and what the repo does about them.

---

## What matches

| Claim (paper / dataset card) | Rebuilt |
| --- | --- |
| 4,394 images, 17,597 QA pairs | exact |
| 4.0 QA pairs per image | exact |
| 14 pathology classes | exact |
| 6 question types, balanced 16.4–17.0% | exact (16.4–17.0%) |
| Per-class prevalence, Cardiomegaly 28.0% → Atelectasis 0.1% | all 14 within 0.1pp |
| Split 3,735 / 659 / 300 images | exact |
| ~8.3 ground-truth boxes per validation image | 8.44 |
| Every `image_id` resolves to a local VinDr-CXR PNG | 4,394 / 4,394 |

The 5,471 validation ground-truth boxes of Section 3.2 come out as 5,561 here.
The paper does not publish its split seed, so an exact match is not expected;
the 1.6% gap is split noise, and it confirms the boxes being counted are the
**raw** per-radiologist readings rather than fused consensus boxes.

---

## Discrepancy 1 — `gt_location` usually belongs to a different pathology

**The most consequential finding.** Section 2.2 states that cross-validating
all 4,394 images against VinDr-CXR "confirmed zero mismatches […] ensuring
100% preservation of expert annotations".

Rebuilding the join gives a different picture:

| | QA pairs | share |
| --- | --- | --- |
| `gt_location` is a radiologist box **of `gt_finding`** | 7,043 | **40.0%** |
| `gt_location` is a radiologist box **of another pathology** | 10,530 | **59.8%** |
| `gt_location` matches no radiologist box at all | 24 | 0.1% |

So the boxes themselves are genuine — 99.9% of them are real VinDr annotations
on the right image — but in three cases out of five the box is paired with the
wrong label. The most frequent confusions:

| `gt_finding` says | the box is actually | count |
| --- | --- | --- |
| Cardiomegaly | Aortic enlargement | 1,906 |
| Cardiomegaly | Pleural thickening | 398 |
| Other lesion | Aortic enlargement | 353 |
| Lung Opacity | Aortic enlargement | 352 |
| Pulmonary fibrosis | Pleural thickening | 351 |

The pattern is consistent with the generation pipeline having drawn the finding
list and the box list for an image independently instead of joining them.

A worked example — image `9a5094b2563a1ef3ff50dc5c7ff71345`, whose three QA
pairs all carry `<loc_691_1375_1653_1831>`:

```
R10  Cardiomegaly         691 1376 1654 1832   <- the published box, correct here
R8   Cardiomegaly         689 1313 1667 1763
R9   Cardiomegaly         693 1376 1657 1800
R9   Aortic enlargement  1052  715 1299  967
R9   Pleural effusion    1789 1729 1875 1992
```

and the failing case `051132a778e61a86eb147c7c6f564dfe`, where
`gt_finding = "Cardiomegaly"` but `gt_location = <loc_1264_743_1611_1019>` is
that image's **Aortic enlargement** box.

**Why the paper's own metrics cannot see this.** Section 3.2 scores a
prediction as a true positive when it overlaps *any* ground-truth box with
IoU ≥ 0.3 — the match is class-agnostic. A box on the aorta labelled
"Cardiomegaly" still overlaps a real lesion, so it scores as correct. The
mismatch costs nothing at evaluation time; it costs at **training** time, where
it teaches the model to point at the wrong organ for 60% of the supervision.

**What this repo does.** Every record carries both, so nothing is lost:

| field | meaning |
| --- | --- |
| `gt_location`, `bbox_orig` | the published box, byte-for-byte |
| `box_label_consistent` | whether the published box belongs to `gt_finding` |
| `gt_location_fixed`, `bbox_fixed_orig` | the consensus box that *does* belong to `gt_finding` |
| `repair_method` | `published` / `best_iou` / `consensus` / `unavailable` |

`data.box_source` in `configs/default.yaml` selects which one supervises
training. It defaults to `fixed`. Set it to `published` to reproduce the
paper's exact setup.

The repair fuses the two or three independent radiologist readings of each
lesion (greedy IoU clustering at 0.3, then averaging) and picks the consensus
box of the right class — the one overlapping the published box where there is
one, otherwise the most strongly agreed-on box of that class.

---

## Discrepancy 2 — the clinical reasoning is ~18 words, not 150–250

Table 2 and the dataset card both specify a "detailed clinical reasoning
paragraph (150–250 words)". The released `reason` field:

| | words |
| --- | --- |
| minimum | 4 |
| median | 20 |
| mean | 17.8 |
| maximum | 40 |
| **pairs in the claimed 150–250 range** | **0 of 17,597** |

A typical value is one sentence:

> "Cardiomegaly is a nonspecific finding often indicative of chronic cardiac
> stress, such as chronic hypertension, valvular disease, or cardiomyopathy."

and the shortest degenerate to `"Visible at specified location."`.

The field is still useful supervision for explainability, just far shorter than
documented. `data.include_reason` (default `true`) appends it to the training
target; the short length is why `data.max_seq_length` of 1024 is comfortable.

---

## Discrepancy 3 — there are no negative samples

The abstract reports "a balanced distribution of 41.7% positive and 58.3%
negative samples, mitigating hallucinations in normal cases".

In `data_v1.json`:

- every one of the 17,597 QA pairs carries a non-empty `gt_finding` **and** a
  non-empty `gt_location`;
- no pair has `gt_finding = "No finding"`;
- no answer begins with "No";
- only 32 answers contain a negation word anywhere.

All 4,394 images were selected as "containing at least one expert-annotated
pathology finding" (Section 2.1), so there is no normal-case subset to draw
negatives from. The released file is 100% positive.

**Consequence.** A model fine-tuned on this data never sees an example of
declining to report a finding, which is the behaviour the balance was meant to
instil. Expect the high recall / low precision profile the paper reports
(recall 0.926, precision 0.471 after fine-tuning). Building genuine negatives
would mean pulling in the ~10,600 VinDr train images with no finding, which is
out of scope here — `vindr.load_images` already exposes them if you want to.

---

## Discrepancy 4 — multi-lesion complexity

Table 3 reports "54.8% [of validation images] with ≥2 pathologies". Recomputed
over the release, **92.7%** of validation images carry boxes from two or more
distinct pathologies, and the figure is 93.0% across all 4,394 images.

The companion number in the same row — 8.3 boxes per validation image —
reproduces cleanly at 8.44, so the box counts are right and only the ≥2
percentage is off. No definition of "pathology" we tried recovers 54.8%.

This matters for how Section 4's conclusion should be read: the low box recall
(0.090) is attributed there to "the lack of multi-lesion training examples",
but training images are just as lesion-dense as validation images. The more
likely cause is that each image gets ~4 QA pairs and therefore at most ~4
supervised boxes, against ~8.4 ground-truth boxes at evaluation — a ceiling of
roughly 0.48 on recall before the model makes a single error.

---

## Discrepancy 5 — the training set is larger than the dataset

Section 3.1 states that training runs "for 3 epochs over the **20,880-sample**
training set".

The entire release contains 17,597 QA pairs. An 85% training split of that is
**14,960** — what this repo builds. 20,880 is not merely a different split
ratio; it exceeds the total number of samples that exist.

Three readings are possible and the paper does not distinguish them: the
training set was augmented (nothing in the text says so), a larger internal
version of the dataset was used and a subset released, or the number is simply
wrong. It matters because it sets the step budget: at an effective batch of 16,
3 epochs is 3,915 steps under the paper's figure but 2,805 under the released
data. `scripts/reproduce.py` and the Kaggle notebook use the released data, so
`--max-steps -1` means 2,805 steps.

---

## Coordinate spaces

`gt_location` is in **original DICOM pixel space**, while the PNGs in
`dataset/train_data/train/` are pre-resized to 224×224. Mixing the two silently
produces boxes in the wrong place, so every record carries all three spaces
explicitly:

| field | space |
| --- | --- |
| `bbox_orig` | original DICOM pixels (what `gt_location` encodes) |
| `bbox_resized` | 224×224, via `resize_factors.csv` |
| `bbox_norm` / `loc_norm` | normalised integers 0–999 |

Training uses the normalised space by default (`data.loc_format: norm1000`):
it is resolution-independent, and its fixed three-digit width is what lets the
soft-IoU loss locate the coordinate digit tokens. `evaluate.py` converts
predictions back to original pixels before computing any IoU.
