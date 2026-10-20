# How evaluation works, from scratch

*(Tiếng Việt: [EVALUATION.vi.md](EVALUATION.vi.md))*

Written for someone who has never done computer vision. It explains what the
model is asked, what it answers, which numbers are computed from those
answers, and how to read them. Every example is a real record from this
dataset.

---

## 1. Five terms, once

| Term | Meaning here |
| --- | --- |
| **Chest X-ray** | One greyscale image. 4,394 of them. |
| **Finding / pathology** | An abnormality a radiologist saw, e.g. *Cardiomegaly* (enlarged heart). 14 possible kinds. |
| **Bounding box** | A rectangle around where the finding is, written `<loc_xmin_ymin_xmax_ymax>`. |
| **Ground truth** | What the radiologist actually drew and labelled. The answer key. |
| **Prediction** | What the model wrote. What we grade. |

The model does **two jobs at once**: say *what* it sees (text) and say *where*
it is (a rectangle). The three tables grade those separately.

---

## 2. What the model is asked

Each image gets 3–5 questions, drawn from six types. Real examples from
`data/processed/val.jsonl`:

| Type | Question | Reference answer |
| --- | --- | --- |
| `Yes_No` | Is there Pulmonary fibrosis? | Yes, Pulmonary fibrosis `<loc_1024_681_1362_969>` |
| `Is_there` | Does this X-ray show Lung Opacity? | Yes, Lung Opacity is listed as a primary finding… `<loc_1309_816_1658_1138>` |
| `What` | What abnormality is visible? | Other lesion `<loc_1787_215_1972_257>` |
| `Where` | Where is the Pulmonary fibrosis? | …localized to the area defined by `<loc_1024_681_1362_969>` |
| `How_many` | How many findings are there? | Pulmonary fibrosis `<loc_1024_681_1362_969>` |
| `Which` | Which side shows the abnormality? | …localized to the patient's left side `<loc_1309_816_1658_1138>` |

### The surprising part

**The metrics do not grade whether a Yes/No question got "yes" or "no" right.**

Two reasons. First, there are no negative samples — every question in the
release is about a finding that is genuinely present, so the correct answer to
every `Yes_No` and `Is_there` question is "yes". A model that always says
"yes" would score 100% on that, which measures nothing.

Second, the reference answers themselves do not respect their question type.
Look at the `How_many` row above: the question asks for a count and the answer
names one pathology instead. The `What` row begins with "Yes,". Grading
against those strings literally would measure the generation pipeline's
inconsistency, not the model.

So the evaluation ignores question type and asks two questions per **image**:

1. **Which pathologies did the model mention, anywhere in its answers?**
   → Table 4
2. **Where did it put its rectangles?** → Tables 5 and 6

---

## 3. The three steps

```
step 1  infer      model + images  ->  preds_val_base.jsonl
step 2  evaluate   predictions     ->  eval_val_base.json   (the three tables)
step 3  results    two eval files  ->  results_val.md       (side by side)
```

### Step 1 — generate predictions

```bash
python -m vindr_cxr_vqa.infer --split val --out outputs/preds_val_base.jsonl
```

Reads each image, asks each question, writes what the model said:

```json
{"sample_id": "0005e8e3..._0", "image_id": "0005e8e3...",
 "question": "Where is the Lung Opacity?",
 "prediction": "The lung opacity is in the right upper zone <loc_706_332_816_411>."}
```

Run it twice: once with no adapter (the **Pretrained** row, the model as
downloaded) and once with `--adapter` (the **Fine-tuned** row).

### Step 2 — score them

```bash
python -m vindr_cxr_vqa.evaluate --pred outputs/preds_val_base.jsonl --split val
```

### Step 3 — put the two side by side

```bash
python -m vindr_cxr_vqa.results --base outputs/eval_val_base.json \
    --ft outputs/eval_val_ft.json --split val
```

---

## 4. Table 4 — did it name the right pathologies?

### The question being asked

For one image, compare two sets of pathology names:

- **Truth**: what radiologists boxed on that image.
- **Prediction**: which pathology names appear in the model's answers.

Real image `9a5094b2563a1ef3ff50dc5c7ff71345` has four: *Aortic enlargement,
Cardiomegaly, Pleural effusion, Pleural thickening*.

Suppose the model's answers mention *Cardiomegaly, Pleural effusion,
Pneumothorax*. Lay the 14 possible pathologies side by side:

| Pathology | In truth? | Model said? | Verdict |
| --- | --- | --- | --- |
| Cardiomegaly | yes | yes | **TP** — true positive, correctly found |
| Pleural effusion | yes | yes | **TP** |
| Aortic enlargement | yes | no | **FN** — false negative, missed it |
| Pleural thickening | yes | no | **FN** |
| Pneumothorax | no | yes | **FP** — false positive, invented it |
| the other 9 | no | no | **TN** — true negative, correctly silent |

Every image contributes 14 such verdicts. They are summed over all 659
validation images, then:

### The three numbers

**Precision — when it speaks up, how often is it right?**

```
            TP              2
P  =  ───────────  =  ─────────  =  0.67
         TP + FP        2 + 1
```
Out of 3 pathologies claimed, 2 were real. Low precision = the model
hallucinates findings.

**Recall — of what was really there, how much did it catch?**

```
            TP              2
R  =  ───────────  =  ─────────  =  0.50
         TP + FN        2 + 2
```
Out of 4 real pathologies, it found 2. Low recall = the model misses disease.

**F1 — one number balancing both.**

```
          2 · P · R        2 · 0.67 · 0.50
F1  =  ───────────────  =  ───────────────  =  0.57
           P + R             0.67 + 0.50
```

F1 is the *harmonic* mean, not the average. That matters: it punishes
lopsidedness. A model with P=1.00 and R=0.02 averages 0.51 but has F1 = 0.04,
which is the honest verdict — a model that finds 2% of disease is useless
however careful it is when it does speak.

**This is the paper's headline number**: F1 0.558 → 0.624 after fine-tuning.

### Why Accuracy is listed but not trusted

```
             TP + TN            2 + 9
Acc  =  ─────────────────  =  ─────────  =  0.79
         TP + TN + FP + FN       14
```

79% sounds good, but it is inflated by the 9 **TN** — pathologies that were
absent and went unmentioned. A validation image carries only **3.59** of the
14 on average, so **a model that says nothing at all still scores 0.743**.

Accuracy rewards silence; precision and recall do not. That is why F1 is the
number to read.

> **A note on the paper's Table 4.** It prints the same value in its Accuracy
> and F1 columns (0.558 twice, then 0.624 twice). Those are different
> quantities and are not equal in general — and the printed F1 is exactly
> `2PR/(P+R)` of the printed precision and recall. The Accuracy column looks
> like a duplicate of F1. This repo reports both: `f1` is the comparable
> number, `accuracy` is the genuine `(TP+TN)/total`.

### How "did it mention Cardiomegaly?" is decided

Plain text search over the answer, with a negation guard
(`evaluate.py:mentioned`). Each pathology has surface forms — *Cardiomegaly*
also matches "enlarged heart", "cardiac enlargement". Then:

```
"There is cardiomegaly."              ->  counts as mentioned
"There is no pleural effusion."       ->  does NOT count
"No pneumothorax. Cardiomegaly is present."
                                      ->  Cardiomegaly counts, Pneumothorax does not
```

The negation only reaches back to the nearest clause boundary, so the third
example is handled correctly.

---

## 5. Table 5 — did it put the box in the right place?

### IoU, the one idea you need

**Intersection over Union** measures rectangle overlap. Overlap area divided
by combined area:

```
   ┌─────────────┐                     overlap (∩)
   │ prediction  │                 IoU = ──────────────
   │      ┌──────┼───────┐               combined (∪)
   │      │ ████ │       │
   └──────┼──────┘       │          0.0 = no overlap at all
          │  truth       │          0.3 = loose but clinically usable
          └──────────────┘          0.5 = good
                                    1.0 = pixel-perfect
```

Concretely, a 10×10 prediction and a 10×10 truth offset by 5 pixels:

```
overlap  = 5 × 10          = 50
combined = 100 + 100 - 50  = 150
IoU      = 50 / 150        = 0.33
```

### Grading the rectangles

Each image has ground-truth boxes (8.3 on average — three radiologists each
drew their own) and however many the model emitted.

1. Compute IoU for every prediction × truth pair.
2. Match them greedily: highest IoU first, each box used once.
3. A match with **IoU ≥ 0.3** is a **TP**.
4. Leftover predictions are **FP** (pointed somewhere wrong).
5. Leftover truth boxes are **FN** (lesions missed entirely).

Then the same precision / recall / F1 formulas as Table 4.

**Why one-to-one matching?** Without it, one lucky rectangle could claim three
overlapping ground-truth boxes, and the FN count would be meaningless. See
`boxes.py:greedy_match`.

**Why 0.3 and not 0.5?** The paper calls 0.3 "a clinically meaningful minimum
overlap" — enough to point a radiologist at the right region, which is the
clinical purpose, even if the rectangle is sloppy.

### mIoU over true positives only

The average IoU of the matches that succeeded. It answers a different
question: *when the model does find a lesion, how tightly does it box it?*

This is why the paper can report recall 0.090 but mIoU 0.615 — the model finds
few lesions, but the ones it finds it boxes well. Two separate failures,
separately visible.

---

## 6. Table 6 — is the output usable per image?

Tables 4 and 5 pool everything into one number. Table 6 asks a clinician's
question instead: **on what fraction of X-rays did the model produce at least
one rectangle a radiologist could use?**

| Metric | Meaning |
| --- | --- |
| `IoU≥0.5 (%)` | share of images with at least one "good" box |
| `IoU≥0.3 (%)` | share with at least one "acceptable" box |
| `Mean IoU` | average over *all* predicted boxes, including the bad ones |

The paper's fine-tuned model: 22.8% good, 48.6% acceptable. So on roughly half
of X-rays it points somewhere useful, and on a fifth it is genuinely accurate.

Note `Mean IoU` here (0.133) is far below `mIoU(TP)` in Table 5 (0.615),
because this one includes every miss, and that one only averages the hits.

---

## 7. Reading the output

```
=== preds_val_ft.jsonl  (split=val, space=norm1000, match=greedy) ===

Table 4 — VQA (text-based lesion classification)
  Accuracy 0.624   Precision 0.471   Recall 0.926   F1 0.624
                                     ↑ finds almost everything...
                                                  ↑ ...but half its claims are wrong

Table 5 — Bounding-box detection (IoU>=0.3, gt=raw)
  Precision 0.120   Recall 0.090   F1 0.103   mIoU(TP) 0.615
                                              ↑ when right, boxes are tight
  4171 predicted vs 5561 ground-truth boxes

Table 6 — Localization quality (per image, n=659)
  iou_ge_0.5_pct         22.8     ← a fifth of images get a good box
  iou_ge_0.3_pct         48.6     ← about half get a usable one
  mean IoU (all preds)   0.133

Output format: 2140/2637 predictions contain a <loc_...> span (81.2%)
```

That last line matters most when things go wrong. If it reads **0.0%**, the
model emitted no rectangles at all and Tables 5 and 6 are *vacuous*, not poor
— they are measuring a formatting failure, not localization ability. The
scorer prints a warning and a sample prediction in that case.

A zero-shot model usually scores near 0% there: it has never seen the
`<loc_…>` convention. Learning the format is part of what fine-tuning does.

---

## 8. Checking the scorer itself

Before trusting any model number, confirm the grader works. Replay the answer
key as if it were the model's output:

```bash
python -m vindr_cxr_vqa.evaluate --oracle --split val
```

Expected, and what this repo actually prints:

```
Table 4   Precision 1.000      every pathology named is genuinely present
Table 6   iou_ge_0.5_pct 100.0 every box lands on a real lesion
```

If precision is not 1.000, the text matching is broken. If localization is not
100%, the coordinate conversion is. **Needs no GPU** — run it on any laptop.

Recall is *not* 1.000 there (about 0.29), and that is correct: the dataset only
asks about ~4 of the ~8.4 lesions per image, so even a perfect answer leaves
most ground-truth boxes unmatched.

That is also a ceiling on the real model: 2,637 questions against 5,561
ground-truth boxes means **box recall cannot exceed 0.474** under this
supervision. So the paper's 0.090 is about **a fifth** of what is reachable,
not a twentieth.

---

## 9. Four ways to misread these numbers

1. **High accuracy means a good model.** No — see §4. Read F1.
2. **Box metrics are class-aware.** They are not. A rectangle on the aorta
   labelled "Cardiomegaly" still counts as a true positive if it overlaps any
   ground-truth box. This is also why the label/box mismatch described in
   [DATA_NOTES.md](DATA_NOTES.md) costs nothing at evaluation time but
   corrupts training.
3. **Validation and test are independent.** They are not — test is a
   300-image subset of the 659 validation images, because Table 3's own
   numbers only add up that way. The validation figures are the ones
   comparable to the paper.
4. **Low box recall means the model is bad at localizing.** Partly. But §8
   shows 0.474 is the structural ceiling, so 0.090 is roughly a fifth of what
   is reachable.

---

## 10. Where each number is computed

| What | File |
| --- | --- |
| Does an answer mention pathology X | `evaluate.py:mentioned` |
| Negation handling | `evaluate.py:_negated` |
| Table 4 | `evaluate.py:eval_vqa` |
| IoU | `boxes.py:iou` |
| One-to-one matching | `boxes.py:greedy_match` |
| Tables 5 and 6 | `evaluate.py:eval_boxes` |
| Format diagnostic | `evaluate.py:diagnose_predictions` |
| Scorer self-test | `evaluate.py:oracle_predictions` |
| Side-by-side tables | `results.py:render` |
