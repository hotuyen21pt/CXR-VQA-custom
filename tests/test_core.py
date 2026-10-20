"""Tests for the parts that run without a GPU."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vindr_cxr_vqa import boxes, vindr  # noqa: E402
from vindr_cxr_vqa.build_dataset import make_splits, repair_box  # noqa: E402
from vindr_cxr_vqa.config import Config, _mini_yaml  # noqa: E402
from vindr_cxr_vqa.evaluate import mentioned, prf  # noqa: E402
from vindr_cxr_vqa.prompts import build_target  # noqa: E402


# --------------------------------------------------------------------------
# <loc_...> parsing and coordinate spaces
# --------------------------------------------------------------------------
def test_parse_and_format_round_trip():
    assert boxes.parse_loc("<loc_691_1375_1653_1831>") == (691, 1375, 1653, 1831)
    assert boxes.format_loc((691, 1375, 1653, 1831)) == "<loc_691_1375_1653_1831>"
    assert boxes.format_loc((1, 2, 3, 4), width=3) == "<loc_001_002_003_004>"


def test_parse_rejects_malformed():
    assert boxes.parse_loc("<loc_1_2_3>") is None
    assert boxes.parse_loc("not a loc") is None


def test_extract_locs_finds_every_span():
    text = "A <loc_1_2_3_4> and B <loc_5_6_7_8>."
    assert boxes.extract_locs(text) == [(1, 2, 3, 4), (5, 6, 7, 8)]


def test_normalized_round_trip_is_tight():
    box, w, h = (691, 1375, 1653, 1831), 2084, 2048
    back = boxes.from_normalized(boxes.to_normalized(box, w, h), w, h)
    # One normalised step is ~2 px at this resolution.
    assert max(abs(a - b) for a, b in zip(box, back)) <= 3


def test_normalized_is_clamped_to_range():
    out = boxes.to_normalized((-10, -10, 5000, 5000), 2000, 2000)
    assert all(0 <= v <= 999 for v in out)


def test_to_resized_uses_both_factors():
    assert boxes.to_resized((100, 200, 300, 400), 0.1, 0.5) == (10, 100, 30, 200)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_iou_identity_and_disjoint():
    b = (0, 0, 10, 10)
    assert boxes.iou(b, b) == pytest.approx(1.0)
    assert boxes.iou(b, (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    assert boxes.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_greedy_match_is_one_to_one():
    # Two predictions both overlap the same gt box; only one may claim it.
    preds = [(0, 0, 10, 10), (1, 1, 11, 11)]
    gts = [(0, 0, 10, 10)]
    matches, unmatched_p, unmatched_g = boxes.greedy_match(preds, gts, 0.3)
    assert len(matches) == 1
    assert unmatched_p == [1]
    assert unmatched_g == []


def test_match_any_lets_both_predictions_count():
    preds = [(0, 0, 10, 10), (1, 1, 11, 11)]
    gts = [(0, 0, 10, 10)]
    matches, unmatched_p, _ = boxes.match_any(preds, gts, 0.3)
    assert len(matches) == 2 and unmatched_p == []


def test_greedy_match_respects_threshold():
    matches, unmatched_p, unmatched_g = boxes.greedy_match(
        [(0, 0, 10, 10)], [(9, 9, 19, 19)], 0.3)
    assert matches == [] and unmatched_p == [0] and unmatched_g == [0]


def test_union_and_mean_box():
    bs = [(0, 0, 10, 10), (10, 10, 20, 20)]
    assert boxes.union_box(bs) == (0, 0, 20, 20)
    assert boxes.mean_box(bs) == (5, 5, 15, 15)
    assert boxes.union_box([]) is None


# --------------------------------------------------------------------------
# consensus fusion
# --------------------------------------------------------------------------
def _rb(cls, rad, box):
    return vindr.RadBox("img", cls, rad, box)


def test_fuse_collapses_three_readings_of_one_lesion():
    fused = vindr.fuse_boxes([
        _rb("Cardiomegaly", "R1", (100, 100, 200, 200)),
        _rb("Cardiomegaly", "R2", (102, 98, 198, 202)),
        _rb("Cardiomegaly", "R3", (100, 100, 200, 200)),
    ])
    assert len(fused) == 1
    assert fused[0]["votes"] == 3
    assert fused[0]["bbox"] == pytest.approx((100.67, 99.33, 199.33, 200.67),
                                             abs=0.01)


def test_fuse_keeps_separate_lesions_apart():
    fused = vindr.fuse_boxes([
        _rb("Nodule/Mass", "R1", (0, 0, 10, 10)),
        _rb("Nodule/Mass", "R2", (500, 500, 510, 510)),
    ])
    assert len(fused) == 2


def test_fuse_never_merges_across_classes():
    fused = vindr.fuse_boxes([
        _rb("Cardiomegaly", "R1", (0, 0, 100, 100)),
        _rb("Aortic enlargement", "R2", (0, 0, 100, 100)),
    ])
    assert len(fused) == 2
    assert {f["class_name"] for f in fused} == {"Cardiomegaly",
                                                "Aortic enlargement"}


def test_majority_labels_needs_more_than_half():
    assert vindr.majority_labels([["A"], ["A"], ["B"]]) == ["A"]
    assert vindr.majority_labels([["A"], ["B"], ["C"]]) == []
    assert vindr.union_labels([["A"], ["B"]]) == ["A", "B"]


# --------------------------------------------------------------------------
# box repair
# --------------------------------------------------------------------------
def test_repair_keeps_a_box_that_is_already_correct():
    raw = [_rb("Cardiomegaly", "R1", (691, 1376, 1654, 1832))]
    fused = vindr.fuse_boxes(raw)
    box, method, ok = repair_box((691, 1375, 1653, 1831), "Cardiomegaly",
                                 raw, fused)
    assert ok and method == "published"
    assert box == (691, 1375, 1653, 1831)


def test_repair_swaps_in_the_right_class():
    raw = [_rb("Cardiomegaly", "R1", (10, 10, 100, 100)),
           _rb("Aortic enlargement", "R1", (500, 500, 600, 600))]
    fused = vindr.fuse_boxes(raw)
    # Published box is the aorta but the label says Cardiomegaly.
    box, method, ok = repair_box((500, 500, 600, 600), "Cardiomegaly",
                                 raw, fused)
    assert not ok and method == "consensus"
    assert box == pytest.approx((10, 10, 100, 100))


def test_repair_reports_unavailable_when_class_absent():
    raw = [_rb("Cardiomegaly", "R1", (10, 10, 100, 100))]
    box, method, ok = repair_box((10, 10, 100, 100), "Pneumothorax",
                                 raw, vindr.fuse_boxes(raw))
    assert method == "unavailable" and not ok
    assert box == (10, 10, 100, 100)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
class _FakeMeta:
    def __init__(self, cls):
        self.boxes = [_rb(cls, "R1", (0, 0, 1, 1))]


def test_splits_hit_the_paper_proportions_and_are_deterministic():
    metas = {f"img{i}": _FakeMeta("Cardiomegaly" if i % 3 else "Atelectasis")
             for i in range(4394)}
    ids = sorted(metas)
    a, ta = make_splits(metas, ids, 0.15, 300, 42, False)
    b, tb = make_splits(metas, ids, 0.15, 300, 42, False)
    assert a == b and ta == tb

    assert sum(1 for v in a.values() if v == "train") == 3735
    assert sum(1 for v in a.values() if v == "val") == 659
    assert len(ta) == 300
    # Test is drawn from validation unless disjoint_test is set.
    assert all(a[i] == "val" for i in ta)


def test_disjoint_test_moves_images_out_of_validation():
    metas = {f"img{i}": _FakeMeta("Cardiomegaly") for i in range(1000)}
    split, test_ids = make_splits(metas, sorted(metas), 0.15, 50, 7, True)
    assert all(split[i] == "test" for i in test_ids)
    assert sum(1 for v in split.values() if v == "val") == 150 - 50


def test_stratification_keeps_rare_classes_in_validation():
    metas = {f"c{i}": _FakeMeta("Cardiomegaly") for i in range(990)}
    metas |= {f"a{i}": _FakeMeta("Atelectasis") for i in range(10)}
    split, _ = make_splits(metas, sorted(metas), 0.15, 50, 42, False)
    rare_val = [i for i in metas if i.startswith("a") and split[i] == "val"]
    assert rare_val, "the rarest class must reach the validation split"


# --------------------------------------------------------------------------
# answer parsing
# --------------------------------------------------------------------------
def test_mentioned_finds_classes_and_synonyms():
    assert "Cardiomegaly" in mentioned("There is cardiomegaly.")
    assert "Cardiomegaly" in mentioned("The heart is enlarged: enlarged heart.")
    assert "Pleural effusion" in mentioned("A small effusion is seen.")


def test_mentioned_respects_negation():
    assert mentioned("There is no pleural effusion.") == set()
    assert mentioned("No evidence of pneumothorax.") == set()
    assert mentioned("Lungs are free of consolidation.") == set()


def test_negation_scope_stops_at_clause_boundary():
    out = mentioned("There is no pneumothorax. Cardiomegaly is present.")
    assert "Cardiomegaly" in out and "Pneumothorax" not in out


def test_prf_is_zero_without_predictions():
    assert prf(0, 0, 5) == (0.0, 0.0, 0.0)
    assert prf(5, 5, 0) == (0.5, 1.0, pytest.approx(2 / 3))


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def _record():
    return {
        "question": "Where is the Cardiomegaly?",
        "answer": "The Cardiomegaly is central <loc_691_1375_1653_1831>.",
        "reason": "Enlarged cardiac silhouette.",
        "gt_finding": "Cardiomegaly",
        "bbox_norm": [330, 442, 790, 589],
        "bbox_fixed_norm": [331, 443, 791, 590],
        "bbox_orig": [691, 1375, 1653, 1831],
        "bbox_fixed_orig": [692, 1376, 1654, 1832],
    }


def test_target_uses_the_configured_box_and_space():
    text, (lo, hi) = build_target(_record(), "fixed", "norm1000", False)
    assert text[lo:hi] == "<loc_331_443_791_590>"
    assert "1375" not in text, "the published pixel box must not leak through"


def test_target_span_offsets_are_exact_with_reason_appended():
    text, (lo, hi) = build_target(_record(), "published", "norm1000", True)
    assert text[lo:hi] == "<loc_330_442_790_589>"
    assert text.endswith("Enlarged cardiac silhouette.")


def test_target_pixel_format_matches_the_release():
    text, (lo, hi) = build_target(_record(), "published", "pixel", False)
    assert text[lo:hi] == "<loc_691_1375_1653_1831>"


def test_target_contains_exactly_one_loc_span():
    text, _ = build_target(_record(), "fixed", "norm1000", True)
    assert len(boxes.extract_locs(text)) == 1


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_mini_yaml_parses_nesting_and_scalars():
    cfg = Config(_mini_yaml("""
paths:
  dataset_dir: dataset   # trailing comment
  depth:
    inner: 3
train:
  lr: 2.0e-5
  epochs: 3
  flag: true
  modules: [a, b, c]
"""))
    assert cfg.paths.dataset_dir == "dataset"
    assert cfg.paths.depth.inner == 3
    assert cfg.train.lr == pytest.approx(2e-5)
    assert cfg.train.flag is True
    assert list(cfg.train.modules) == ["a", "b", "c"]


def test_config_raises_a_helpful_error_for_unknown_keys():
    with pytest.raises(AttributeError, match="available"):
        Config({"a": 1}).nope


def test_shipped_config_loads_and_matches_the_paper():
    from vindr_cxr_vqa.config import load_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "default.yaml"))
    assert cfg.lora.r == 16 and cfg.lora.alpha == 32
    assert cfg.lora.dropout == pytest.approx(0.1)
    assert float(cfg.train.lr) == pytest.approx(2e-5)
    assert cfg.train.batch_size * cfg.train.grad_accum == 16
    assert cfg.train.epochs == 3
    assert cfg.loss.lambda_bbox == pytest.approx(0.5)
    assert cfg.eval.iou_threshold == pytest.approx(0.3)


# --------------------------------------------------------------------------
# the built dataset, when it is present
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
needs_build = pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED, "val.jsonl")),
    reason="run python -m vindr_cxr_vqa.build_dataset first")


@needs_build
def test_built_splits_match_table_3():
    counts = {}
    for name in ("train", "val", "test"):
        with open(os.path.join(PROCESSED, f"{name}.jsonl"), encoding="utf-8") as fh:
            counts[name] = len({json.loads(l)["image_id"] for l in fh if l.strip()})
    assert counts == {"train": 3735, "val": 659, "test": 300}


@needs_build
def test_every_record_has_both_boxes_and_a_resolvable_image():
    from vindr_cxr_vqa.config import load_config
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    with open(os.path.join(PROCESSED, "val.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    assert rows
    for r in rows[:200]:
        assert boxes.parse_loc(r["gt_location"]) is not None
        assert boxes.parse_loc(r["gt_location_fixed"]) is not None
        assert len(r["bbox_norm"]) == 4 and all(0 <= v <= 999 for v in r["bbox_norm"])
        assert os.path.exists(os.path.join(cfg.paths.dataset_dir, r["image_path"]))


@needs_build
def test_repaired_boxes_belong_to_their_finding():
    """Every repaired box must sit on a radiologist box of the right class."""
    with open(os.path.join(PROCESSED, "images.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    with open(os.path.join(PROCESSED, "val.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]

    checked = bad = 0
    for r in rows[:500]:
        if r["repair_method"] == "unavailable":
            continue
        same_class = [b["bbox_orig"] for b in index[r["image_id"]]["boxes_raw"]
                      if b["class_name"] == r["gt_finding"]]
        if not same_class:
            continue
        checked += 1
        if max(boxes.iou(r["bbox_fixed_orig"], g) for g in same_class) < 0.3:
            bad += 1
    assert checked > 0
    assert bad == 0, f"{bad}/{checked} repaired boxes miss their own class"
