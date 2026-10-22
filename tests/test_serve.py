"""Tests for the UI backend.

The parts worth testing are the ones that run before any GPU does: finding the
adapter among the shapes a training run leaves behind, and the coordinate
bookkeeping that decides where a box lands on screen. Neither needs torch, so
these run anywhere.
"""

from __future__ import annotations

import json
import os
import zipfile

import pytest

from vindr_cxr_vqa.config import Config
from vindr_cxr_vqa.constants import LOC_NORM_RANGE
from vindr_cxr_vqa.serve import (describe_boxes, describe_gt, mock_answer,
                                 newest_checkpoint, pred_space, resolve_adapter,
                                 split_answer, to_rel)


# --------------------------------------------------------------------------
# adapter resolution
# --------------------------------------------------------------------------
def make_adapter(path, weights="adapter_model.safetensors"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as fh:
        json.dump({"r": 16, "lora_alpha": 32}, fh)
    with open(os.path.join(path, weights), "wb") as fh:
        fh.write(b"\0" * 16)
    return str(path)


def test_no_adapter_means_the_zero_shot_baseline():
    assert resolve_adapter(None) is None
    assert resolve_adapter("") is None


def test_a_finished_run_is_used_as_is(tmp_path):
    run = make_adapter(tmp_path / "medgemma-lora")
    assert resolve_adapter(run) == os.path.abspath(run)


def test_a_finished_run_wins_over_its_own_checkpoints(tmp_path):
    run = make_adapter(tmp_path / "medgemma-lora")
    make_adapter(os.path.join(run, "checkpoint-400"))
    assert resolve_adapter(run) == os.path.abspath(run)


def test_an_interrupted_run_falls_back_to_the_newest_checkpoint(tmp_path):
    run = tmp_path / "medgemma-lora"
    make_adapter(run / "checkpoint-100")
    make_adapter(run / "checkpoint-900")
    make_adapter(run / "checkpoint-300")
    # Numeric, not lexicographic: "900" must beat "100", and "300" must not
    # beat "900" the way string ordering would have it.
    assert newest_checkpoint(str(run)).endswith("checkpoint-900")
    assert resolve_adapter(str(run)).endswith("checkpoint-900")


def test_the_kaggle_zip_is_unpacked(tmp_path):
    """The training notebook archives the adapter; shutil.make_archive puts
    the files at the root of the zip."""
    run = make_adapter(tmp_path / "run")
    zip_path = tmp_path / "medgemma-lora.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            z.write(os.path.join(run, name), name)
    found = resolve_adapter(str(zip_path))
    assert os.path.isfile(os.path.join(found, "adapter_config.json"))


def test_a_zip_with_a_folder_inside_is_unpacked(tmp_path):
    run = make_adapter(tmp_path / "run")
    zip_path = tmp_path / "lora.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            z.write(os.path.join(run, name), f"medgemma-lora/{name}")
    found = resolve_adapter(str(zip_path))
    assert found.endswith("medgemma-lora")
    assert os.path.isfile(os.path.join(found, "adapter_config.json"))


def test_a_directory_that_is_not_an_adapter_says_so(tmp_path):
    empty = tmp_path / "outputs"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        resolve_adapter(str(empty))
    assert "adapter_config.json" in str(exc.value)


def test_a_missing_path_says_so(tmp_path):
    with pytest.raises(SystemExit):
        resolve_adapter(str(tmp_path / "nope"))


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------
def cfg_for(loc_format="norm1000"):
    return Config({
        "data": {"loc_format": loc_format, "box_source": "fixed"},
        "eval": {"iou_threshold": 0.3, "max_new_tokens": 256},
    })


def test_pred_space_follows_the_training_config():
    assert pred_space(cfg_for("norm1000")) == "norm1000"
    assert pred_space(cfg_for("pixel")) == "pixel"


def test_rel_is_a_fraction_of_the_image():
    assert to_rel((0, 0, 100, 50), 200, 100) == [0.0, 0.0, 0.5, 0.5]


def test_a_normalised_box_lands_where_the_ground_truth_is():
    """A <loc_...> covering the right half maps onto the right half, whatever
    the image's real pixel size is -- which is the whole job of the overlay."""
    hi = LOC_NORM_RANGE - 1
    text = f"<loc_{hi // 2}_0_{hi}_{hi}>"
    gt = [(1000.0, 0.0, 2000.0, 2000.0)]
    boxes = describe_boxes(text, "norm1000", 2000, 2000, gt, 0.3)
    assert len(boxes) == 1
    assert boxes[0]["rel"][0] == pytest.approx(0.5, abs=0.01)
    assert boxes[0]["best_iou"] == pytest.approx(1.0, abs=0.01)


def test_a_box_that_misses_scores_zero_and_matches_nothing():
    gt = [(0.0, 0.0, 100.0, 100.0)]
    boxes = describe_boxes("<loc_900_900_999_999>", "norm1000", 1000, 1000,
                           gt, 0.3)
    assert boxes[0]["best_iou"] == 0.0
    assert boxes[0]["matched_gt"] is None


def test_an_answer_with_no_box_yields_none():
    assert describe_boxes("There is no finding.", "norm1000", 10, 10, [], 0.3) == []


def test_an_image_with_no_ground_truth_has_nothing_to_score_against():
    assert describe_gt(None) == []
    boxes = describe_boxes("<loc_1_1_9_9>", "norm1000", 10, 10, [], 0.3)
    assert boxes[0]["best_iou"] is None


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def test_the_reasoning_tail_is_split_off():
    answer, reason = split_answer("A finding. Reasoning: because X.")
    assert answer == "A finding."
    assert reason == "because X."


def test_an_answer_without_reasoning_is_left_whole():
    answer, reason = split_answer("A finding.")
    assert (answer, reason) == ("A finding.", "")


# --------------------------------------------------------------------------
# mock mode
# --------------------------------------------------------------------------
def test_mock_without_a_record_still_emits_a_parsable_box():
    text = mock_answer(None, "Where is the Cardiomegaly?", cfg_for())
    boxes = describe_boxes(text, "norm1000", 1000, 1000, [], 0.3)
    assert len(boxes) == 1
    assert all(0 <= c < LOC_NORM_RANGE for c in boxes[0]["raw"])


def test_mock_from_a_record_lands_near_its_ground_truth():
    """The nudge is small on purpose: the overlay should show two distinct
    rectangles, not a near-miss that reads as a bug in the coordinates."""
    record = {
        "answer": "The Cardiomegaly is at <loc_100_100_400_400>.",
        "reason": "because",
        "bbox_fixed_norm": [100, 100, 400, 400],
        "bbox_norm": [100, 100, 400, 400],
    }
    text = mock_answer(record, "Where is the Cardiomegaly?", cfg_for())
    gt = [(100.0, 100.0, 400.0, 400.0)]
    boxes = describe_boxes(text, "norm1000", LOC_NORM_RANGE - 1,
                           LOC_NORM_RANGE - 1, gt, 0.3)
    assert boxes[0]["best_iou"] > 0.4
    assert "Reasoning:" in text


def test_a_kaggle_archive_leaves_its_checkpoints_in_the_zip(tmp_path):
    """1.1 GB of optimizer state is not something inference needs on disk."""
    run = make_adapter(tmp_path / "run")
    make_adapter(tmp_path / "run" / "checkpoint-600")
    with open(tmp_path / "run" / "checkpoint-600" / "optimizer.pt", "wb") as fh:
        fh.write(b"\0" * 4096)

    zip_path = tmp_path / "medgemma-lora.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(tmp_path / "run"):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, tmp_path / "run"))

    found = resolve_adapter(str(zip_path))
    assert os.path.isfile(os.path.join(found, "adapter_config.json"))
    assert not os.path.exists(os.path.join(found, "checkpoint-600")), (
        "the checkpoints were extracted anyway")


def test_an_archive_of_only_checkpoints_still_extracts_them(tmp_path):
    """Without a final adapter the checkpoints are all there is to load."""
    src = tmp_path / "run"
    make_adapter(src / "checkpoint-300")
    zip_path = tmp_path / "lora.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(src):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, src))

    found = resolve_adapter(str(zip_path))
    assert found.endswith("checkpoint-300")


# --------------------------------------------------------------------------
# dtype selection (the CPU path's two silent failures)
# --------------------------------------------------------------------------
def test_an_explicit_dtype_is_never_second_guessed(monkeypatch):
    torch = pytest.importorskip("torch")
    from vindr_cxr_vqa import modeling
    assert modeling.resolve_dtype("float32") is torch.float32
    assert modeling.resolve_dtype("bfloat16") is torch.bfloat16
    assert modeling.resolve_dtype(torch.float16) is torch.float16


def test_a_cpu_without_bf16_kernels_gets_fp32(monkeypatch):
    """bf16 there is not slower, it is ~184x slower, and it reads as a hang."""
    torch = pytest.importorskip("torch")
    from vindr_cxr_vqa import modeling
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(modeling, "cpu_has_bf16_kernels", lambda: False)
    assert modeling.resolve_dtype("auto") is torch.float32


def test_a_cpu_with_bf16_kernels_keeps_bf16(monkeypatch):
    """AVX512-BF16 or AMX: half the memory, and the kernels are real."""
    torch = pytest.importorskip("torch")
    from vindr_cxr_vqa import modeling
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(modeling, "cpu_has_bf16_kernels", lambda: True)
    assert modeling.resolve_dtype("auto") is torch.bfloat16


def test_fp16_is_never_chosen_off_a_gpu(monkeypatch):
    """torch has almost no fp16 CPU kernels; it dies on the first matmul."""
    torch = pytest.importorskip("torch")
    from vindr_cxr_vqa import modeling
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for has_bf16 in (True, False):
        monkeypatch.setattr(modeling, "cpu_has_bf16_kernels", lambda: has_bf16)
        assert modeling.resolve_dtype("auto") is not torch.float16


def test_the_bf16_probe_falls_back_when_the_private_op_is_gone(monkeypatch):
    """``torch.ops.mkldnn._is_mkldnn_bf16_supported`` is private and may not
    exist; the capability string answers the same question more coarsely."""
    torch = pytest.importorskip("torch")
    from vindr_cxr_vqa import modeling

    def gone():
        raise RuntimeError("no such operator")

    # torch.ops resolves attributes through __getattr__, so deleting the name
    # would not hide it -- the probe has to be made to raise instead.
    monkeypatch.setattr(torch.ops.mkldnn, "_is_mkldnn_bf16_supported", gone,
                        raising=False)
    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability",
                        lambda: "AVX512")
    assert modeling.cpu_has_bf16_kernels() is True

    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability",
                        lambda: "AVX2")
    assert modeling.cpu_has_bf16_kernels() is False


# --------------------------------------------------------------------------
# a checkpoint whose files were renamed on the way down
# --------------------------------------------------------------------------
def test_renamed_json_files_are_named_in_the_error(tmp_path):
    """Browsers save .json as .txt. The weights keep their name, so the
    directory looks almost right and the plain error points nowhere."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    for name in ("adapter_config.txt", "tokenizer.txt",
                 "tokenizer_config.txt", "chat_template.txt"):
        (d / name).write_text("{}")

    with pytest.raises(SystemExit) as exc:
        resolve_adapter(str(d))
    message = str(exc.value)
    assert "renamed" in message
    assert "adapter_config.txt" in message and "adapter_config.json" in message
    assert "chat_template.jinja" in message


def test_a_directory_with_no_weights_gets_no_rename_hint(tmp_path):
    """Random .txt files next to nothing are not a mangled checkpoint."""
    d = tmp_path / "notes"
    d.mkdir()
    (d / "adapter_config.txt").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        resolve_adapter(str(d))
    assert "renamed" not in str(exc.value)


def test_a_correctly_named_checkpoint_needs_no_hint(tmp_path):
    d = make_adapter(tmp_path / "ok")
    assert resolve_adapter(d) == os.path.abspath(d)
