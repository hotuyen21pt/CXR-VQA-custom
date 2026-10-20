"""Tests for the differentiable box loss. Skipped when torch is absent."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from vindr_cxr_vqa.losses import (SoftBoxGIoULoss, gather_digit_logits,  # noqa: E402
                                  giou_loss, multitask_loss)

#: Stand-in vocabulary where digit d has token id 100 + d.
DIGIT_IDS = [100 + d for d in range(10)]
VOCAB = 128


def _one_hot_logits(digits: list[int], strength: float = 30.0) -> torch.Tensor:
    """Logits that put almost all mass on the requested digit at each step."""
    out = torch.zeros(1, len(digits), VOCAB)
    for i, d in enumerate(digits):
        out[0, i, DIGIT_IDS[d]] = strength
    return out


# --------------------------------------------------------------------------
def test_giou_of_identical_boxes_is_zero_loss():
    b = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    assert giou_loss(b, b).item() == pytest.approx(0.0, abs=1e-5)


def test_giou_loss_grows_with_separation():
    t = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    near = giou_loss(torch.tensor([[0.1, 0.1, 0.3, 0.3]]), t)
    far = giou_loss(torch.tensor([[0.7, 0.7, 0.9, 0.9]]), t)
    assert far.item() > near.item()


def test_giou_stays_finite_for_disjoint_boxes():
    """Plain IoU is flat at zero here; GIoU must still carry signal."""
    loss = giou_loss(torch.tensor([[0.8, 0.8, 0.9, 0.9]]),
                     torch.tensor([[0.0, 0.0, 0.1, 0.1]]))
    assert torch.isfinite(loss) and 1.0 < loss.item() <= 2.0


def test_giou_handles_inverted_corners():
    loss = giou_loss(torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                     torch.tensor([[0.1, 0.1, 0.5, 0.5]]))
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_giou_handles_degenerate_zero_area_box():
    loss = giou_loss(torch.tensor([[0.3, 0.3, 0.3, 0.3]]),
                     torch.tensor([[0.1, 0.1, 0.5, 0.5]]))
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
def test_soft_decoder_recovers_the_encoded_coordinates():
    fn = SoftBoxGIoULoss(DIGIT_IDS, digits_per_coord=3)
    # <loc_123_456_789_012>
    logits = _one_hot_logits([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2])
    coords = fn.decode(logits)[0] * 999.0
    assert coords.tolist() == pytest.approx([123, 456, 789, 12], abs=0.2)


def test_soft_decoder_ignores_mass_outside_the_digit_tokens():
    fn = SoftBoxGIoULoss(DIGIT_IDS, digits_per_coord=3)
    logits = _one_hot_logits([0] * 12)
    logits[0, :, 7] = 50.0           # a non-digit token dominating the vocab
    assert fn.decode(logits)[0].tolist() == pytest.approx([0, 0, 0, 0], abs=1e-3)


def test_soft_decoder_is_differentiable():
    fn = SoftBoxGIoULoss(DIGIT_IDS, digits_per_coord=3)
    logits = _one_hot_logits([1, 0, 0, 1, 0, 0, 5, 0, 0, 5, 0, 0, ],
                             strength=4.0).requires_grad_(True)
    target = torch.tensor([[0.5, 0.5, 0.9, 0.9]])
    loss = fn(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0, "no gradient reached the logits"


def test_perfect_prediction_gives_near_zero_box_loss():
    fn = SoftBoxGIoULoss(DIGIT_IDS, digits_per_coord=3)
    logits = _one_hot_logits([1, 0, 0, 2, 0, 0, 5, 0, 0, 6, 0, 0])
    target = torch.tensor([[100 / 999, 200 / 999, 500 / 999, 600 / 999]])
    assert fn(logits, target).item() == pytest.approx(0.0, abs=1e-3)


def test_gradient_pushes_the_box_toward_the_target():
    fn = SoftBoxGIoULoss(DIGIT_IDS, digits_per_coord=3)
    logits = _one_hot_logits([1, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
                             strength=4.0).requires_grad_(True)
    target = torch.tensor([[0.6, 0.6, 0.8, 0.8]])
    before = fn(logits, target)
    before.backward()
    with torch.no_grad():
        stepped = logits - 50.0 * logits.grad
    assert fn(stepped, target).item() < before.item()


# --------------------------------------------------------------------------
def test_gather_digit_logits_shifts_by_one_for_causal_lm():
    logits = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    out = gather_digit_logits(logits, torch.tensor([[3, 5]]))
    # Position t predicts token t+1, so label index i reads logits at i-1.
    assert out.flatten().tolist() == [2.0, 4.0]


def test_gather_digit_logits_clamps_at_sequence_start():
    logits = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    assert gather_digit_logits(logits, torch.tensor([[0]])).item() == 0.0


def test_multitask_loss_applies_lambda():
    lm = torch.tensor(2.0)
    assert multitask_loss(lm, torch.tensor(1.0), 0.5).item() == pytest.approx(2.5)
    assert multitask_loss(lm, None, 0.5).item() == pytest.approx(2.0)


def test_soft_loss_rejects_a_bad_digit_table():
    with pytest.raises(ValueError):
        SoftBoxGIoULoss([1, 2, 3])


def test_resolve_dtype_honours_an_explicit_choice():
    from vindr_cxr_vqa import modeling
    assert modeling.resolve_dtype("float16") is torch.float16
    assert modeling.resolve_dtype("bfloat16") is torch.bfloat16
    assert modeling.resolve_dtype(torch.float32) is torch.float32


def test_resolve_dtype_auto_picks_a_half_precision_type():
    from vindr_cxr_vqa import modeling
    assert modeling.resolve_dtype("auto") in (torch.bfloat16, torch.float16)
