"""The multi-task objective of Section 3.1.

    L_total = L_VQA + lambda * L_bbox        (lambda = 0.5)

``L_VQA`` is ordinary cross-entropy over the answer tokens.  ``L_bbox`` is
described as "IoU-based regression" on coordinates that are "predicted via
special tokens (``<loc_x1_y1_x2_y2>``) embedded in the generated answer".

Making an IoU differentiable through tokens takes one step: at each coordinate
digit position the model emits a distribution over the ten digit tokens, so the
*expected* digit is a differentiable function of the logits, and three expected
digits assemble into an expected coordinate.  Four coordinates give a soft box,
and GIoU against the ground-truth box is then differentiable end to end --
without adding a regression head, exactly as the paper requires.

This needs every coordinate to occupy a fixed number of digits, which is why
``loc_format: norm1000`` (three digits, range 0-999) is the training default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SoftBoxGIoULoss(torch.nn.Module):
    """Differentiable GIoU over coordinates decoded from digit-token logits.

    Args:
        digit_token_ids: token id of each digit ``0``..``9``, in order.
        digits_per_coord: digits per coordinate (3 for the 0-999 space).
    """

    def __init__(self, digit_token_ids: list[int], digits_per_coord: int = 3):
        super().__init__()
        if len(digit_token_ids) != 10:
            raise ValueError("expected exactly 10 digit token ids")
        self.register_buffer(
            "digit_ids", torch.tensor(digit_token_ids, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "digit_values", torch.arange(10, dtype=torch.float32),
            persistent=False)
        self.digits_per_coord = digits_per_coord
        self.scale = float(10 ** digits_per_coord - 1)  # 999

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected coordinates from logits at the digit positions.

        Args:
            logits: ``(B, 4 * digits_per_coord, vocab)`` -- the model's output
                at the positions that generate the coordinate digits.
        Returns:
            ``(B, 4)`` soft coordinates scaled to ``[0, 1]``.
        """
        # Restrict to the ten digit tokens and renormalise: the model is free
        # to put mass elsewhere, but only the digits carry coordinate meaning.
        digit_logits = logits.index_select(-1, self.digit_ids)   # (B, D, 10)
        probs = F.softmax(digit_logits.float(), dim=-1)
        expected = (probs * self.digit_values).sum(-1)           # (B, D)

        b, d = expected.shape
        k = self.digits_per_coord
        place = torch.tensor([10.0 ** (k - 1 - i) for i in range(k)],
                             device=expected.device, dtype=expected.dtype)
        coords = (expected.view(b, 4, k) * place).sum(-1)        # (B, 4)
        return coords / self.scale

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """GIoU loss between the decoded soft box and ``target``.

        Args:
            logits: ``(B, 4*digits_per_coord, vocab)`` at the digit positions.
            target: ``(B, 4)`` ground-truth box, same ``[0, 1]`` scale.
        """
        # Trainer moves `model` onto the GPU, but this loss is handed to it
        # separately and is nobody's submodule, so its buffers would sit on
        # the CPU for ever and index_select would fail on the first step.
        # Follow the logits instead, once, rather than asking every caller to
        # remember to move us.
        if self.digit_ids.device != logits.device:
            self.to(logits.device)
        return giou_loss(self.decode(logits), target.to(torch.float32))


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean generalised-IoU loss, ``1 - GIoU``, over a batch of ``(B, 4)`` boxes.

    GIoU rather than plain IoU because it still produces a gradient when the
    predicted and ground-truth boxes do not overlap -- the common case early in
    training, where a plain IoU loss would be flat at zero.
    """
    # Order the corners so a box that comes out inverted still has an area.
    px1 = torch.minimum(pred[:, 0], pred[:, 2])
    py1 = torch.minimum(pred[:, 1], pred[:, 3])
    px2 = torch.maximum(pred[:, 0], pred[:, 2])
    py2 = torch.maximum(pred[:, 1], pred[:, 3])
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    tgt_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    ix1, iy1 = torch.maximum(px1, tx1), torch.maximum(py1, ty1)
    ix2, iy2 = torch.minimum(px2, tx2), torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    union = pred_area + tgt_area - inter
    iou = inter / union.clamp(min=1e-7)

    cx1, cy1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    cx2, cy2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
    enclosing = ((cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0))

    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-7)
    return (1.0 - giou).mean()


def gather_digit_logits(logits: torch.Tensor,
                        positions: torch.Tensor) -> torch.Tensor:
    """Pick out the logits that predict the coordinate digits.

    A causal LM's logits at position ``t`` predict the token at ``t + 1``, so
    the logits that generate the digit at label index ``i`` sit at ``i - 1``.

    Args:
        logits: ``(B, T, vocab)``.
        positions: ``(B, D)`` label indices of the digit tokens.
    Returns:
        ``(B, D, vocab)``.
    """
    idx = (positions - 1).clamp(min=0)
    return logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, logits.size(-1)))


def multitask_loss(lm_loss: torch.Tensor, bbox_loss: torch.Tensor | None,
                   lambda_bbox: float) -> torch.Tensor:
    """``L_total = L_VQA + lambda * L_bbox``, skipping the term when absent."""
    if bbox_loss is None:
        return lm_loss
    return lm_loss + lambda_bbox * bbox_loss
