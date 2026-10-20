"""Bounding-box utilities: <loc_...> token (de)serialisation, coordinate
spaces and IoU-family metrics.

The paper encodes a box inside the generated answer as a single textual span

    <loc_xmin_ymin_xmax_ymax>

with coordinates in the ORIGINAL DICOM pixel space of the VinDr-CXR image
(Section 2.1 / Table 2).  Training, however, is easier when every coordinate
has a fixed width, so this module also supports a normalized 0-999 space
(``LOC_NORM_RANGE``) that yields constant-length ``<loc_123_456_789_012>``
spans.  Which space is written into the supervision target is a config switch
(``loc_format`` in configs/default.yaml), never a hard-coded choice.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .constants import LOC_NORM_RANGE

Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)

#: Matches one ``<loc_a_b_c_d>`` span. Tolerates negative values so that
#: malformed model output is surfaced rather than silently skipped.
LOC_RE = re.compile(r"<loc_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)>")


# --------------------------------------------------------------------------
# (de)serialisation
# --------------------------------------------------------------------------
def parse_loc(token: str) -> Box | None:
    """Parse a single ``<loc_x1_y1_x2_y2>`` span. ``None`` if it is malformed."""
    m = LOC_RE.fullmatch(token.strip())
    return tuple(float(g) for g in m.groups()) if m else None  # type: ignore[return-value]


def extract_locs(text: str) -> list[Box]:
    """Every ``<loc_...>`` span occurring in ``text``, in order of appearance."""
    return [tuple(float(g) for g in m.groups()) for m in LOC_RE.finditer(text)]  # type: ignore[misc]


def format_loc(box: Sequence[float], width: int = 0) -> str:
    """Render a box as a ``<loc_...>`` span.

    ``width`` left-pads each coordinate with zeros, which is what makes the
    normalized space fixed-length (``width=3`` for the 0-999 range).
    """
    xs = (f"{int(round(v)):0{width}d}" for v in box)
    return "<loc_" + "_".join(xs) + ">"


def loc_token_width(norm_range: int = LOC_NORM_RANGE) -> int:
    """Digits needed per coordinate in the normalized space (3 for 0-999)."""
    return len(str(norm_range - 1))


# --------------------------------------------------------------------------
# coordinate spaces
# --------------------------------------------------------------------------
def to_resized(box: Sequence[float], fx: float, fy: float) -> Box:
    """Original pixel space -> resized-image pixel space (e.g. 224x224)."""
    return (box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy)


def to_normalized(
    box: Sequence[float], width: int, height: int, norm_range: int = LOC_NORM_RANGE
) -> Box:
    """Original pixel space -> integer [0, norm_range-1] space.

    Clamped, because a handful of VinDr boxes extend a pixel or two past the
    image border.
    """
    hi = norm_range - 1
    sx, sy = hi / max(width, 1), hi / max(height, 1)
    out = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
    return tuple(min(hi, max(0.0, round(v))) for v in out)  # type: ignore[return-value]


def from_normalized(
    box: Sequence[float], width: int, height: int, norm_range: int = LOC_NORM_RANGE
) -> Box:
    """Inverse of :func:`to_normalized` (back to original pixel space)."""
    hi = norm_range - 1
    sx, sy = width / hi, height / hi
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two axis-aligned boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(preds: Sequence[Sequence[float]],
               gts: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[iou(p, g) for g in gts] for p in preds]


def greedy_match(
    preds: Sequence[Sequence[float]],
    gts: Sequence[Sequence[float]],
    thr: float = 0.3,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """One-to-one greedy matching by descending IoU.

    Section 3.2 defines a prediction as a true positive when it overlaps a
    ground-truth box with IoU >= ``thr``, counts the remaining predictions as
    false positives and the unmatched ground-truth boxes as false negatives.
    Matching one-to-one is what makes those two statements consistent: without
    it, several predictions could claim the same ground-truth box and the FN
    count would no longer be well defined.

    Returns ``(matches, unmatched_pred_idx, unmatched_gt_idx)`` where each
    match is ``(pred_idx, gt_idx, iou)``.
    """
    pairs = sorted(
        ((i, j, v)
         for i, row in enumerate(iou_matrix(preds, gts))
         for j, v in enumerate(row) if v >= thr),
        key=lambda t: -t[2],
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for i, j, v in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, v))
    return (
        matches,
        [i for i in range(len(preds)) if i not in used_p],
        [j for j in range(len(gts)) if j not in used_g],
    )


def match_any(
    preds: Sequence[Sequence[float]],
    gts: Sequence[Sequence[float]],
    thr: float = 0.3,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Looser variant: a prediction is a TP if it overlaps ANY gt box.

    This is the literal reading of the first sentence of Section 3.2 and is
    selectable via ``--match any`` so the two protocols can be compared.
    """
    matches: list[tuple[int, int, float]] = []
    unmatched_p: list[int] = []
    hit_g: set[int] = set()
    for i, row in enumerate(iou_matrix(preds, gts)):
        if not row:
            unmatched_p.append(i)
            continue
        j = max(range(len(row)), key=lambda k: row[k])
        if row[j] >= thr:
            matches.append((i, j, row[j]))
            hit_g.add(j)
        else:
            unmatched_p.append(i)
    return matches, unmatched_p, [j for j in range(len(gts)) if j not in hit_g]


def union_box(boxes: Iterable[Sequence[float]]) -> Box | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def mean_box(boxes: Iterable[Sequence[float]]) -> Box | None:
    boxes = list(boxes)
    if not boxes:
        return None
    n = len(boxes)
    return tuple(sum(b[k] for b in boxes) / n for k in range(4))  # type: ignore[return-value]
