"""Prompt and supervision-target construction.

The paper embeds the predicted box inside the generated answer as a
``<loc_x1_y1_x2_y2>`` span, "enabling unified end-to-end training without
architectural modifications" (Section 3.1).  The target string is therefore
just text, and this module is the single place that decides what that text
looks like.
"""

from __future__ import annotations

from .boxes import format_loc, loc_token_width
from .constants import LOC_NORM_RANGE

SYSTEM_PROMPT = (
    "You are an expert radiologist analysing a chest X-ray. "
    "Answer the question, cite the location of each finding you mention as a "
    "<loc_xmin_ymin_xmax_ymax> span, and then explain your reasoning."
)

#: Appended to the question so the model knows which coordinate space to emit.
LOC_INSTRUCTION = {
    "norm1000": (" Give every location as <loc_xmin_ymin_xmax_ymax> with "
                 "coordinates normalised to 0-999."),
    "pixel": " Give every location as <loc_xmin_ymin_xmax_ymax> in image pixels.",
}

REASON_PREFIX = " Reasoning: "


def target_box_token(record: dict, box_source: str, loc_format: str) -> str:
    """The ``<loc_...>`` span that supervises this record.

    ``box_source`` selects the published ``gt_location`` or the repaired
    ``gt_location_fixed``; ``loc_format`` selects the coordinate space.  The
    normalized space is fixed-width, which is what lets the soft-IoU loss find
    its digit tokens.
    """
    fixed = box_source == "fixed"
    if loc_format == "norm1000":
        key = "bbox_fixed_norm" if fixed else "bbox_norm"
        return format_loc(record[key], loc_token_width(LOC_NORM_RANGE))
    key = "bbox_fixed_orig" if fixed else "bbox_orig"
    return format_loc(record[key])


def build_question(record: dict, loc_format: str) -> str:
    return record["question"] + LOC_INSTRUCTION[loc_format]


def build_target(record: dict, box_source: str, loc_format: str,
                 include_reason: bool) -> tuple[str, tuple[int, int]]:
    """Return ``(target_text, (loc_start_char, loc_end_char))``.

    The answer text of the release already contains a ``<loc_...>`` span in
    original pixels.  Rather than patching that string, the answer is rebuilt
    around the span this config asks for, so that the supervised coordinates
    and the configured space can never drift apart.
    """
    loc = target_box_token(record, box_source, loc_format)
    stem = _answer_without_loc(record["answer"]).rstrip(" .")
    text = f"{stem} {loc}."
    start = text.index(loc)
    if include_reason and record.get("reason"):
        text += REASON_PREFIX + record["reason"].strip()
    return text, (start, start + len(loc))


def _answer_without_loc(answer: str) -> str:
    """Drop the published ``<loc_...>`` spans from an answer string."""
    from .boxes import LOC_RE
    return LOC_RE.sub("", answer).replace("  ", " ").strip()


def build_messages(record: dict, image, loc_format: str) -> list[dict]:
    """Chat messages in the format MedGemma's processor expects."""
    return [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": image},
                     {"type": "text", "text": build_question(record, loc_format)}]},
    ]
