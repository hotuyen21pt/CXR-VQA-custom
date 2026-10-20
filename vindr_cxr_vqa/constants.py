"""Fixed vocabularies and paths for VinDr-CXR-VQA.

Everything here is taken verbatim from the paper (arXiv:2511.00504v2) or from
the released dataset card, so that downstream code never hard-codes a string.
"""

from __future__ import annotations

# --- 14 pathology classes kept by the paper (Section 2.1) -------------------
# Ordered by prevalence in the released data_v1.json, which matches the
# percentages printed on the HuggingFace dataset card.
PATHOLOGIES: tuple[str, ...] = (
    "Cardiomegaly",
    "Pulmonary fibrosis",
    "Lung Opacity",
    "Pleural effusion",
    "Aortic enlargement",
    "Calcification",
    "Other lesion",
    "Pleural thickening",
    "Infiltration",
    "Pneumothorax",
    "Nodule/Mass",
    "ILD",
    "Consolidation",
    "Atelectasis",
)
PATHOLOGY_TO_ID = {name: i for i, name in enumerate(PATHOLOGIES)}

# The full VinDr-CXR local-label vocabulary (22 findings). Present so that the
# builder can tell "dropped by the paper" apart from "not a finding at all".
VINDR_LOCAL_LABELS: tuple[str, ...] = (
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA",
    "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst",
    "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture",
    "Other lesion",
)
NO_FINDING = "No finding"

# --- 6 question types (Section 2.1) ----------------------------------------
QUESTION_TYPES: tuple[str, ...] = (
    "Where",      # spatial localization
    "What",       # pathology identification
    "Is_there",   # existence verification
    "How_many",   # counting
    "Which",      # anatomical classification
    "Yes_No",     # binary confirmation
)
DIFFICULTIES: tuple[str, ...] = ("Easy", "Medium")

# --- Surface forms used to decide whether a generated answer mentions a class.
# Keys are canonical class names; values are lowercase substrings to search for.
PATHOLOGY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement",
                     "enlargement of the cardiac silhouette"),
    "Pulmonary fibrosis": ("pulmonary fibrosis", "fibrosis", "fibrotic"),
    "Lung Opacity": ("lung opacity", "pulmonary opacity", "opacity", "opacities"),
    "Pleural effusion": ("pleural effusion", "effusion"),
    "Aortic enlargement": ("aortic enlargement", "enlarged aorta",
                           "aortic dilatation", "widened aorta"),
    "Calcification": ("calcification", "calcified", "calcific"),
    "Other lesion": ("other lesion",),
    "Pleural thickening": ("pleural thickening", "thickened pleura"),
    "Infiltration": ("infiltration", "infiltrate"),
    "Pneumothorax": ("pneumothorax",),
    "Nodule/Mass": ("nodule/mass", "nodule", "nodular", "mass"),
    "ILD": ("ild", "interstitial lung disease", "interstitial disease"),
    "Consolidation": ("consolidation", "consolidative"),
    "Atelectasis": ("atelectasis", "atelectatic", "collapse"),
}

# --- Default layout of the local VinDr-CXR copy ----------------------------
DEFAULT_IMAGE_SIZE = 224          # train/ and test/ PNGs are pre-resized
LOC_NORM_RANGE = 1000             # <loc_...> normalized to [0, 999]

__all__ = [
    "PATHOLOGIES", "PATHOLOGY_TO_ID", "VINDR_LOCAL_LABELS", "NO_FINDING",
    "QUESTION_TYPES", "DIFFICULTIES", "PATHOLOGY_SYNONYMS",
    "DEFAULT_IMAGE_SIZE", "LOC_NORM_RANGE",
]
