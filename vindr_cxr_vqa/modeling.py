"""Loading MedGemma across transformers versions and GPU sizes.

Shared by :mod:`train` and :mod:`infer` so the two can never drift apart in
how they quantize, pick a dtype, or call ``from_pretrained``.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

#: Headroom, in GiB, below which 4-bit base weights are worth the trouble.
#: MedGemma-4B is ~8 GB in bf16; training adds activations and optimizer state.
FP16_VRAM_FLOOR_GB = 24.0

# torch and transformers are imported inside the functions that need them, so
# that the quantization decision stays importable -- and testable -- on a
# machine with no GPU stack installed.


def has_bitsandbytes() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def gpu_memory_gb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def resolve_dtype(requested: Any) -> Any:
    """Turn ``model.torch_dtype`` into a real dtype.

    ``auto`` means bf16 where the card supports it (Ampere and newer) and fp16
    otherwise, which is what the paper used on its A6000.

    With no CUDA at all it means bf16 as well, for a different reason: torch's
    CPU kernels are largely missing for fp16, so a model loaded that way dies
    on the first matmul with "addmm_impl_cpu_ not implemented for 'Half'".
    bf16 has the CPU kernels and the same 8 GB footprint, which is what makes
    the container on a laptop run at all.
    """
    import torch
    if isinstance(requested, torch.dtype):
        return requested
    if requested in ("auto", None, ""):
        if not torch.cuda.is_available():
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, str(requested))


def build_quantization_config(load_in_4bit: bool, dtype: Any = None):
    """4-bit NF4 config, or ``None`` when full-precision weights are used.

    Returns ``(config, enabled)``. When 4-bit is asked for but bitsandbytes is
    missing, this falls back to unquantized weights if the GPU is large enough
    and otherwise stops with an actionable message -- rather than letting
    transformers raise an ImportError deep inside ``from_pretrained``.
    """
    if not load_in_4bit:
        return None, False

    if not has_bitsandbytes():
        vram = gpu_memory_gb()
        if vram >= FP16_VRAM_FLOOR_GB:
            print(f"[model] bitsandbytes is not installed, but this GPU has "
                  f"{vram:.0f} GB — loading unquantized weights instead "
                  f"(faster, and what the paper did).")
            return None, False
        raise SystemExit(
            "train.load_in_4bit is true but bitsandbytes is not installed, "
            f"and this GPU has only {vram:.0f} GB.\n"
            "Either `pip install -U bitsandbytes>=0.46.1`, or set "
            "train.load_in_4bit: false if the model fits unquantized.")

    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    ), True


def disable_broken_peft_dispatchers() -> list[str]:
    """Stop peft from tripping over optional backends it cannot use.

    When peft attaches LoRA it walks a list of dispatchers, one per
    quantization backend. The probe for an unusable backend is supposed to
    return ``False``, but several of them *raise* when the package is present
    and merely out of date -- Kaggle's image ships torchao 0.10 against a peft
    that wants 0.16+, and ``get_peft_model`` dies with an ImportError before
    any adapter is created.

    None of those backends are involved here: the base weights are bf16, or
    4-bit via bitsandbytes, which has its own dispatcher. So a probe that
    raises is forced to ``False``. A probe that answers normally is left
    alone, and nothing is installed, removed or upgraded.

    Returns the names of the probes that had to be disabled.
    """
    try:
        from peft import import_utils
    except ImportError:
        return []

    disabled: list[str] = []
    for probe_name, module_path in (
        ("is_torchao_available", "peft.tuners.lora.torchao"),
        ("is_aqlm_available", "peft.tuners.lora.aqlm"),
        ("is_auto_awq_available", "peft.tuners.lora.awq"),
        ("is_eetq_available", "peft.tuners.lora.eetq"),
        ("is_hqq_available", "peft.tuners.lora.hqq"),
    ):
        probe = getattr(import_utils, probe_name, None)
        if probe is None:
            continue
        try:
            probe()
            continue
        except Exception as exc:                      # noqa: BLE001
            print(f"[model] peft's {probe_name}() fails ({exc}); disabling "
                  "that dispatcher, which this run does not use.")

        setattr(import_utils, probe_name, lambda: False)
        # The dispatcher module bound the name at import time, so patching
        # import_utils alone would not be seen.
        try:
            mod = importlib.import_module(module_path)
            setattr(mod, probe_name, lambda: False)
        except ImportError:
            pass
        disabled.append(probe_name)
    return disabled


def load_base_model(cfg, load_in_4bit: bool | None = None):
    """Load MedGemma with the right dtype, quantization and device map.

    Returns ``(model, used_4bit)``; ``used_4bit`` tells the caller whether to
    run PEFT's k-bit preparation and pick a paged optimizer.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    if load_in_4bit is None:
        load_in_4bit = bool(cfg.train.load_in_4bit)
    dtype = resolve_dtype(cfg.model.torch_dtype)
    quant, used_4bit = build_quantization_config(load_in_4bit, dtype)

    kwargs = dict(
        quantization_config=quant,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    # transformers renamed torch_dtype to dtype in 5.0 and warns on the old
    # name; try the new one first and fall back for older releases.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model.name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model.name, torch_dtype=dtype, **kwargs)

    print(f"[model] {cfg.model.name}")
    print(f"[model] dtype={dtype}, 4-bit={used_4bit}, "
          f"vram={gpu_memory_gb():.0f} GB")
    return model, used_4bit


# --------------------------------------------------------------------------
# TrainingArguments across the 4.x / 5.x split
# --------------------------------------------------------------------------
#: Arguments transformers dropped, and the name that took the job over.
#: 5.15 removed ``warmup_ratio``: ``warmup_steps`` now accepts a float in
#: [0, 1) and reads it as a fraction of the schedule, which is exactly what
#: the old argument meant. Older releases keep both, so the ratio has to be
#: handed to whichever name the installed version actually has.
TRAINING_ARG_RENAMES = {"warmup_ratio": "warmup_steps"}


def accepted_arguments(func) -> set[str] | None:
    """Keyword names ``func(...)`` takes, or ``None`` when it takes ``**kwargs``."""
    import inspect
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)}


def adapt_training_kwargs(kwargs: dict, accepted: set[str] | None):
    """Fit ``TrainingArguments`` kwargs to the installed transformers.

    Renamed arguments move to their new name; anything this version has no
    home for at all is dropped rather than raising, so a run that is already
    an hour in does not die on a keyword. Returns ``(kwargs, notes)``, where
    the notes are lines the caller should print -- a silent drop would quietly
    change how the model trains.
    """
    if accepted is None:
        return dict(kwargs), []

    out, notes = {}, []
    for name, value in kwargs.items():
        if name in accepted:
            out[name] = value
            continue
        new = TRAINING_ARG_RENAMES.get(name)
        if new and new in accepted and new not in kwargs:
            out[new] = value
            notes.append(f"this transformers has no {name}; passing it as "
                         f"{new}={value!r} instead")
        else:
            notes.append(f"WARNING: this transformers does not accept "
                         f"{name}={value!r}; dropping it")
    return out, notes


# --------------------------------------------------------------------------
# Which modules LoRA is allowed to wrap
# --------------------------------------------------------------------------
#: Submodules that belong to the image encoder rather than the language model.
VISION_PREFIXES = ("vision_tower", "multi_modal_projector")


def language_side_targets(module_names, suffixes, exclude=VISION_PREFIXES):
    """Fully-qualified LoRA targets, with the image encoder left out.

    peft matches ``target_modules`` by suffix, so a bare ``q_proj`` also wraps
    every attention block of the siglip vision tower. Those adapters are
    created *after* the freeze loop has run, which quietly makes the tower
    trainable again -- and a trainable tower is one whose activations must all
    be kept for the backward pass. With eager attention at batch 4 that is a
    4 GiB attention map per layer across 27 layers, which is what filled a
    95 GB card at step 0.

    (Only q/k/v leaked, not the output projection: siglip calls that one
    ``out_proj``, so ``o_proj`` never matched it.)

    Resolving the names here rather than relying on peft's matching rules
    keeps the decision visible, and behaves the same on every peft version.
    """
    wanted = set(suffixes)
    return [name for name in module_names
            if name.rsplit(".", 1)[-1] in wanted
            and not any(p in name for p in exclude)]
