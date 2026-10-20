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
    """
    import torch
    if isinstance(requested, torch.dtype):
        return requested
    if requested in ("auto", None, ""):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
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
