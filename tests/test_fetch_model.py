"""Tests for the offline model downloader.

The allow-list is the whole risk here. A pattern that is too narrow does not
fail loudly: the eight gigabytes arrive, the processor loads, and only the
first question reveals that some small file never came. So the test is about
which files survive the filter, not about downloading anything.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import fetch_model  # noqa: E402

#: What google/medgemma-1.5-4b-it actually holds, from the hub API.
MEDGEMMA_FILES = [
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def kept(files):
    """The files snapshot_download would actually fetch."""
    return {f for f in files
            if any(fnmatch.fnmatch(f, p) for p in fetch_model.ALLOW)
            and not any(fnmatch.fnmatch(f, p) for p in fetch_model.IGNORE)}


def test_everything_from_pretrained_needs_is_kept():
    survivors = kept(MEDGEMMA_FILES)
    for required in ("config.json", "tokenizer.json", "tokenizer.model",
                     "preprocessor_config.json", "processor_config.json",
                     "model.safetensors.index.json",
                     "model-00001-of-00002.safetensors",
                     "model-00002-of-00002.safetensors"):
        assert required in survivors, f"{required} would not be downloaded"


def test_the_chat_template_is_kept():
    """transformers moved it out of tokenizer_config.json into its own file.

    Without it the processor loads and then raises "this processor does not
    have a chat template" on the first prompt -- after the 8 GB is already on
    disk.
    """
    assert "chat_template.jinja" in kept(MEDGEMMA_FILES)


def test_the_duplicate_weight_formats_are_not():
    """The same weights as .bin or .gguf would double the download."""
    survivors = kept(MEDGEMMA_FILES + ["pytorch_model.bin", "model.gguf",
                                       "tf_model.h5", "original/consolidated.pth"])
    assert not {f for f in survivors
                if f.endswith((".bin", ".gguf", ".h5", ".pth"))}


# --------------------------------------------------------------------------
# has_chat_template
# --------------------------------------------------------------------------
def test_a_jinja_file_counts(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    assert fetch_model.has_chat_template(str(tmp_path))


def test_a_template_embedded_in_tokenizer_config_counts(tmp_path):
    """Older models keep it there; warning about those would cry wolf."""
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}))
    assert fetch_model.has_chat_template(str(tmp_path))


def test_a_snapshot_with_neither_is_reported(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "x"}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
    assert not fetch_model.has_chat_template(str(tmp_path))


def test_unreadable_json_does_not_crash_the_check(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{ this is not json")
    assert not fetch_model.has_chat_template(str(tmp_path))


def test_an_empty_directory_has_no_template(tmp_path):
    assert not fetch_model.has_chat_template(str(tmp_path))
