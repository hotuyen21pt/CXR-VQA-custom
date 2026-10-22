"""Tests for the generated Kaggle notebooks.

A notebook cannot be imported, so a typo in a cell is only found by running it
on Kaggle -- twenty minutes and an 8 GB model load later. These tests parse
every cell instead, and exercise the one piece of real logic the inference
notebook carries that the package does not: finding the LoRA adapter among the
shapes Kaggle hands it (a directory, a checkpoint, a zip).
"""

from __future__ import annotations

import ast
import json
import os
import sys
import zipfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import make_infer_notebook  # noqa: E402
import make_notebook  # noqa: E402

NOTEBOOKS = {
    "kaggle_train.ipynb": make_notebook,
    "kaggle_infer.ipynb": make_infer_notebook,
}


def _stub_magics(src: str) -> str:
    """IPython's `!cmd` and `%pip` lines are not Python; drop them."""
    out, skip = [], False
    for line in src.split("\n"):
        stripped = line.lstrip()
        if skip:
            skip = line.rstrip().endswith("\\")
            continue
        if stripped.startswith(("%", "!")):
            out.append(" " * (len(line) - len(stripped)) + "pass")
            skip = line.rstrip().endswith("\\")
        else:
            out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------
# The notebooks on disk
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_every_code_cell_parses(name):
    path = os.path.join(REPO, "notebooks", name)
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = _stub_magics("".join(cell["source"]))
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - only on a bad cell
            pytest.fail(f"{name} cell {i} line {exc.lineno}: {exc.msg}")


def _generated_cells(module):
    """The cells a generator would write.

    The two generators return different shapes -- make_notebook predates
    notebook_cells and builds the whole notebook itself, while
    make_infer_notebook returns just the cells for ``nc.notebook`` to wrap --
    so this normalises rather than forcing one of them to change.
    """
    built = module.build()
    return built["cells"] if isinstance(built, dict) else built


@pytest.mark.parametrize("name,module", sorted(NOTEBOOKS.items()))
def test_notebook_on_disk_matches_its_generator(name, module):
    """The .ipynb is generated; a hand edit to it would be lost on the next run."""
    path = os.path.join(REPO, "notebooks", name)
    with open(path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["cells"] == _generated_cells(module), (
        f"{name} is out of date; run python scripts/{module.__name__}.py")


# --------------------------------------------------------------------------
# The inference notebook's adapter resolution
# --------------------------------------------------------------------------
@pytest.fixture
def adapter_cell(tmp_path):
    """The adapter cell's helpers, with its unzip target redirected at tmp_path."""
    dest = str(tmp_path / "unzipped")
    source = make_infer_notebook.ADAPTER.split("if ADAPTER_PATH is None:")[0]
    source = source.replace('"/kaggle/working/adapter"', repr(dest))
    # The setup cell imported these for the whole notebook; the adapter cell
    # just uses them.
    namespace: dict = {"os": os, "json": json, "glob": __import__("glob"),
                       "shutil": __import__("shutil")}
    exec(compile(source, "<adapter cell>", "exec"), namespace)
    namespace["_dest"] = dest
    return namespace


def make_adapter(path, weights="adapter_model.safetensors"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as fh:
        json.dump({"r": 16, "lora_alpha": 32}, fh)
    with open(os.path.join(path, weights), "wb") as fh:
        fh.write(b"\0" * 16)
    return str(path)


def test_finished_run_prefers_the_final_adapter(adapter_cell, tmp_path):
    run = make_adapter(tmp_path / "in" / "medgemma-lora")
    make_adapter(os.path.join(run, "checkpoint-200"))
    make_adapter(os.path.join(run, "checkpoint-400"))
    assert adapter_cell["is_adapter"](run)
    assert adapter_cell["find_adapter"](str(tmp_path / "in")) == run


def test_interrupted_run_falls_back_to_the_newest_checkpoint(adapter_cell,
                                                            tmp_path):
    run = tmp_path / "in" / "medgemma-lora"
    make_adapter(run / "checkpoint-100")
    make_adapter(run / "checkpoint-300")
    assert not adapter_cell["is_adapter"](str(run))
    assert adapter_cell["newest_checkpoint"](str(run)).endswith("checkpoint-300")
    assert adapter_cell["find_adapter"](str(tmp_path / "in")).endswith(
        "checkpoint-300")


def test_bin_weights_count_as_an_adapter(adapter_cell, tmp_path):
    path = make_adapter(tmp_path / "lora", "adapter_model.bin")
    assert adapter_cell["is_adapter"](path)


def test_zip_written_by_make_archive_is_unpacked(adapter_cell, tmp_path):
    """shutil.make_archive zips the contents, so the files land at the root."""
    run = make_adapter(tmp_path / "run")
    zip_path = tmp_path / "in" / "medgemma-lora.zip"
    os.makedirs(zip_path.parent, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            z.write(os.path.join(run, name), name)
    found = adapter_cell["find_adapter"](str(tmp_path / "in"))
    assert found == adapter_cell["_dest"] and adapter_cell["is_adapter"](found)


def test_zip_with_a_folder_inside_is_unpacked(adapter_cell, tmp_path):
    run = make_adapter(tmp_path / "run")
    zip_path = tmp_path / "lora.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            z.write(os.path.join(run, name), f"medgemma-lora/{name}")
    found = adapter_cell["unzip_adapter"](str(zip_path))
    assert adapter_cell["is_adapter"](found)
    assert found.endswith("medgemma-lora")


def test_nothing_attached_returns_none(adapter_cell, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert adapter_cell["find_adapter"](str(empty)) is None


def test_the_image_tree_is_never_searched(adapter_cell, tmp_path):
    """4,394 PNGs sit under dataset/train_data/train; walking it is wasted time."""
    make_adapter(tmp_path / "in" / "dataset" / "train_data" / "train" / "x")
    assert adapter_cell["find_adapter"](str(tmp_path / "in")) is None
