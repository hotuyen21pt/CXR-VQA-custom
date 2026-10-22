"""HTTP backend for the inference UI in ``ui/``.

One process holds the model, so loading MedGemma's weights -- minutes -- is
paid once and every answer after that costs seconds.  The React app never
touches torch; it asks this server over HTTP, which means the GPU can be a
Kaggle session, a Colab runtime or a remote box while the UI runs on a laptop.

    # the real thing, on the machine that has the GPU and the adapter
    python -m vindr_cxr_vqa.serve --adapter outputs/medgemma-lora

    # zero-shot baseline: same command without --adapter
    python -m vindr_cxr_vqa.serve

    # no GPU, no torch: canned answers, so the UI itself can be exercised
    python -m vindr_cxr_vqa.serve --mock

Standard library only -- ``http.server`` rather than FastAPI -- so that the
mock mode, which is the one most likely to be run on a bare machine, needs no
``pip install`` at all.  Pillow, torch and transformers are imported lazily,
only when a real model is actually being served.

Every box leaves this server in three forms: ``raw`` as the model wrote it,
``pixel`` in the image's own coordinates, and ``rel`` normalised to 0-1.  The
UI draws ``rel`` as percentages, so an overlay is correct whatever size the
browser renders the image at.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import posixpath
import random
import re
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .boxes import LOC_RE, Box, extract_locs, from_normalized, greedy_match, iou
from .config import load_config
from .constants import LOC_NORM_RANGE, PATHOLOGIES, QUESTION_TYPES
from .evaluate import mentioned
from .prompts import REASON_PREFIX

UI_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "dist")


# --------------------------------------------------------------------------
# finding the adapter among the shapes a training run leaves behind
# --------------------------------------------------------------------------
def is_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def newest_checkpoint(path: str) -> str | None:
    """The highest-numbered ``checkpoint-N`` inside an interrupted run."""
    if not os.path.isdir(path):
        return None
    found = [(int(m.group(1)), os.path.join(path, name))
             for name in os.listdir(path)
             if (m := re.fullmatch(r"checkpoint-(\d+)", name))
             and is_adapter(os.path.join(path, name))]
    return max(found)[1] if found else None


def stale(dest: str, src: str) -> bool:
    """Whether ``dest`` was extracted from an older version of ``src``.

    Compared against the extracted config rather than the directory itself,
    because a directory's mtime moves when anything inside it is touched.
    """
    try:
        return (os.path.getmtime(src)
                > os.path.getmtime(os.path.join(dest, "adapter_config.json")))
    except OSError:
        return False


def unzip_adapter(src: str, dest: str) -> None:
    """Extract the adapter from ``src``, leaving the training debris behind.

    A Kaggle archive carries the final adapter next to every checkpoint it
    kept, and each of those drags an optimizer state twice the size of the
    weights -- 1.1 GB of zip for 114 MB that inference needs. When the final
    adapter is there, only the files beside it are extracted.

    Resuming training still wants the checkpoints, but that is train.py's job
    and it reads the archive itself.
    """
    import zipfile
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        root_adapter = "adapter_config.json" in names
        wanted = [n for n in names
                  if not n.endswith("/")
                  and (not root_adapter or "/" not in n)]
        z.extractall(dest, members=wanted)


def resolve_adapter(path: str | None) -> str | None:
    """Turn whatever the user pasted into a directory peft can load.

    A Kaggle run hands back ``medgemma-lora.zip`` -- ``/kaggle/working`` is
    wiped, so the training notebook archives the adapter as notebook output.
    Unzipping it by hand is a step worth not asking for, and an interrupted
    run leaves only ``checkpoint-N`` subdirectories, which is the other shape
    that turns up in practice.
    """
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))

    if path.lower().endswith(".zip"):
        if not os.path.isfile(path):
            raise SystemExit(f"--adapter: no such file: {path}")
        dest = os.path.join(os.path.dirname(path),
                            os.path.basename(path)[:-4] + "-unzipped")
        if is_adapter(dest) and stale(dest, path):
            # A finished run is usually dropped in under the name the last one
            # had. Reusing the old extraction then serves the previous
            # adapter against the new archive, silently, and the only symptom
            # is that a model which should have improved did not.
            print(f"[serve] {path} is newer than {dest}; re-extracting")
            shutil.rmtree(dest)
        if not is_adapter(dest):
            unzip_adapter(path, dest)
            print(f"[serve] unzipped {path} -> {dest}")
        # make_archive zips the contents, so the files may sit at the root or
        # one directory down, depending on how the zip was made.
        if is_adapter(dest):
            return dest
        for name in sorted(os.listdir(dest)):
            inner = os.path.join(dest, name)
            if is_adapter(inner):
                return inner
        raise SystemExit(f"--adapter: no adapter_config.json inside {path}")

    if not os.path.isdir(path):
        raise SystemExit(f"--adapter: no such directory: {path}")
    if is_adapter(path):
        return path
    ckpt = newest_checkpoint(path)
    if ckpt:
        print(f"[serve] {path} holds no final adapter; using {ckpt}")
        return ckpt
    raise SystemExit(
        f"--adapter: {path} has no adapter_config.json and no checkpoint-N "
        "inside it. Point it at the directory train.py wrote "
        "(outputs/medgemma-lora), or at the .zip the Kaggle notebook saved."
        + renamed_hint(path))


#: Downloading a checkpoint file by file through a browser often rewrites the
#: extension -- adapter_config.json arrives as adapter_config.txt. The weights
#: keep their name, so the directory looks almost right and the error does not
#: mention the files that are sitting there.
EXPECTED_NAMES = {
    "adapter_config": "adapter_config.json",
    "tokenizer": "tokenizer.json",
    "tokenizer_config": "tokenizer_config.json",
    "chat_template": "chat_template.jinja",
    "preprocessor_config": "preprocessor_config.json",
}


def renamed_hint(path: str) -> str:
    """A rename suggestion when the directory holds the right files, renamed."""
    try:
        names = os.listdir(path)
    except OSError:
        return ""
    if not any(n.startswith("adapter_model.") for n in names):
        return ""

    renames = []
    for name in sorted(names):
        stem, ext = os.path.splitext(name)
        want = EXPECTED_NAMES.get(stem)
        if want and name != want:
            renames.append((name, want))
    if not renames:
        return ""

    lines = "\n".join(f"  mv {os.path.join(path, a)} {os.path.join(path, b)}"
                      for a, b in renames)
    return ("\n\nThe weights are there, so this looks like a checkpoint whose "
            "files were renamed on the way down (a browser saving .json as "
            ".txt does this):\n" + lines)


# --------------------------------------------------------------------------
# the model, and the dataset it is asked about
# --------------------------------------------------------------------------
class Engine:
    """Everything a request needs: the records, the images, and the model.

    The model is loaded on first use rather than at startup, so the server
    answers ``/api/health`` immediately and the UI can say "loading" instead
    of leaving a dead port for the several minutes a cold load takes.
    """

    def __init__(self, cfg, adapter: str | None, mock: bool,
                 load_in_4bit: bool | None = None,
                 max_new_tokens: int | None = None):
        self.cfg = cfg
        self.adapter = adapter
        self.mock = mock
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens or int(cfg.eval.max_new_tokens)
        self.model = None
        self.processor = None
        self.state = "mock" if mock else "cold"
        self.error: str | None = None
        self._lock = threading.Lock()

        self.index: dict[str, dict] = {}
        self.records: dict[str, list[dict]] = {}
        self._load_dataset()

    # -- dataset -----------------------------------------------------------
    def _load_dataset(self) -> None:
        pdir = self.cfg.paths.processed_dir
        path = os.path.join(pdir, "images.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.index = json.load(fh)
        for split in ("val", "test", "train"):
            jsonl = os.path.join(pdir, f"{split}.jsonl")
            if not os.path.exists(jsonl):
                continue
            with open(jsonl, encoding="utf-8") as fh:
                self.records[split] = [json.loads(l) for l in fh if l.strip()]

    def record(self, sample_id: str) -> dict | None:
        for rows in self.records.values():
            for r in rows:
                if r["sample_id"] == sample_id:
                    return r
        return None

    def image_path(self, image_id: str) -> str | None:
        img = self.index.get(image_id)
        if not img:
            return None
        path = os.path.join(self.cfg.paths.dataset_dir, img["image_path"])
        return path if os.path.exists(path) else None

    # -- model -------------------------------------------------------------
    def ensure_model(self) -> None:
        """Load the weights once, under a lock, and remember a failure.

        Two browser tabs hitting predict at the same moment on a cold server
        would otherwise load MedGemma twice and run the GPU out of memory.
        """
        if self.mock or self.state == "ready":
            return
        with self._lock:
            if self.state == "ready":
                return
            if self.state == "error":
                raise RuntimeError(self.error or "model failed to load")
            self.state = "loading"
            try:
                from .infer import load_model, load_processor
                print("[serve] loading the processor...", flush=True)
                self.processor = load_processor(self.cfg, self.adapter)
                print("[serve] loading the model (this takes minutes)...",
                      flush=True)
                self.model = load_model(self.cfg, self.adapter,
                                        self.load_in_4bit)
                self.state = "ready"
                print("[serve] model ready", flush=True)
            except Exception as exc:                      # noqa: BLE001
                self.state = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                raise

    def generate(self, question: str, image, max_new_tokens: int) -> str:
        from .infer import answer
        return answer(self.model, self.processor, self.cfg,
                      [{"question": question}], [image],
                      max_new_tokens=max_new_tokens)[0]


# --------------------------------------------------------------------------
# coordinate bookkeeping
# --------------------------------------------------------------------------
def pred_space(cfg) -> str:
    """The coordinate space the model was trained to emit."""
    return "norm1000" if cfg.data.loc_format == "norm1000" else "pixel"


def to_pixel(box: Box, space: str, width: int, height: int) -> Box:
    if space == "norm1000":
        return from_normalized(box, width, height, LOC_NORM_RANGE)
    return box


def to_rel(box: Box, width: int, height: int) -> list[float]:
    """0-1 of the image's own extent, which is what the overlay draws."""
    w, h = max(width, 1), max(height, 1)
    return [round(box[0] / w, 5), round(box[1] / h, 5),
            round(box[2] / w, 5), round(box[3] / h, 5)]


def split_answer(text: str) -> tuple[str, str]:
    """Separate the answer from the ``Reasoning:`` tail the target appends."""
    marker = REASON_PREFIX.strip()
    cut = text.find(marker)
    if cut == -1:
        return text.strip(), ""
    return text[:cut].strip(), text[cut + len(marker):].strip()


def describe_boxes(text: str, space: str, width: int, height: int,
                   gts_pixel: list[Box], iou_threshold: float) -> list[dict]:
    """Every ``<loc_...>`` in the answer, in all three spaces, scored on IoU."""
    raws = extract_locs(text)
    pixels = [to_pixel(b, space, width, height) for b in raws]
    matches, _, _ = greedy_match(pixels, gts_pixel, iou_threshold)
    matched = {i: (j, v) for i, j, v in matches}

    out = []
    for i, (raw, px) in enumerate(zip(raws, pixels)):
        best = max((iou(px, g) for g in gts_pixel), default=None)
        j, v = matched.get(i, (None, None))
        out.append({
            "raw": [int(round(c)) for c in raw],
            "pixel": [round(c, 1) for c in px],
            "rel": to_rel(px, width, height),
            "best_iou": round(best, 4) if best is not None else None,
            "matched_gt": j,
            "matched_iou": round(v, 4) if v is not None else None,
        })
    return out


def describe_gt(img: dict | None) -> list[dict]:
    """The fused consensus boxes an image carries, if it is a dataset image."""
    if not img:
        return []
    w, h = img["width"], img["height"]
    return [{
        "class_name": b["class_name"],
        "pixel": [round(c, 1) for c in b["bbox_orig"]],
        "rel": to_rel(tuple(b["bbox_orig"]), w, h),
    } for b in img.get("boxes_fused", [])]


# --------------------------------------------------------------------------
# mock answers
# --------------------------------------------------------------------------
def mock_answer(record: dict | None, question: str, cfg) -> str:
    """A plausible answer with no model behind it, for exercising the UI.

    For a dataset record it replays the ground truth with the box nudged a few
    percent, so that the overlay shows two distinct rectangles and the IoU
    arithmetic has something real to chew on.  For an uploaded image there is
    no ground truth, so it invents a centred box.  Either way the response is
    flagged ``mock: true``, and none of it reflects what the model would say.
    """
    rng = random.Random(question)
    norm = cfg.data.loc_format == "norm1000"
    hi = LOC_NORM_RANGE - 1

    if record is not None:
        key = "bbox_fixed_norm" if norm else "bbox_fixed_orig"
        fallback = record["bbox_norm" if norm else "bbox_orig"]
        box = list(record.get(key) or fallback)
        span = max(box[2] - box[0], box[3] - box[1]) or 10
        box = [c + rng.uniform(-0.09, 0.09) * span for c in box]
        stem = LOC_RE.sub("", record["answer"]).replace("  ", " ").strip(" .")
        reason = record.get("reason", "")
    else:
        lo, high = (0.32 * hi, 0.70 * hi) if norm else (300.0, 700.0)
        box = [lo, lo, high, high]
        stem = ("The finding is localized to the area defined by the "
                "coordinates")
        reason = ("Mock mode: no model is running. This answer is canned, for "
                  "checking the interface only.")

    width = 3 if norm else 0
    loc = "<loc_" + "_".join(f"{int(round(max(0.0, c))):0{width}d}"
                             for c in box) + ">"
    text = f"{stem} {loc}."
    return text + REASON_PREFIX + reason if reason else text


# --------------------------------------------------------------------------
# the API
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    engine: Engine = None            # type: ignore[assignment]
    server_version = "vindr-cxr-vqa"

    # -- plumbing ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The UI is served from a different origin in development (Vite on
        # 5173) and through a tunnel in a Kaggle session, so the API is open.
        # It serves a read-only model over a LAN; do not expose it publicly.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self):                                 # noqa: N802
        self._send(204, b"", "text/plain")

    # -- routing -----------------------------------------------------------
    def do_GET(self):                                     # noqa: N802
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/api/health":
                return self._json(self.health())
            if url.path == "/api/meta":
                return self._json(self.meta())
            if url.path == "/api/samples":
                return self._json(self.samples(query))
            if url.path == "/api/image":
                return self.image(query)
            return self.static(url.path)
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):                                    # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path == "/api/predict":
                return self._json(self.predict(self._read_body()))
            if url.path == "/api/load":
                self.engine.ensure_model()
                return self._json(self.health())
            return self._json({"error": "not found"}, 404)
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- endpoints ---------------------------------------------------------
    def health(self) -> dict:
        eng = self.engine
        return {
            "state": eng.state,
            "mock": eng.mock,
            "adapter": eng.adapter,
            "mode": "fine-tuned" if eng.adapter else "pretrained (zero-shot)",
            "model": eng.cfg.model.name,
            "error": eng.error,
            "images_indexed": len(eng.index),
            "splits": {k: len(v) for k, v in eng.records.items()},
            "dataset_dir": eng.cfg.paths.dataset_dir,
            "dataset_dir_exists": os.path.isdir(eng.cfg.paths.dataset_dir),
        }

    def meta(self) -> dict:
        cfg = self.engine.cfg
        return {
            "pathologies": list(PATHOLOGIES),
            "question_types": list(QUESTION_TYPES),
            "loc_format": cfg.data.loc_format,
            "pred_space": pred_space(cfg),
            "box_source": cfg.data.box_source,
            "iou_threshold": float(cfg.eval.iou_threshold),
            "good_iou": float(cfg.eval.good_localization_iou),
            "max_new_tokens": self.engine.max_new_tokens,
        }

    def samples(self, query: dict) -> dict:
        eng = self.engine
        split = (query.get("split") or ["val"])[0]
        rows = eng.records.get(split, [])
        needle = (query.get("q") or [""])[0].strip().lower()
        qtype = (query.get("type") or [""])[0]
        finding = (query.get("finding") or [""])[0]
        limit = min(int((query.get("limit") or ["60"])[0]), 500)
        offset = int((query.get("offset") or ["0"])[0])

        def keep(r: dict) -> bool:
            if qtype and r.get("type") != qtype:
                return False
            if finding and r.get("gt_finding") != finding:
                return False
            if needle:
                hay = f"{r['question']} {r['image_id']} {r.get('gt_finding')}"
                if needle not in hay.lower():
                    return False
            return True

        hits = [r for r in rows if keep(r)]
        page = hits[offset:offset + limit]
        return {
            "total": len(hits),
            "offset": offset,
            "items": [{
                "sample_id": r["sample_id"],
                "image_id": r["image_id"],
                "question": r["question"],
                "type": r.get("type"),
                "difficulty": r.get("difficulty"),
                "gt_finding": r.get("gt_finding"),
                "box_label_consistent": r.get("box_label_consistent"),
            } for r in page],
        }

    def image(self, query: dict) -> None:
        image_id = (query.get("image_id") or [""])[0]
        path = self.engine.image_path(image_id)
        if not path:
            return self._json({"error": f"no image for {image_id!r}; check "
                                        "paths.dataset_dir"}, 404)
        with open(path, "rb") as fh:
            body = fh.read()
        self._send(200, body, "image/png", {"Cache-Control": "max-age=3600"})

    def predict(self, body: dict) -> dict:
        eng, cfg = self.engine, self.engine.cfg
        sample_id = body.get("sample_id")
        record = eng.record(sample_id) if sample_id else None
        if sample_id and record is None:
            return {"error": f"unknown sample_id {sample_id!r}"}

        question = (body.get("question") or "").strip()
        if not question and record:
            question = record["question"]
        if not question:
            return {"error": "question is required"}

        img_meta = eng.index.get(record["image_id"]) if record else None
        width = int(body.get("width") or (img_meta or {}).get("width") or 0)
        height = int(body.get("height") or (img_meta or {}).get("height") or 0)
        space = pred_space(cfg)

        started = time.time()
        if eng.mock:
            text = mock_answer(record, question, cfg)
        else:
            eng.ensure_model()
            image = self._open_image(body, record)
            if image is None:
                return {"error": "no image: pass sample_id, or image_b64"}
            if not width or not height:
                width, height = image.size
            text = eng.generate(question, image,
                                int(body.get("max_new_tokens")
                                    or eng.max_new_tokens))
        elapsed = time.time() - started

        # An uploaded image with no VinDr metadata still needs a frame for the
        # pixel numbers; the normalised space needs none, so this only bites
        # when the model was trained to emit raw pixels.
        if not width or not height:
            width = height = LOC_NORM_RANGE - 1

        gt = describe_gt(img_meta)
        gts_pixel = [tuple(g["pixel"]) for g in gt]
        boxes = describe_boxes(text, space, width, height, gts_pixel,
                               float(cfg.eval.iou_threshold))
        answer_text, reason = split_answer(text)

        return {
            "mock": eng.mock,
            "prediction": text,
            "answer": answer_text,
            "reason": reason,
            "findings": sorted(mentioned(text)),
            "pred_space": space,
            "pred_boxes": boxes,
            "gt_boxes": gt,
            "best_iou": max((b["best_iou"] or 0.0 for b in boxes), default=None),
            "latency_s": round(elapsed, 2),
            "image_width": width,
            "image_height": height,
            "image_url": (f"/api/image?image_id={record['image_id']}"
                          if record else None),
            "record": {
                "sample_id": record["sample_id"],
                "gt_finding": record.get("gt_finding"),
                "gt_answer": record.get("answer"),
                "gt_reason": record.get("reason"),
                "type": record.get("type"),
            } if record else None,
        }

    @staticmethod
    def _open_image(body: dict, record: dict | None):
        from PIL import Image
        blob = body.get("image_b64")
        if blob:
            if "," in blob[:64]:                 # strip a data: URL prefix
                blob = blob.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(blob))).convert("RGB")
        if record:
            path = Handler.engine.image_path(record["image_id"])
            if path:
                return Image.open(path).convert("RGB")
        return None

    # -- the built React app, when there is one ----------------------------
    def static(self, path: str) -> None:
        if not os.path.isdir(UI_DIST):
            return self._json({"error": "API only; the UI is not built. Run "
                                        "`npm run dev` in ui/, or `npm run "
                                        "build` to have it served from here."},
                              404)
        rel = posixpath.normpath(path).lstrip("/") or "index.html"
        target = os.path.join(UI_DIST, *rel.split("/"))
        if not os.path.isfile(target):
            target = os.path.join(UI_DIST, "index.html")   # SPA fallback
        with open(target, "rb") as fh:
            body = fh.read()
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        self._send(200, body, ctype)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--adapter", default=os.environ.get("VINDR_ADAPTER"),
                    help="LoRA directory from training; omit for the "
                         "zero-shot baseline. Env: VINDR_ADAPTER")
    ap.add_argument("--model", default=os.environ.get("VINDR_MODEL"),
                    help="override model.name: a local MedGemma directory or "
                         "a hub id. Env: VINDR_MODEL")
    ap.add_argument("--dataset-dir", default=os.environ.get("VINDR_DATASET_DIR"),
                    help="override paths.dataset_dir (where the PNGs are)")
    ap.add_argument("--processed-dir",
                    default=os.environ.get("VINDR_PROCESSED_DIR"),
                    help="override paths.processed_dir (the built JSONL)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true",
                    help="serve canned answers; no torch, no GPU, no weights")
    ap.add_argument("--preload", action="store_true",
                    help="load the weights at startup instead of on the first "
                         "question")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--dtype", default=os.environ.get("VINDR_DTYPE"),
                    help="override model.torch_dtype (bfloat16, float16, "
                         "float32). Default 'auto' picks bf16 off-GPU, where "
                         "torch has no fp16 CPU kernels. Env: VINDR_DTYPE")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.dtype:
        cfg.model.torch_dtype = args.dtype
    if args.dataset_dir:
        cfg.paths.dataset_dir = os.path.abspath(args.dataset_dir)
    if args.processed_dir:
        cfg.paths.processed_dir = os.path.abspath(args.processed_dir)

    adapter = resolve_adapter(args.adapter)
    engine = Engine(cfg, adapter, args.mock,
                    load_in_4bit=False if args.no_4bit else None,
                    max_new_tokens=args.max_new_tokens)
    Handler.engine = engine

    mode = "MOCK (no model)" if args.mock else cfg.model.name
    print(f"[serve] mode      : {mode}")
    print(f"[serve] adapter   : {adapter or '(none - zero-shot baseline)'}")
    print(f"[serve] dataset   : {cfg.paths.dataset_dir} "
          f"({'ok' if os.path.isdir(cfg.paths.dataset_dir) else 'MISSING'})")
    print(f"[serve] processed : {cfg.paths.processed_dir} "
          f"({len(engine.index)} images, "
          f"{sum(len(v) for v in engine.records.values())} QA pairs)")
    if args.preload and not args.mock:
        engine.ensure_model()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n[serve] http://{args.host}:{args.port}  (Ctrl-C to stop)")
    print("[serve] point the UI's backend URL at that address\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")


if __name__ == "__main__":
    main()
