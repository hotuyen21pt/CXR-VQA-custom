"""Download MedGemma-4B-it to a local folder for offline use.

Run this on a machine WITH internet, then upload the result as a Kaggle
Dataset so a no-internet notebook can attach it.

    huggingface-cli login            # the model is gated: accept its licence
    python scripts/fetch_model.py    # -> outputs/models/medgemma-4b-it (~8 GB)

Then: kaggle.com/datasets -> New Dataset -> upload that folder.

Needs `huggingface_hub`; everything else in the repo does not.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_REPO = "google/medgemma-4b-it"

#: Weights plus the tokenizer/processor files `from_pretrained` needs.
#: Excludes the duplicate .bin/.gguf/.pth copies, which would double the size.
ALLOW = ["*.safetensors", "*.json", "*.model", "*.txt"]
IGNORE = ["*.bin", "*.pth", "*.gguf", "*.h5", "*.msgpack", "original/*"]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token", default=None,
                    help="HF token; falls back to HF_TOKEN or a cached login")
    ap.add_argument("--all-files", action="store_true",
                    help="do not filter out .bin/.gguf duplicates")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(repo_root, "outputs", "models",
                                   args.repo.split("/")[-1])
    os.makedirs(out, exist_ok=True)

    # `or None` matters: an exported-but-empty HF_TOKEN -- which is what a
    # .env file with `HF_TOKEN=` produces -- is an empty string, not absent.
    # Passed through, it becomes the header `Bearer ` and the hub fails with
    # "Illegal header value", which reads like a bug in the client rather
    # than a missing token.
    token = args.token or os.environ.get("HF_TOKEN") or None
    print(f"[model] downloading {args.repo}\n[model]   -> {out}")
    print("[model] this is ~8 GB and gated; accept the licence at "
          f"https://huggingface.co/{args.repo} first")
    if token is None:
        print("[model] no token: --token, HF_TOKEN or a cached "
              "`hf auth login` are all unset. A gated repo will 401.")

    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=out,
            token=token,
            allow_patterns=None if args.all_files else ALLOW,
            ignore_patterns=None if args.all_files else IGNORE,
        )
    except Exception as exc:                      # noqa: BLE001 — report and exit
        print(f"\n[model] download failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print("[model] if this is a 401/403, you have not accepted the model "
              "licence or HF_TOKEN is unset", file=sys.stderr)
        raise SystemExit(1)

    files = sorted(os.listdir(out))
    print(f"\n[model] {len(files)} files, {human(dir_size(out))}")
    for name in files:
        if not name.startswith("."):
            print("  ", name)

    missing = [n for n in ("config.json", "preprocessor_config.json",
                           "tokenizer.json")
               if not os.path.exists(os.path.join(out, n))]
    if missing:
        print(f"\n[model] WARNING: missing {missing} — "
              "re-run with --all-files")
    else:
        print("\n[model] ready. Upload this folder as a Kaggle Dataset, then "
              "set model.name to its /kaggle/input path.")


if __name__ == "__main__":
    main()
