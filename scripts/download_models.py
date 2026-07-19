"""Download model files listed in scripts/models_manifest.json.

    python scripts/download_models.py            # download everything with a source
    python scripts/download_models.py --dry-run  # show what would happen

Supports direct URLs (stdlib urllib, streamed to disk) and Hugging Face files
(requires `pip install huggingface_hub`). Entries with no source are skipped --
notably the LLM, which you build via Genie in WSL and copy in manually.

NOTE: verify every source in the manifest before the event. Nothing here asserts
that a given URL or repo id is correct forever; treat the defaults as hints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_manifest.json")


def _download_url(url: str, dest: str) -> None:
    # httpx verifies TLS via certifi (avoids the Windows "unable to get local
    # issuer certificate" error from stdlib urllib) and follows GitHub's
    # release-asset redirects. Writes to a .part file then renames atomically.
    import httpx

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest + ".part"
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded // (1 << 20)
                if total:
                    pct = downloaded * 100 // total
                    sys.stdout.write(f"\r  {pct:3d}% ({mb} MB)")
                else:
                    sys.stdout.write(f"\r  {mb} MB")
                sys.stdout.flush()
    os.replace(tmp, dest)
    sys.stdout.write("\r  done              \n")


def _download_hf(repo: str, filename: str, dest: str) -> None:
    from huggingface_hub import hf_hub_download

    print(f"  hf: {repo}/{filename}")
    path = hf_hub_download(repo_id=repo, filename=filename)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    import shutil

    shutil.copyfile(path, dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    failures = 0
    for entry in manifest.get("entries", []):
        name = entry.get("name")
        dest = os.path.join(ROOT, entry.get("dest", ""))
        url = entry.get("url") or ""
        hf_repo = entry.get("hf_repo") or ""
        hf_file = entry.get("hf_file") or ""
        print(f"[{name}] -> {dest}")
        if entry.get("note"):
            print(f"  note: {entry['note']}")

        if os.path.exists(dest):
            print("  already present, skipping.")
            continue
        if not url and not hf_repo:
            print("  no source configured, skipping.")
            continue
        if args.dry_run:
            print("  (dry-run) would download.")
            continue

        try:
            if url:
                _download_url(url, dest)
            else:
                _download_hf(hf_repo, hf_file, dest)
        except Exception as e:  # keep going through the rest
            failures += 1
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    print("\nDone." if not failures else f"\nDone with {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
