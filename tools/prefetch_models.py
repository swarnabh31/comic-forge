"""Warm up the comic-rendering pipeline BEFORE the first real request.

WHY THIS EXISTS
---------------
The app downloads its image model LAZILY: nothing is fetched when `uvicorn`
starts, and the first "Generate comic!" click is the one that pays for the
one-time model download (Qwen-Image-2.1 is ~33 GB) PLUS the one-time pipeline
load (~20-30 s). On a slow or flaky connection that first click looks like the
app is frozen.

This script does that one-time work now, under progress bars you can watch:

    python tools/prefetch_models.py

It reads .env, so it automatically targets whichever IMAGE_PROVIDER you have
active (qwen or local-SDXL). It also CHECKS the local Hugging Face cache FIRST:
if you already downloaded the model for some other task, the script reports
"already in cache" and no download is triggered. It only fetches what is
genuinely missing (huggingface_hub matches files individually and resumes
incomplete ones — it never re-downloads a file that is already complete).

The cache it looks at is the standard one:
    ~/.cache/huggingface/hub/        (or wherever HF_HOME / HF_HUB_CACHE points)
If you downloaded your models into a custom folder earlier, point the same
folder at the script and the app, e.g. on Windows:
    set HF_HOME=D:\hf-cache
    python tools/prefetch_models.py
After that, both this script and the app will find and reuse that cache.

The download is resume-safe: if it fails part-way, just re-run the script and
it continues where it stopped.

After this finishes, the first "Generate comic!" request skips both the
download and the pipeline build and goes straight to drawing.

FORCE A RE-DOWNLOAD (rare — e.g. corrupt cache):
    python tools/prefetch_models.py --force
"""
import argparse
import sys
import time
from pathlib import Path

# Make `import config` etc. work from the repo's top level.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import config  # noqa: E402  (reads .env)


def _cache_status(repo_id: str) -> tuple[bool, str]:
    """Check the local Hugging Face cache for `repo_id` WITHOUT downloading.

    Returns (fully_cached, detail). Uses huggingface_hub in local-only mode,
    so no network call is made: it either finds every required file present
    on disk, or it names what's missing.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        snapshot_download(repo_id, local_files_only=True)
        return True, "all files present locally — no download needed"
    except LocalEntryNotFoundError:
        return False, "not (fully) present in the local cache"
    except Exception as exc:  # repo missing from cache entirely, refs unresolved, ...
        return False, f"{type(exc).__name__}: {exc}"[:160]


def _cached_size(repo_id: str) -> str | None:
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir():
            if repo.repo_id == repo_id:
                return f"{repo.size_on_disk / 1024**3:.1f} GiB on disk"
    except Exception:
        pass
    return None


def _announce(model_id: str, size_hint: str, extra: str = "") -> None:
    """Step 0: explicit cache check so the user sees FOUND vs NOT FOUND."""
    complete, detail = _cache_status(model_id)
    line = _cached_size(model_id)
    if complete:
        print(f"  [cache-check] ✅ {model_id}: {detail}" + (f" ({line})" if line else ""))
        print(f"  [cache-check]    → skipping download; the pipeline load below will be a fast local load.")
    else:
        print(f"  [cache-check] ⚠️  {model_id}: {detail}")
        print(f"  [cache-check]    → downloading ~{size_hint} (resume-safe; re-run continues where it stopped).")
        if extra:
            print(f"  [cache-check]    → {extra}")


def _force_drop_cache(repo_id: str) -> bool:
    """Actually delete a repo's cache so the next load re-downloads it.

    Uses huggingface_hub's own cache manager (verified present in 1.33):
    `scan_cache_dir().delete_files(*files).execute()`. We enumerate the repo
    through the official API (repos -> revisions -> files) rather than
    hand-deleting directory paths, so we never touch files that belong to
    other repos sharing the same cache folder.
    """
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    repo = next((r for r in info.repos if r.repo_id == repo_id), None)
    if repo is None:
        print(f"  [cache] nothing cached for {repo_id} to delete")
        return False
    files = [f for revision in repo.revisions for f in revision.files]
    if not files:
        print(f"  [cache] {repo_id}: repo present but no files found; nothing to do")
        return False
    strategy = info.delete_files(*files)
    print(f"  [cache] {repo_id}: {len(files)} files, expected freed {strategy.expected_freed_size_str}")
    strategy.execute()
    print(f"  [cache] {repo_id}: deleted. Next run will re-download it.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download + pre-warm the comic pipeline models.")
    parser.add_argument("--force", action="store_true", help="Re-download weights even if they are already cached (corrupt-cache recovery).")
    args = parser.parse_args()

    provider = config.IMAGE_PROVIDER
    print(f"IMAGE_PROVIDER = {provider}")
    print(f"Cache location = {Path.home() / '.cache' / 'huggingface' / 'hub'}  (override with HF_HOME)")

    if args.force:
        if provider == "qwen":
            repos = [config.QWEN_MODEL_ID]
        elif provider == "local":
            repos = [config.SD_MODEL_ID, config.IP_ADAPTER_REPO]
        else:
            print("\n--force has nothing to do for IMAGE_PROVIDER=hf_inference (no local models).")
            return 0
        print("\n--force will DELETE the locally cached model weights and re-download:")
        for repo_id in repos:
            print(f"    - {repo_id}")
        confirm = input("Delete the cached model(s) and re-download? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled — cache left untouched.")
            return 1
        for repo_id in repos:
            _force_drop_cache(repo_id)
        print("\nProceeding with a fresh download + pipeline warm-up…\n")

    if provider == "qwen":
        import qwen_gen
        _announce(qwen_gen._QWEN_MODEL_ID, "~33 GB")
        print(f"\n[1/2] Loading pipeline '{qwen_gen._QWEN_MODEL_ID}' …")
        t0 = time.time()
        pipe = qwen_gen._load_pipeline()
        print(f"[1/2] Pipeline built in {time.time() - t0:.1f}s")

        print("\n[2/2] First real inference (forces every component to initialize once)…")
        t0 = time.time()
        qwen_gen.generate("a red circle on a white background", None, 42, size=512, steps=4)
        print(f"[2/2] Test image generated in {time.time() - t0:.1f}s (discarded)")
        assert pipe is qwen_gen._pipe, "internal pipeline state mismatch"

    elif provider == "local":
        import imagegen
        _announce(imagegen.config.SD_MODEL_ID, "~7 GB")
        _announce(
            imagegen.config.IP_ADAPTER_REPO,
            "~1 GB",
            f"needed for character references (weight '{imagegen.config.IP_ADAPTER_WEIGHT_SDXL}'),"
            " downloaded automatically if missing",
        )
        print(f"\n[1/2] Loading pipeline '{imagegen.config.SD_MODEL_ID}' …")
        t0 = time.time()
        imagegen._load_pipeline()
        print(f"[1/2] SDXL pipeline built in {time.time() - t0:.1f}s")

        print(f"\n[2/2] Loading IP-Adapter from '{imagegen.config.IP_ADAPTER_REPO}' (~1 GB)…")
        t0 = time.time()
        imagegen._ensure_ip_adapter()
        print(f"[2/2] IP-Adapter ready in {time.time() - t0:.1f}s")

        print("\nTest inference (discarded)…")
        t0 = time.time()
        imagegen.generate("a red circle on a white background", None, 42, size=512, steps=4)
        print(f"Test image generated in {time.time() - t0:.1f}s")

    elif provider == "hf_inference":
        print("\nNo local models to prefetch (IMAGE_PROVIDER=hf_inference uses the HF API).")
        print("Make sure HF_TOKEN is set in .env and that you can reach the Hugging Face API.")

    else:
        print(f"\nERROR: unknown IMAGE_PROVIDER '{provider}'. Expected qwen, local, or hf_inference.", file=sys.stderr)
        return 1

    print("\nDone. The first 'Generate comic!' request will skip the download and pipeline build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
