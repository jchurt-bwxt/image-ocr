"""Download the evaluated OCR weights from a pinned public Hugging Face mirror.

Every file must match the original evaluated SHA256 before RapidOCR receives
its local path. No model registry fallback is used by this downloader.
"""
from __future__ import annotations

import hashlib
from http.client import HTTPException
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


MODEL_REPOSITORY = "DjB314/RapidOCR"
MODEL_REVISION = "04e88d0483f0bfcdd0f29429594244c10bfc2867"
ASSETS = {
    "det": {
        "filename": "PP-OCRv6_det_medium.onnx",
        "source_path": "v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx",
        "bytes": 62119454,
        "sha256": "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2",
    },
    "rec": {
        "filename": "PP-OCRv6_rec_medium.onnx",
        "source_path": "v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
        "bytes": 76629984,
        "sha256": "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
    },
    "cls": {
        "filename": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "source_path": "v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "bytes": 585532,
        "sha256": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    },
}


def asset_url(asset: dict) -> str:
    return f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{asset['source_path']}"


def valid_model(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset["sha256"]


def _download_model(cache: Path, asset: dict) -> Path:
    destination = cache / asset["filename"]
    temporary = None
    try:
        request = Request(asset_url(asset), headers={"User-Agent": "image-ocr-model-download/1.0"})
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=cache, prefix=asset["filename"] + ".", suffix=".part", delete=False,
        ) as output:
            temporary = Path(output.name)
            downloaded = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                downloaded += len(chunk)
                if downloaded > asset["bytes"]:
                    raise ValueError("Downloaded model exceeds its expected size.")
                output.write(chunk)
        if not valid_model(temporary, asset):
            raise ValueError("Downloaded model size or SHA256 does not match the evaluated weights.")
        # Failed/interrupted downloads never replace a valid local checkpoint.
        temporary.replace(destination)
        return destination
    except (OSError, ValueError, HTTPException) as exc:
        raise RuntimeError(
            f"Could not download {asset['filename']} from Hugging Face: {exc}\n"
            f"Download URL: {asset_url(asset)}\n"
            f"You can manually download that file into {cache} and retry."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_enhanced_model_paths(cache: str | Path) -> dict[str, Path]:
    """Reuse verified local weights, or atomically fetch missing/corrupt ones."""
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stage, asset in ASSETS.items():
        destination = cache / asset["filename"]
        paths[stage] = destination if valid_model(destination, asset) else _download_model(cache, asset)
    return paths


if __name__ == "__main__":
    import argparse
    from ocr_readers import model_cache_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=model_cache_directory())
    args = parser.parse_args()
    for stage, path in get_enhanced_model_paths(args.cache_dir).items():
        print(f"{stage}: {path}")
