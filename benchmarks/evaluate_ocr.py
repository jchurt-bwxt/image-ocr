"""Evaluate full-image OCR against independent image-level transcriptions.

Labels are used exclusively for scoring, never as OCR inputs or to select a
prediction. The primary metric is corpus alphanumeric CER: total edit distance
divided by total label characters after NFKC/casefold and removal of every
non-alphanumeric character. Strict CER retains punctuation and collapses runs
of whitespace to one space. Extra background text is counted as insertion
errors because the full prediction is compared with the full supplied label.
CER can exceed 1 when there are many insertions. Failed images are scored as
empty predictions and remain in every aggregate denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class BenchmarkCase:
    image_id: str
    reference: str
    image_path: Path


def normalize_strict(text: str) -> str:
    """Ignore case and whitespace formatting, preserving punctuation."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalize_alnum(text: str) -> str:
    """Ignore case, whitespace, and punctuation; retain letters and digits."""
    return "".join(char for char in normalize_strict(text) if char.isalnum())


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    """Levenshtein distance with unit insertion/deletion/substitution costs."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, ref in enumerate(reference, 1):
        current = [row]
        for column, pred in enumerate(prediction, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (ref != pred),
            ))
        previous = current
    return previous[-1]


def score_text(reference: str, prediction: str) -> dict[str, Any]:
    result = {}
    for name, normalizer in (("strict", normalize_strict), ("alnum", normalize_alnum)):
        ref, pred = normalizer(reference), normalizer(prediction)
        edits = edit_distance(ref, pred)
        result[name] = {
            "reference": ref,
            "prediction": pred,
            "character_edits": edits,
            "reference_characters": len(ref),
            "cer": edits / max(1, len(ref)),
            "exact_match": ref == pred,
        }
    return result


def load_cases(labels_path: str | Path, images_directory: str | Path) -> list[BenchmarkCase]:
    """Map each unique label ID to exactly one image stem, ignoring case."""
    labels_path, images_directory = Path(labels_path), Path(images_directory)
    if not labels_path.is_file():
        raise ValueError(f"Labels file does not exist: {labels_path}")
    if not images_directory.is_dir():
        raise ValueError(f"Images directory does not exist: {images_directory}")
    labels = []
    seen = set()
    for line_number, line in enumerate(labels_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if ";" not in line:
            raise ValueError(f"Label line {line_number} must contain 'image_id; transcription'.")
        image_id, reference = (part.strip() for part in line.split(";", 1))
        if not image_id or not reference:
            raise ValueError(f"Label line {line_number} has an empty image ID or transcription.")
        if "/" in image_id or "\\" in image_id:
            raise ValueError(f"Label line {line_number}: image IDs must be filenames without directories.")
        key = image_id.casefold()
        if key in seen:
            raise ValueError(f"Duplicate image ID on label line {line_number}: {image_id}")
        seen.add(key)
        labels.append((image_id, reference))
    if not labels:
        raise ValueError("The labels file contains no labeled images.")

    images: dict[str, list[Path]] = {}
    for image_path in sorted(images_directory.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(image_path.stem.casefold(), []).append(image_path)
    cases = []
    for image_id, reference in labels:
        matches = images.get(image_id.casefold(), [])
        if not matches:
            raise ValueError(f"No image found for label ID {image_id!r} in {images_directory}.")
        if len(matches) != 1:
            raise ValueError(f"Multiple images have the label stem {image_id!r}: {', '.join(p.name for p in matches)}")
        cases.append(BenchmarkCase(image_id, reference, matches[0]))
    return cases


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    summary: dict[str, Any] = {
        "image_count": count,
        "successful_images": sum(result["error"] is None for result in results),
        "failed_images": sum(result["error"] is not None for result in results),
        "total_image_seconds": sum(result["seconds"] for result in results),
        "metrics": {},
    }
    for name in ("strict", "alnum"):
        metrics = [result["metrics"][name] for result in results]
        edits = sum(metric["character_edits"] for metric in metrics)
        characters = sum(metric["reference_characters"] for metric in metrics)
        exact = sum(metric["exact_match"] for metric in metrics)
        summary["metrics"][name] = {
            "character_edits": edits,
            "reference_characters": characters,
            "corpus_cer": edits / max(1, characters),
            "mean_image_cer": sum(metric["cer"] for metric in metrics) / max(1, count),
            "exact_match_count": exact,
            "exact_match_rate": exact / max(1, count),
        }
    return summary


def evaluate_cases(
    cases: list[BenchmarkCase],
    engine: Any,
    options: Any,
    *,
    image_loader: Callable | None = None,
    text_extractor: Callable | None = None,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run one OCR prediction per case without exposing its label to the engine."""
    if image_loader is None or text_extractor is None:
        from ocr_engine import extracted_text, load_image
        image_loader = image_loader or load_image
        text_extractor = text_extractor or extracted_text
    results = []
    for index, case in enumerate(cases, 1):
        progress(f"[{index}/{len(cases)}] {case.image_id}")
        start = clock()
        prediction, error = "", None
        diagnostics = []
        timing = {}
        try:
            image = image_loader(case.image_path)
            payload = engine.analyze(image, options)
            prediction = text_extractor(payload)
            diagnostics = payload.get("ocr_diagnostics", [])
            timing = payload.get("timing", {})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            progress(f"  ERROR: {error}")
        elapsed = max(0.0, clock() - start)
        metrics = score_text(case.reference, prediction)
        results.append({
            "image_id": case.image_id,
            "image_path": str(case.image_path),
            "reference": case.reference,
            "prediction": prediction,
            "metrics": metrics,
            "seconds": elapsed,
            "error": error,
            "ocr_diagnostics": diagnostics,
            "timing": timing,
        })
        progress(f"  Alnum CER: {metrics['alnum']['cer']:.3f}; {elapsed:.2f} s")
    return results


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_info() -> dict[str, Any]:
    packages = {}
    for package in (
        "easyocr", "rapidocr", "onnxruntime", "torch", "torchvision", "numpy",
        "opencv-python-headless", "opencv-python", "Pillow", "transformers", "truststore", "huggingface-hub", "requests", "certifi",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    source_hashes = {name: file_sha256(ROOT / name) for name in (
        "ocr_engine.py", "ocr_readers.py", "ocr_model_assets.py", "large_ocr.py", "ocr_profiles.py", "ocr_comparison.py", "ocr_network.py",
    )}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "ocr_engine_sha256": source_hashes["ocr_engine.py"],
        "source_sha256": source_hashes,
    }


def build_options(profile: str, device: str, languages: tuple[str, ...]):
    from ocr_engine import AnalysisOptions
    kwargs: dict[str, Any] = {"mode": "all", "device": device, "languages": languages}
    if "ocr_profile" in inspect.signature(AnalysisOptions).parameters:
        kwargs["ocr_profile"] = profile
    elif profile != "baseline":
        raise ValueError("This backend does not yet support the enhanced OCR profile.")
    options = AnalysisOptions(**kwargs)
    options.validate()
    return options


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from ocr_profiles import OCR_PROFILES, PROFILE_LABELS
    parser.add_argument("--profile", choices=OCR_PROFILES, default="baseline")
    parser.add_argument("--large-max-seconds", type=float, default=120)
    parser.add_argument("--labels", type=Path, default=ROOT / "sample_images" / "id_img_labels.txt")
    parser.add_argument("--images", type=Path, default=ROOT / "sample_images")
    parser.add_argument("--output", type=Path, help="JSON report path (default: benchmarks/results/<profile>.json)")
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--languages", default="en", help="Comma-separated EasyOCR language codes.")
    parser.add_argument("--ids", help="Optional comma-separated image IDs to evaluate, preserving label order.")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.labels, args.images)
        if args.ids is not None:
            selected = {value.strip().casefold() for value in args.ids.split(",") if value.strip()}
            available = {case.image_id.casefold() for case in cases}
            if not selected or selected - available:
                raise ValueError(f"Unknown or empty --ids selection: {', '.join(sorted(selected - available))}")
            cases = [case for case in cases if case.image_id.casefold() in selected]
        languages = tuple(value.strip() for value in args.languages.split(",") if value.strip())
        options = build_options(args.profile, args.device, languages)
        from dataclasses import replace
        options = replace(options, large_max_seconds=args.large_max_seconds)
        options.validate()
    except ValueError as exc:
        parser.error(str(exc))

    from ocr_engine import OCREngine
    start = time.perf_counter()
    engine = OCREngine()
    try:
        results = evaluate_cases(cases, engine, options)
    finally:
        engine.close()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": PROFILE_LABELS[args.profile],
        "profile": args.profile,
        "configuration": asdict(options),
        "environment": environment_info(),
        "dataset": {
            "labels_path": str(args.labels.resolve()),
            "labels_sha256": file_sha256(args.labels),
            "images_directory": str(args.images.resolve()),
            "image_sha256": {case.image_id: file_sha256(case.image_path) for case in cases},
        },
        "evaluation_scope": (
            "Full extracted text versus full image-level transcription; incidental background text counts as insertions. "
            "Labels are used only for scoring. Failed images are scored as empty predictions. "
            "Primary metric: corpus alphanumeric CER (NFKC/casefold, letters and digits only). "
            "Strict CER retains punctuation and collapses whitespace. CER may exceed 1. "
            "First-image timing includes lazy model initialization/downloads; subsequent images reuse the engine."
        ),
        "wall_seconds": time.perf_counter() - start,
        "summary": summarize(results),
        "images": results,
    }
    output = args.output or ROOT / "benchmarks" / "results" / f"{args.profile}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(f"\n{summary['image_count']} images, {summary['failed_images']} failures")
    for name, metrics in summary["metrics"].items():
        print(f"{name}: corpus CER {metrics['corpus_cer']:.4f}, mean CER {metrics['mean_image_cer']:.4f}, exact {metrics['exact_match_count']}/{summary['image_count']}")
    print(f"Report: {output}")
    return 1 if summary["failed_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
