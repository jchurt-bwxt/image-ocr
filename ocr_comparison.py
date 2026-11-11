"""Sequential model comparisons with explicit cold-start and processing times."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Callable

from ocr_engine import AnalysisOptions, OCREngine, extracted_text, save_outputs
from ocr_profiles import OCR_PROFILES, PROFILE_LABELS


def run_model(image, options: AnalysisOptions, engine: OCREngine, progress: Callable[[str], None] = lambda _: None) -> dict:
    started = time.perf_counter()
    payload, error, status = None, None, "ok"
    try:
        payload = engine.analyze(image, options, progress=progress)
    except TimeoutError as exc:
        status, error = "timeout", str(exc)
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    timing = payload.get("timing", {}) if payload is not None else getattr(engine, "last_timing", {})
    if not isinstance(timing, dict):
        timing = {}
    return {
        "profile": options.ocr_profile, "label": PROFILE_LABELS[options.ocr_profile],
        "status": status, "total_seconds": elapsed,
        "model_setup_seconds": timing.get("model_setup_seconds", 0.0),
        "analysis_seconds": timing.get("analysis_seconds", elapsed),
        "model_reused": timing.get("model_reused", False),
        "text": extracted_text(payload) if payload is not None else "",
        "payload": payload, "error": error,
    }


def compare_image(image, options: AnalysisOptions, engine: OCREngine | None = None, progress=print) -> dict:
    if options.device == "cuda" or tuple(options.languages) != ("en",):
        raise ValueError("Model comparison requires English and CPU processing.")
    if options.device == "auto":
        options = replace(options, device="cpu")
    options.validate()
    owned = engine is None
    engine = engine or OCREngine()
    started = time.perf_counter()
    rows = []
    try:
        for profile in OCR_PROFILES:
            progress(f"Comparing {PROFILE_LABELS[profile]}…")
            rows.append(run_model(image, replace(options, ocr_profile=profile), engine, progress))
    finally:
        if owned:
            engine.close()
    return {"image_size": list(image.size), "mode": options.mode, "device": options.device, "models": rows,
            "wall_seconds": time.perf_counter() - started,
            "timing_scope": "Loaded image to OCR result; model setup/download is included in total and listed separately. File selection, common image decoding, preview rendering, and export are excluded.",
            "large_inference_budget_seconds": options.large_max_seconds}


def save_comparison(image, report: dict, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # An overwritten comparison must not retain an old successful result for a
    # model that failed, timed out, or was skipped in the new run.
    for profile in OCR_PROFILES:
        for name in ("annotated.png", "results.json", "text.txt"):
            path = directory / profile / name
            if path.is_file():
                path.unlink()
    (directory / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in report["models"]:
        if row.get("payload") is not None:
            save_outputs(image, row["payload"], directory / row["profile"])
