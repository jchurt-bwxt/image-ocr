#!/usr/bin/env python3
"""Launch the desktop interface, or run one image from the command line."""
from __future__ import annotations

import argparse
from pathlib import Path

from ocr_engine import AnalysisOptions, MODES, OCREngine, load_image, save_outputs
from ocr_comparison import compare_image, run_model, save_comparison
from ocr_profiles import OCR_PROFILES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image OCR: launch the desktop UI with no arguments, or supply an image for CLI analysis.")
    parser.add_argument("image", type=Path, nargs="?", help="Image to analyze; omit to open the desktop UI.")
    parser.add_argument("--mode", choices=MODES, default="all", help="Search the whole image or inside OpenCV-detected rectangles/triangles.")
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="Minimum shape area as a fraction of image area.")
    parser.add_argument("--ocr-profile", choices=OCR_PROFILES, default="enhanced", help="Select baseline, enhanced (default), or large OCR.")
    parser.add_argument("--compare-models", action="store_true", help="Run all OCR models sequentially and export comparison.json plus model subfolders.")
    parser.add_argument("--large-max-seconds", type=float, default=120.0, help="Large-model processing time limit, greater than 0 and at most 120 s; excludes model download/load time.")
    parser.add_argument("--languages", default="en", help="Comma-separated OCR language codes. Enhanced/large support en; baseline supports other EasyOCR languages.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.image is None:
        from image_ocr_gui import launch_gui
        launch_gui()
        return
    options = AnalysisOptions(
        mode=args.mode,
        languages=tuple(part.strip() for part in args.languages.split(",") if part.strip()),
        device=args.device, min_area_ratio=args.min_area_ratio, ocr_profile=args.ocr_profile,
        large_max_seconds=args.large_max_seconds,
    )
    engine = None
    try:
        options.validate()
        if args.compare_models and (options.languages != ("en",) or options.device == "cuda"):
            raise ValueError("Compare all models requires English (en) and auto/CPU. Baseline alone supports other languages/CUDA.")
        image = load_image(args.image)
        engine = OCREngine()
        directory = args.output_dir or args.image.with_name(f"{args.image.stem}_results")
        if args.compare_models:
            report = compare_image(image, options, engine=engine, progress=print)
            report["image"] = str(args.image.resolve())
            for row in report["models"]:
                if row.get("payload") is not None:
                    row["payload"]["image"] = str(args.image.resolve())
            save_comparison(image, report, directory)
            rows = report["models"]
        else:
            row = run_model(image, options, engine, progress=print)
            if row["status"] == "ok" and row.get("payload") is not None:
                row["payload"]["image"] = str(args.image.resolve())
                save_outputs(image, row["payload"], directory)
            rows = [row]
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        raise SystemExit(f"Analysis failed: {exc}") from exc
    finally:
        if engine is not None:
            engine.close()
    for row in rows:
        print(f"\n{row['label']}: {row['status']} — startup {row['model_setup_seconds']:.2f} s, processing {row['analysis_seconds']:.2f} s, total {row['total_seconds']:.2f} s")
        print(row.get("text") or row.get("error") or "No text recognized.")
    if args.compare_models:
        print(f"Saved comparison.json and model subfolders to {directory}")
    elif rows[0]["status"] == "ok":
        print(f"Saved annotated.png, results.json and text.txt to {directory}")
    else:
        raise SystemExit(f"Analysis {rows[0]['status']}: {rows[0].get('error') or 'No result returned.'}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

