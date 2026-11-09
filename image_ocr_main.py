#!/usr/bin/env python3
"""Launch the desktop interface, or run one image from the command line."""
from __future__ import annotations

import argparse
from pathlib import Path

from ocr_engine import AnalysisOptions, DEFAULT_MODEL, MODES, OCREngine, extracted_text, load_image, save_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image OCR: launch the desktop UI with no arguments, or supply an image for CLI analysis.")
    parser.add_argument("image", type=Path, nargs="?", help="Image to analyze; omit to open the desktop UI.")
    parser.add_argument("--mode", choices=MODES, default=None, help="Default: all text, or prompt mode if --prompt is supplied.")
    parser.add_argument("--prompt", default="", help="Object description for prompt mode, for example 'rectangular sign'.")
    parser.add_argument("--threshold", type=float, default=0.30, help="Prompt detection confidence, from 0 to 1.")
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="Minimum shape area as a fraction of image area.")
    parser.add_argument("--languages", default="en", help="Comma-separated EasyOCR language codes, for example en,fr.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Grounding DINO model ID or local SafeTensors snapshot folder; used only in prompt mode.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.image is None:
        from image_ocr_gui import launch_gui
        launch_gui()
        return
    options = AnalysisOptions(
        mode=args.mode or ("prompt" if args.prompt.strip() else "all"),
        prompt=args.prompt, threshold=args.threshold, model=args.model,
        languages=tuple(part.strip() for part in args.languages.split(",") if part.strip()),
        device=args.device, min_area_ratio=args.min_area_ratio,
    )
    try:
        options.validate()
        image = load_image(args.image)
        payload = OCREngine().analyze(image, options, progress=print)
        payload["image"] = str(args.image.resolve())
        directory = args.output_dir or args.image.with_name(f"{args.image.stem}_results")
        save_outputs(image, payload, directory)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        raise SystemExit(f"Analysis failed: {exc}") from exc
    print(extracted_text(payload) or "No text recognized.")
    print(f"Saved annotated.png, results.json and text.txt to {directory}")


if __name__ == "__main__":
    main()

