#!/usr/bin/env python3
"""Prompt-based image detection followed by OCR inside each detected region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import easyocr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find prompt-matching regions in an image with a pretrained "
            "open-vocabulary detector, then read text inside each region."
        )
    )
    parser.add_argument("image", type=Path, help="Input image (.jpg, .jpeg, .png, etc.)")
    parser.add_argument(
        "--prompt",
        required=True,
        help='What to detect, preferably a short phrase (for example: "rectangular object").',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Minimum detection confidence from 0 to 1 (default: 0.30).",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated EasyOCR language codes (default: en; for example: en,fr).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto; CUDA requires a compatible PyTorch install).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Hugging Face model ID or local model folder containing config.json and "
            f"SafeTensors weights (default: {DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save results (default: a folder named <image>_results next to the image).",
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable in this PyTorch install.")
    return requested


def resolve_model_source(model: str) -> tuple[str, bool]:
    """Return the model ID/path and whether it is a local SafeTensors snapshot."""
    model_path = Path(model).expanduser()
    if model_path.is_file():
        if model_path.suffix == ".safetensors":
            raise SystemExit(
                "Pass the model folder, not the .safetensors file itself. "
                "The folder also needs config and processor/tokenizer files."
            )
        raise SystemExit(f"--model points to a file, but a model ID or folder is required: {model_path}")
    if not model_path.is_dir():
        return model, False

    has_weights = (model_path / "model.safetensors").is_file()
    has_sharded_weights = (model_path / "model.safetensors.index.json").is_file()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Local model folder is missing config.json: {model_path}")
    if not (has_weights or has_sharded_weights):
        raise SystemExit(
            f"Local model folder has no model.safetensors or model.safetensors.index.json: {model_path}"
        )
    return str(model_path.resolve()), True


def detect_and_read(
    image: Image.Image,
    prompt: str,
    threshold: float,
    detector: Any,
    reader: easyocr.Reader,
) -> list[dict[str, Any]]:
    detections = detector(image, candidate_labels=[prompt], threshold=threshold)
    image_width, image_height = image.size
    results: list[dict[str, Any]] = []

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        x1 = max(0, min(image_width, int(box["xmin"])))
        y1 = max(0, min(image_height, int(box["ymin"])))
        x2 = max(0, min(image_width, int(box["xmax"])))
        y2 = max(0, min(image_height, int(box["ymax"])))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        # detail=1 returns [polygon, text, confidence] for each recognized line.
        ocr_lines = reader.readtext(np.asarray(crop), detail=1, paragraph=False)
        text_lines = [
            {"text": str(line[1]), "confidence": float(line[2])}
            for line in ocr_lines
            if len(line) >= 3 and str(line[1]).strip()
        ]
        results.append(
            {
                "id": index,
                "label": detection.get("label", prompt),
                "confidence": float(detection["score"]),
                "box_xyxy": [x1, y1, x2, y2],
                "text": " ".join(line["text"] for line in text_lines),
                "ocr_lines": text_lines,
            }
        )
    return results


def save_annotated(image: Image.Image, results: list[dict[str, Any]], path: Path) -> None:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for item in results:
        x1, y1, x2, y2 = item["box_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
        text = f'{item["label"]} {item["confidence"]:.2f}'
        if item["text"]:
            text += f' | {item["text"][:70]}'
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle((left, top, right, bottom), fill=(255, 40, 40))
        draw.text((x1, y1), text, fill="white", font=font)
    annotated.save(path)


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Input image does not exist: {args.image}")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1.")
    if not args.prompt.strip():
        raise SystemExit("--prompt cannot be empty.")

    device = select_device(args.device)
    model_source, is_local_model = resolve_model_source(args.model)
    languages = [part.strip() for part in args.languages.split(",") if part.strip()]
    if not languages:
        raise SystemExit("Provide at least one OCR language with --languages.")
    output_dir = args.output_dir or args.image.with_name(f"{args.image.stem}_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to RGB for consistent detector/OCR input across image formats.
    with Image.open(args.image) as source:
        image = source.convert("RGB")

    if is_local_model:
        print(f"Loading local SafeTensors checkpoint from {model_source!r} on {device}.")
    else:
        print(f"Loading detector {model_source!r} on {device}; first run downloads model weights.")
    detector = pipeline(
        task="zero-shot-object-detection",
        model=model_source,
        device=0 if device == "cuda" else -1,
        # Keep Hub options at the pipeline level. Transformers passes these to
        # AutoConfig/model loading itself; duplicating them in model_kwargs can
        # produce duplicate-keyword errors in recent Transformers versions.
        local_files_only=is_local_model,
        model_kwargs={
            "use_safetensors": True,
        },
    )
    reader = easyocr.Reader(languages, gpu=(device == "cuda"))
    results = detect_and_read(image, args.prompt.strip(), args.threshold, detector, reader)

    json_path = output_dir / "results.json"
    annotated_path = output_dir / "annotated.png"
    payload = {
        "image": str(args.image.resolve()),
        "prompt": args.prompt.strip(),
        "model": model_source,
        "threshold": args.threshold,
        "detections": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_annotated(image, results, annotated_path)

    print(f"Found {len(results)} region(s).")
    for item in results:
        recognized = item["text"] or "(no text recognized)"
        print(f'  #{item["id"]} {item["label"]} ({item["confidence"]:.2f}): {recognized}')
    print(f"Saved structured results to {json_path}")
    print(f"Saved annotated image to {annotated_path}")


if __name__ == "__main__":
    main()
