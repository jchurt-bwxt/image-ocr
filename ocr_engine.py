"""Shared OCR and region analysis for the desktop interface and CLI.

Model libraries are imported only when needed. Whole-image OCR and geometry
search never load the prompt detector. One engine can be reused across images.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
MODES = ("all", "rectangles", "triangles", "prompt")


@dataclass(frozen=True)
class AnalysisOptions:
    mode: str = "all"
    prompt: str = ""
    threshold: float = 0.30
    languages: tuple[str, ...] = ("en",)
    device: str = "auto"
    model: str = DEFAULT_MODEL
    min_area_ratio: float = 0.001
    ocr_profile: str = "enhanced"

    def validate(self) -> None:
        if self.ocr_profile not in ("baseline", "enhanced"):
            raise ValueError("OCR profile must be baseline or enhanced.")
        if self.ocr_profile == "enhanced" and tuple(self.languages) != ("en",):
            raise ValueError("Enhanced OCR is evaluated for English markings. Choose baseline for other OCR languages.")
        if self.ocr_profile == "enhanced" and self.device == "cuda":
            raise ValueError("Enhanced OCR uses ONNX Runtime on CPU. Choose auto/cpu, or baseline for CUDA OCR.")
        if self.mode not in MODES:
            raise ValueError(f"Choose an analysis mode from {', '.join(MODES)}.")
        if self.mode == "prompt" and not self.prompt.strip():
            raise ValueError("Enter an object description for prompt detection.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Detection threshold must be between 0 and 1.")
        if not 0 < self.min_area_ratio <= 1:
            raise ValueError("Minimum shape area must be greater than 0 and at most 1.")
        if not self.languages or any(not lang.strip() for lang in self.languages):
            raise ValueError("Enter at least one OCR language code, for example en.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda.")


def load_image(path: str | Path) -> Image.Image:
    """Normalize camera orientation and composite transparent pixels onto white."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(background, image).convert("RGB")


def select_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Choose CPU or install a compatible PyTorch build.")
    return requested


def resolve_model_source(model: str) -> tuple[str, bool]:
    if not model.strip():
        raise ValueError("Choose a detector model folder or enter a Hugging Face model ID.")
    path = Path(model).expanduser()
    if path.is_file():
        raise ValueError("Select the model folder containing config.json and SafeTensors weights, not a file.")
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise ValueError(f"Model folder is missing config.json: {path}")
        if not any((path / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
            raise ValueError(f"Model folder is missing model.safetensors or its shard index: {path}")
        return str(path.resolve()), True
    # Recognize explicit paths so a typo is not treated as a remote model ID.
    if path.is_absolute() or model.startswith((".", "~")) or "\\" in model or ":" in model:
        raise ValueError(f"Model folder does not exist: {path}")
    return model, False


def _box(polygon: list[list[float]], size: tuple[int, int]) -> list[int]:
    width, height = size
    return [
        max(0, min(width, math.floor(min(p[0] for p in polygon)))),
        max(0, min(height, math.floor(min(p[1] for p in polygon)))),
        max(0, min(width, math.ceil(max(p[0] for p in polygon)))),
        max(0, min(height, math.ceil(max(p[1] for p in polygon)))),
    ]


def _iou(a: list[int], b: list[int]) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(1, area_a + area_b - overlap)


def geometry_regions(image: Image.Image, shape: str, min_area_ratio: float = 0.001) -> list[dict[str, Any]]:
    """Find closed, convex rectangles/triangles; collapse duplicate border contours."""
    import cv2
    if shape not in ("rectangles", "triangles"):
        raise ValueError("Geometry mode must be rectangles or triangles.")
    if not 0 < min_area_ratio <= 1:
        raise ValueError("Minimum shape area must be greater than 0 and at most 1.")
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image.width * image.height * min_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(vertices) != (4 if shape == "rectangles" else 3) or not cv2.isContourConvex(vertices):
            continue
        points = vertices.reshape(-1, 2).astype(float)
        if shape == "rectangles":
            cosines = []
            for i in range(4):
                a = points[i - 1] - points[i]
                b = points[(i + 1) % 4] - points[i]
                cosines.append(abs(float(np.dot(a, b))) / max(1e-9, float(np.linalg.norm(a) * np.linalg.norm(b))))
            if max(cosines) > 0.30:
                continue
        polygon = points.astype(int).tolist()
        box = _box(polygon, image.size)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        candidates.append((area, {"label": shape[:-1], "confidence": None, "box_xyxy": box, "polygon": polygon}))
    # Inner/outer edges of one outline frequently create multiple contours.
    regions = []
    for _, region in sorted(candidates, key=lambda item: item[0], reverse=True):
        if not any(_iou(region["box_xyxy"], other["box_xyxy"]) > 0.80 for other in regions):
            regions.append(region)
    return sorted(regions, key=lambda item: (item["box_xyxy"][1], item["box_xyxy"][0]))


class OCREngine:
    """Reuse loaded models; callers should serialize analysis on one engine."""
    def __init__(self) -> None:
        self._reader_key = None
        self._reader = None
        self._detector_key = None
        self._detector = None

    def _get_reader(self, options: AnalysisOptions) -> Any:
        key = (options.ocr_profile, tuple(options.languages), options.device)
        if key != self._reader_key:
            if options.ocr_profile == "enhanced":
                from ocr_readers import IndustrialOCRReader
                reader = IndustrialOCRReader()
            else:
                import easyocr
                reader = easyocr.Reader(list(options.languages), gpu=select_device(options.device) == "cuda")
            self._reader, self._reader_key = reader, key
        return self._reader

    def _get_detector(self, options: AnalysisOptions) -> Any:
        source, local = resolve_model_source(options.model)
        key = (source, options.device)
        if key != self._detector_key:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            device = select_device(options.device)
            processor = AutoProcessor.from_pretrained(source, local_files_only=local)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                source, use_safetensors=True, local_files_only=local,
            ).to(device).eval()

            def detect(image: Image.Image, prompt: str, threshold: float) -> list[dict[str, Any]]:
                text = prompt.strip().lower().rstrip(".") + "."
                inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs, input_ids=inputs.input_ids, threshold=threshold,
                    text_threshold=0.25, target_sizes=[(image.height, image.width)],
                )[0]
                labels = result.get("text_labels", [prompt] * len(result["boxes"]))
                return [
                    {"box": dict(zip(("xmin", "ymin", "xmax", "ymax"), box.tolist())),
                     "score": float(score), "label": str(label)}
                    for box, score, label in zip(result["boxes"], result["scores"], labels)
                ]

            self._detector, self._detector_key = detect, key
        return self._detector

    @staticmethod
    def _read(reader: Any, image: Image.Image, offset: tuple[int, int] = (0, 0)) -> list[dict[str, Any]]:
        lines = []
        for polygon, text, confidence in reader.readtext(np.asarray(image), detail=1, paragraph=False):
            if not str(text).strip():
                continue
            lines.append({
                "text": str(text), "confidence": float(confidence),
                "polygon": [[float(x) + offset[0], float(y) + offset[1]] for x, y in polygon],
            })
        return lines

    def analyze(self, image: Image.Image, options: AnalysisOptions, progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
        options.validate()
        image = image.convert("RGB")
        progress("Loading OCR model (first use may download weights)…")
        reader = self._get_reader(options)
        results = []
        diagnostics = []
        if options.mode == "all":
            progress("Reading text across the whole image…")
            for line in self._read(reader, image):
                results.append({
                    "label": "text", "confidence": line["confidence"],
                    "box_xyxy": _box(line["polygon"], image.size), "polygon": line["polygon"],
                    "text": line["text"], "ocr_lines": [line],
                })
            if getattr(reader, "last_details", None):
                diagnostics.append({"region": "whole_image", **reader.last_details})
        else:
            if options.mode == "prompt":
                progress("Loading detector and locating matching objects…")
                detector = self._get_detector(options)
                regions = []
                for detection in detector(image, options.prompt, options.threshold):
                    b = detection["box"]
                    box = _box([[b["xmin"], b["ymin"]], [b["xmax"], b["ymax"]]], image.size)
                    if box[2] > box[0] and box[3] > box[1]:
                        regions.append({"box_xyxy": box, "polygon": None,
                                        "label": detection.get("label", options.prompt),
                                        "confidence": float(detection["score"])})
            else:
                progress(f"Finding {options.mode}…")
                regions = geometry_regions(image, options.mode, options.min_area_ratio)
            for index, region in enumerate(regions, start=1):
                progress(f"Reading text in region {index} of {len(regions)}…")
                x1, y1, x2, y2 = region["box_xyxy"]
                crop = image.crop((x1, y1, x2, y2))
                if region["polygon"]:
                    # Mask corners outside a triangle/rotated rectangle before OCR.
                    mask = Image.new("L", crop.size)
                    ImageDraw.Draw(mask).polygon([(x - x1, y - y1) for x, y in region["polygon"]], fill=255)
                    crop = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)
                lines = self._read(reader, crop, offset=(x1, y1))
                if getattr(reader, "last_details", None):
                    diagnostics.append({"region": index, "crop_origin": [x1, y1],
                                        "candidate_coordinates": "relative to crop", **reader.last_details})
                results.append({**region, "text": "\n".join(line["text"] for line in lines), "ocr_lines": lines})
        for index, result in enumerate(results, start=1):
            result["id"] = index
        progress(f"Complete: {len(results)} region(s), {sum(len(r['ocr_lines']) for r in results)} text line(s).")
        return {"mode": options.mode, "prompt": options.prompt if options.mode == "prompt" else "",
                "model": options.model if options.mode == "prompt" else None,
                "threshold": options.threshold, "languages": list(options.languages),
                "min_area_ratio": options.min_area_ratio, "image_size": list(image.size), "detections": results,
                "ocr_profile": options.ocr_profile, "ocr_diagnostics": diagnostics}


def annotate_image(image: Image.Image, results: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for item in results:
        box = item["box_xyxy"]
        polygon = item.get("polygon")
        if polygon:
            points = [tuple(p) for p in polygon]
            draw.line(points + [points[0]], fill="#1769ce", width=3)
        else:
            draw.rectangle(box, outline="#1769ce", width=3)
        for line in item.get("ocr_lines", []):
            points = [tuple(p) for p in line["polygon"]]
            draw.line(points + [points[0]], fill="#008344", width=2)
        x, y = box[:2]
        # Numbered tags stay readable even for long or non-Latin OCR text.
        text = str(item["id"])
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill="#1769ce")
        draw.text((x, y), text, fill="white")
    return annotated


def extracted_text(payload: dict[str, Any]) -> str:
    return "\n\n".join(item["text"] for item in payload["detections"] if item["text"].strip())


def save_outputs(image: Image.Image, payload: dict[str, Any], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    annotate_image(image, payload["detections"]).save(directory / "annotated.png")
    (directory / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "text.txt").write_text(extracted_text(payload), encoding="utf-8")
