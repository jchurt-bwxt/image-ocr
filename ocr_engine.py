"""Shared OCR and region analysis for the desktop interface and CLI.

Model libraries are imported only when needed. One engine can be reused across
images. Geometry is detected with OpenCV, independently of the OCR choice.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
import gc
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ocr_profiles import OCR_PROFILES, PROFILE_LABELS

MODES = ("all", "rectangles", "triangles")


@dataclass(frozen=True)
class AnalysisOptions:
    mode: str = "all"
    languages: tuple[str, ...] = ("en",)
    device: str = "auto"
    min_area_ratio: float = 0.001
    ocr_profile: str = "enhanced"
    large_max_seconds: float = 120.0

    def validate(self) -> None:
        if self.ocr_profile not in OCR_PROFILES:
            raise ValueError(f"OCR profile must be one of {', '.join(OCR_PROFILES)}.")
        if self.ocr_profile != "baseline" and tuple(self.languages) != ("en",):
            raise ValueError("Enhanced and large OCR are evaluated for English markings. Choose baseline for other OCR languages.")
        if self.ocr_profile != "baseline" and self.device == "cuda":
            raise ValueError("Enhanced and large OCR use CPU inference. Choose auto/cpu, or baseline for CUDA OCR.")
        if self.mode not in MODES:
            raise ValueError(f"Choose an analysis mode from {', '.join(MODES)}.")
        if not 0 < self.min_area_ratio <= 1:
            raise ValueError("Minimum shape area must be greater than 0 and at most 1.")
        if not self.languages or any(not lang.strip() for lang in self.languages):
            raise ValueError("Enter at least one OCR language code, for example en.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda.")
        if not 0 < self.large_max_seconds <= 120:
            raise ValueError("Large-model analysis budget must be greater than 0 and at most 120 seconds.")


def load_image(path: str | Path) -> Image.Image:
    """Normalize camera orientation and composite transparent pixels onto white."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(background, image).convert("RGB")


def select_device(requested: str) -> str:
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Choose CPU or install a compatible PyTorch build.")
    if requested == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 2))
    return requested


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
    """Reuse the active model; switching models releases the previous reader."""
    def __init__(self) -> None:
        import cv2
        cv2.setNumThreads(1)
        self._reader_key = None
        self._reader = None
        self.last_timing: dict[str, Any] = {}

    def close(self) -> None:
        reader = self._reader
        self._reader, self._reader_key = None, None
        if reader is not None and callable(getattr(reader, "close", None)):
            reader.close()
        del reader
        gc.collect()

    def _get_reader(self, options: AnalysisOptions) -> Any:
        key = (options.ocr_profile, tuple(options.languages), options.device)
        if key != self._reader_key:
            self.close()
            if options.ocr_profile == "large":
                from large_ocr import LargeOCRReader
                reader = LargeOCRReader()
            elif options.ocr_profile == "enhanced":
                from ocr_readers import IndustrialOCRReader
                reader = IndustrialOCRReader()
            else:
                import easyocr
                reader = easyocr.Reader(list(options.languages), gpu=select_device(options.device) == "cuda")
            self._reader, self._reader_key = reader, key
        return self._reader

    @staticmethod
    def _read(reader: Any, image: Image.Image, offset: tuple[int, int] = (0, 0), deadline: float | None = None) -> list[dict[str, Any]]:
        kwargs = {"detail": 1, "paragraph": False}
        if deadline is not None and getattr(reader, "supports_deadline", False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Large OCR image analysis reached its time budget.")
            kwargs["timeout"] = remaining
        lines = []
        for polygon, text, confidence in reader.readtext(np.asarray(image), **kwargs):
            if not str(text).strip():
                continue
            lines.append({
                "text": str(text), "confidence": float(confidence) if confidence is not None else None,
                "polygon": [[float(x) + offset[0], float(y) + offset[1]] for x, y in polygon],
            })
        return lines

    def analyze(self, image: Image.Image, options: AnalysisOptions, progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
        started = time.perf_counter()
        self.last_timing = {"model_setup_seconds": 0.0, "analysis_seconds": 0.0,
                            "total_seconds": 0.0, "model_reused": False}
        analysis_started = None
        try:
            options.validate()
            image = image.convert("RGB")
            self.last_timing["model_reused"] = self._reader_key == (options.ocr_profile, tuple(options.languages), options.device)
            progress("Loading OCR model; first use may download weights. Large-model startup is timed separately…")
            reader = self._get_reader(options)
            self.last_timing["model_setup_seconds"] = time.perf_counter() - started
            analysis_started = time.perf_counter()
            deadline = time.monotonic() + options.large_max_seconds if options.ocr_profile == "large" else None
            results, diagnostics = [], []
            if options.mode == "all":
                progress("Reading text across the whole image…")
                for line in self._read(reader, image, deadline=deadline):
                    results.append({"label": "text", "confidence": line["confidence"],
                                    "box_xyxy": _box(line["polygon"], image.size), "polygon": line["polygon"],
                                    "text": line["text"], "ocr_lines": [line]})
                if getattr(reader, "last_details", None):
                    diagnostics.append({"region": "whole_image", **reader.last_details})
            else:
                progress(f"Finding {options.mode}…")
                regions = geometry_regions(image, options.mode, options.min_area_ratio)
                for index, region in enumerate(regions, start=1):
                    progress(f"Reading text in region {index} of {len(regions)}…")
                    x1, y1, x2, y2 = region["box_xyxy"]
                    crop = image.crop((x1, y1, x2, y2))
                    mask = Image.new("L", crop.size)
                    ImageDraw.Draw(mask).polygon([(x - x1, y - y1) for x, y in region["polygon"]], fill=255)
                    crop = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)
                    lines = self._read(reader, crop, offset=(x1, y1), deadline=deadline)
                    if getattr(reader, "last_details", None):
                        diagnostics.append({"region": index, "crop_origin": [x1, y1],
                                            "candidate_coordinates": "relative to crop", **reader.last_details})
                    results.append({**region, "text": "\n".join(line["text"] for line in lines), "ocr_lines": lines})
            for index, result in enumerate(results, start=1):
                result["id"] = index
            progress(f"Complete: {len(results)} region(s), {sum(len(r['ocr_lines']) for r in results)} text line(s).")
            return {"mode": options.mode, "languages": list(options.languages), "requested_device": options.device,
                    "min_area_ratio": options.min_area_ratio, "image_size": list(image.size), "detections": results,
                    "ocr_profile": options.ocr_profile, "ocr_model": PROFILE_LABELS[options.ocr_profile],
                    "ocr_diagnostics": diagnostics, "timing": self.last_timing,
                    "warnings": ["Large OCR reached its output-token limit; text may be incomplete."]
                    if any(d.get("token_limit_reached") for d in diagnostics) else []}
        except (TimeoutError, RuntimeError):
            if options.ocr_profile == "large":
                self.close()
            raise
        finally:
            ended = time.perf_counter()
            self.last_timing["total_seconds"] = ended - started
            if analysis_started is None:
                self.last_timing["model_setup_seconds"] = ended - started
            else:
                self.last_timing["analysis_seconds"] = ended - analysis_started


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
