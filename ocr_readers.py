"""Pretrained OCR for faint industrial markings, with label-free pass selection."""
from __future__ import annotations

import math
import os
from pathlib import Path
import unicodedata
from typing import Any

import cv2
import numpy as np
from PIL import Image


MODEL_ID = "PP-OCRv6-medium (RapidOCR 3.10.0 / ONNX Runtime CPU)"
PREPROCESSING = ("original", "clahe", "blackhat")


def model_cache_directory() -> Path:
    override = os.environ.get("IMAGE_OCR_MODEL_CACHE")
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "ImageOCR" / "models"
    return Path.home() / ".cache" / "image-ocr" / "models"


def image_variants(image: Image.Image):
    """Keep image dimensions unchanged so every polygon stays in source pixels."""
    image = image.convert("RGB")
    yield "original", image
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield "clahe", Image.fromarray(clahe).convert("RGB")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    flattened = 255 - cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    yield "blackhat", Image.fromarray(flattened).convert("RGB")


def rank_pass(lines: list[tuple]) -> float:
    """A fixed confidence/length heuristic; it never reads expected labels.

    Sqrt length reduces the incentive to select a long hallucinated string,
    while retaining short identifiers. Scores are not calibrated probabilities.
    """
    total = 0.0
    for _polygon, text, confidence in lines:
        text = unicodedata.normalize("NFKC", str(text)).casefold()
        count = sum(char.isalnum() for char in text)
        if math.isfinite(float(confidence)):
            total += math.sqrt(count) * max(0.0, min(1.0, float(confidence)))
    return total


class IndustrialOCRReader:
    """EasyOCR-compatible adapter around a pinned PP-OCRv6 medium pipeline.

    Each pass is one complete OCR prediction. Select a whole pass to retain its
    reading order and avoid double-counting overlapping detections. All raw
    alternatives are retained in last_details for human review/export.
    """
    def __init__(self) -> None:
        try:
            import onnxruntime
            # ONNX Runtime enables telemetry by default. OCR images and analysis
            # belong on the local machine; disable telemetry before any session.
            onnxruntime.disable_telemetry_events()
            from rapidocr import RapidOCR, OCRVersion, ModelType, LangRec, EngineType
        except ImportError as exc:
            raise ImportError("Enhanced OCR requires rapidocr and onnxruntime. Install requirements.txt.") from exc
        cache = model_cache_directory()
        cache.mkdir(parents=True, exist_ok=True)
        self.engine = RapidOCR(params={
            "Global.model_root_dir": str(cache),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.MEDIUM,
            "Det.lang_type": "ch",
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType.MEDIUM,
            "Rec.lang_type": LangRec.CH,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Cls.ocr_version": OCRVersion.PPOCRV4,
            "Cls.model_type": ModelType.MOBILE,
        })
        self.last_details: dict[str, Any] = {}

    def readtext(self, image: np.ndarray, **_kwargs) -> list[tuple]:
        variants = []
        for name, variant in image_variants(Image.fromarray(image)):
            # A PIL image makes RGB/BGR conversion explicit in RapidOCR.
            result = self.engine(variant)
            boxes = result.boxes if result.boxes is not None else []
            texts = result.txts if result.txts is not None else []
            scores = result.scores if result.scores is not None else []
            lines = [(np.asarray(box).tolist(), str(text), float(score))
                     for box, text, score in zip(boxes, texts, scores)
                     if str(text).strip()]
            variants.append((name, lines, rank_pass(lines)))
        # max is stable: an equal score prefers the original image.
        name, lines, score = max(variants, key=lambda item: item[2])
        self.last_details = {
            "model": MODEL_ID, "selected_preprocessing": name, "selection_score": score,
            "candidates": [{"preprocessing": prep, "selection_score": rank,
                            "lines": [{"text": text, "confidence": conf, "polygon": polygon}
                                      for polygon, text, conf in candidate]}
                           for prep, candidate, rank in variants],
        }
        return lines
