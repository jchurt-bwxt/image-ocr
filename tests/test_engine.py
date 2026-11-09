"""Behavior checks that do not load or download neural-network weights."""

import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from ocr_engine import (
    AnalysisOptions,
    OCREngine,
    extracted_text,
    geometry_regions,
    load_image,
    resolve_model_source,
    save_outputs,
)


class FakeReader:
    def __init__(self):
        self.inputs = []

    def readtext(self, image, **kwargs):
        self.inputs.append(np.asarray(image).copy())
        return [([[1, 2], [8, 2], [8, 7], [1, 7]], "Café 42", 0.97)]


class AnalysisTests(unittest.TestCase):
    def test_whole_image_ocr_does_not_load_object_detector(self):
        engine = OCREngine()
        reader = FakeReader()
        image = Image.new("RGB", (90, 60), "white")
        with patch.object(engine, "_get_reader", return_value=reader), patch.object(
            engine, "_get_detector", side_effect=AssertionError("Detector must stay unloaded")
        ):
            result = engine.analyze(image, AnalysisOptions(mode="all"))
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["detections"][0]["text"], "Café 42")
        self.assertEqual(reader.inputs[0].shape[:2], (60, 90))
        self.assertEqual(extracted_text(result).strip(), "Café 42")

    def test_geometry_masks_outside_polygon_and_offsets_ocr_coordinates(self):
        engine = OCREngine()
        reader = FakeReader()
        region = {
            "label": "triangle",
            "confidence": None,
            "box_xyxy": [10, 20, 40, 50],
            "polygon": [[10, 20], [40, 20], [10, 50]],
        }
        image = Image.new("RGB", (70, 70), "black")
        with patch("ocr_engine.geometry_regions", return_value=[region]), patch.object(
            engine, "_get_reader", return_value=reader
        ), patch.object(engine, "_get_detector", side_effect=AssertionError("Unneeded detector")):
            result = engine.analyze(image, AnalysisOptions(mode="triangles"))
        self.assertTrue(np.all(reader.inputs[0][28, 28] == 255))
        self.assertTrue(np.all(reader.inputs[0][2, 2] == 0))
        line = result["detections"][0]["ocr_lines"][0]
        self.assertEqual(line["polygon"][0], [11, 22])
        self.assertEqual(result["detections"][0]["confidence"], None)

    def test_prompt_boxes_are_clipped_to_image(self):
        engine = OCREngine()
        reader = FakeReader()
        detector = lambda *args: [
            {"label": "sign", "score": 0.9, "box": {"xmin": -7, "ymin": -9, "xmax": 100, "ymax": 200}}
        ]
        with patch.object(engine, "_get_reader", return_value=reader), patch.object(
            engine, "_get_detector", return_value=detector
        ):
            result = engine.analyze(Image.new("RGB", (50, 40)), AnalysisOptions(mode="prompt", prompt="sign"))
        self.assertEqual(result["detections"][0]["box_xyxy"], [0, 0, 50, 40])
        self.assertEqual(reader.inputs[0].shape[:2], (40, 50))

    def test_prompt_without_description_is_rejected(self):
        with self.assertRaises(ValueError):
            OCREngine().analyze(Image.new("RGB", (20, 20)), AnalysisOptions(mode="prompt", prompt="  "))

    def test_image_without_text_returns_empty_result(self):
        engine = OCREngine()
        reader = FakeReader()
        reader.readtext = lambda image, **kwargs: []
        with patch.object(engine, "_get_reader", return_value=reader):
            result = engine.analyze(Image.new("RGB", (50, 50), "white"), AnalysisOptions())
        self.assertEqual(result["detections"], [])
        self.assertEqual(extracted_text(result), "")

    def test_invalid_threshold_is_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            OCREngine().analyze(Image.new("RGB", (20, 20)), AnalysisOptions(threshold=1.2))


class GeometryTests(unittest.TestCase):
    @staticmethod
    def shapes_image():
        image = Image.new("RGB", (400, 220), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 40, 160, 150), outline="black", width=4)
        draw.polygon(((270, 25), (200, 180), (350, 180)), outline="black", width=4)
        return image

    def test_rectangle_search_finds_rectangle_without_triangle(self):
        regions = geometry_regions(self.shapes_image(), "rectangles", 0.01)
        self.assertEqual(len(regions), 1)
        x1, y1, x2, y2 = regions[0]["box_xyxy"]
        self.assertLessEqual(x1, 35)
        self.assertGreaterEqual(x2, 155)
        self.assertEqual(len(regions[0]["polygon"]), 4)

    def test_triangle_search_finds_triangle_without_rectangle(self):
        regions = geometry_regions(self.shapes_image(), "triangles", 0.01)
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0]["polygon"]), 3)

    def test_rotated_rectangle_is_supported(self):
        image = Image.new("RGB", (190, 190), "white")
        ImageDraw.Draw(image).polygon(((70, 30), (150, 90), (110, 145), (30, 85)), outline="black", width=3)
        self.assertEqual(len(geometry_regions(image, "rectangles", 0.01)), 1)

    def test_small_contours_are_filtered(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 30, 30), outline="black", width=2)
        self.assertEqual(geometry_regions(image, "rectangles", 0.02), [])


class FileTests(unittest.TestCase):
    def test_model_id_is_preserved(self):
        self.assertEqual(resolve_model_source("IDEA-Research/grounding-dino-tiny"), ("IDEA-Research/grounding-dino-tiny", False))

    def test_local_weights_without_configuration_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaises(ValueError):
                resolve_model_source(directory)

    def test_valid_local_snapshot_resolves_to_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors").write_bytes(b"placeholder")
            (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
            source, is_local = resolve_model_source(directory)
            self.assertTrue(is_local)
            self.assertEqual(Path(source), Path(directory).resolve())

    def test_image_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (80, 40), "white")
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            loaded = load_image(path)
            self.assertEqual(loaded.size, (40, 80))

    def test_export_writes_image_json_and_unicode_text(self):
        payload = {
            "mode": "all",
            "detections": [{
                "id": 1, "label": "text", "confidence": 0.97,
                "box_xyxy": [1, 2, 30, 20], "polygon": [[1, 2], [30, 2], [30, 20], [1, 20]],
                "text": "Café 42", "ocr_lines": [],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            save_outputs(Image.new("RGB", (60, 40), "white"), payload, directory)
            output = Path(directory)
            self.assertEqual(json.loads((output / "results.json").read_text(encoding="utf-8"))["detections"][0]["text"], "Café 42")
            self.assertEqual((output / "text.txt").read_text(encoding="utf-8").strip(), "Café 42")
            with Image.open(output / "annotated.png") as annotated:
                self.assertEqual(annotated.size, (60, 40))


class ModelLoadingTests(unittest.TestCase):
    def test_local_detector_uses_safetensors_and_reuses_loaded_model(self):
        processor = Mock()
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        processor_loader = Mock(return_value=processor)
        model_loader = Mock(return_value=model)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            AutoModelForZeroShotObjectDetection=SimpleNamespace(from_pretrained=model_loader),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve()
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").write_bytes(b"placeholder")
            engine = OCREngine()
            options = AnalysisOptions(mode="prompt", prompt="sign", model=directory, device="cpu")
            with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
                first_detector = engine._get_detector(options)
                same_model_new_prompt = AnalysisOptions(mode="prompt", prompt="box", threshold=0.5, model=directory, device="cpu")
                self.assertIs(first_detector, engine._get_detector(same_model_new_prompt))
            processor_loader.assert_called_once_with(str(source), local_files_only=True)
            model_loader.assert_called_once_with(str(source), use_safetensors=True, local_files_only=True)
            model.to.assert_called_once_with("cpu")
            model.eval.assert_called_once_with()
            # A subsequent whole-image run should not enter the detector loader.
            with patch.object(engine, "_get_reader", return_value=FakeReader()), patch.object(
                engine, "_get_detector", side_effect=AssertionError("Whole-image OCR must skip detection")
            ):
                result = engine.analyze(Image.new("RGB", (50, 50)), AnalysisOptions(mode="all"))
            self.assertEqual(result["detections"][0]["text"], "Café 42")


if __name__ == "__main__":
    unittest.main()
