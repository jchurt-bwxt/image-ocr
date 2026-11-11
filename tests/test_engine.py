"""Analysis, geometry, and export regressions without neural-network weights."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from ocr_engine import AnalysisOptions, OCREngine, extracted_text, geometry_regions, load_image, save_outputs


class FakeReader:
    def __init__(self, confidence=0.97):
        self.inputs = []
        self.kwargs = []
        self.confidence = confidence
        self.close = Mock()

    def readtext(self, image, **kwargs):
        self.inputs.append(np.asarray(image).copy())
        self.kwargs.append(kwargs)
        return [([[1, 2], [8, 2], [8, 7], [1, 7]], "Café 42", self.confidence)]


class AnalysisTests(unittest.TestCase):
    def test_whole_image_reads_all_text_without_a_detector(self):
        engine = OCREngine()
        self.assertFalse(hasattr(engine, "_get_detector"))
        self.assertFalse(hasattr(engine, "_detector"))
        reader = FakeReader()
        image = Image.new("RGB", (90, 60), "white")
        with patch.object(engine, "_get_reader", return_value=reader):
            result = engine.analyze(image, AnalysisOptions(mode="all"))
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["detections"][0]["text"], "Café 42")
        self.assertEqual(reader.inputs[0].shape[:2], (60, 90))
        self.assertEqual(extracted_text(result).strip(), "Café 42")

    def test_removed_prompt_mode_is_rejected_before_loading_a_reader(self):
        engine = OCREngine()
        with patch.object(engine, "_get_reader", side_effect=AssertionError("No model should load")):
            with self.assertRaises(ValueError):
                engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions(mode="prompt"))

    def test_geometry_masks_outside_polygon_and_offsets_ocr_coordinates(self):
        engine = OCREngine()
        reader = FakeReader()
        region = {"label": "triangle", "confidence": None, "box_xyxy": [10, 20, 40, 50],
                  "polygon": [[10, 20], [40, 20], [10, 50]]}
        image = Image.new("RGB", (70, 70), "black")
        with patch("ocr_engine.geometry_regions", return_value=[region]), patch.object(engine, "_get_reader", return_value=reader):
            result = engine.analyze(image, AnalysisOptions(mode="triangles"))
        self.assertTrue(np.all(reader.inputs[0][28, 28] == 255))
        self.assertTrue(np.all(reader.inputs[0][2, 2] == 0))
        line = result["detections"][0]["ocr_lines"][0]
        self.assertEqual(line["polygon"][0], [11, 22])
        self.assertIsNone(result["detections"][0]["confidence"])

    def test_large_model_confidence_is_not_invented(self):
        engine = OCREngine()
        with patch.object(engine, "_get_reader", return_value=FakeReader(confidence=None)):
            result = engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions(ocr_profile="large"))
        self.assertIsNone(result["detections"][0]["confidence"])
        self.assertIsNone(result["detections"][0]["ocr_lines"][0]["confidence"])

    def test_large_geometry_regions_share_one_image_budget(self):
        engine = OCREngine()
        reader = FakeReader(confidence=None)
        reader.supports_deadline = True
        region = {"label": "rectangle", "confidence": None, "box_xyxy": [0, 0, 10, 10],
                  "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        with patch.object(engine, "_get_reader", return_value=reader), patch(
            "ocr_engine.geometry_regions", return_value=[region, region]
        ), patch("ocr_engine.time.monotonic", side_effect=[100.0, 101.0, 105.0]):
            engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions(mode="rectangles", ocr_profile="large", large_max_seconds=10))
        self.assertEqual([kwargs["timeout"] for kwargs in reader.kwargs], [9.0, 5.0])

    def test_large_timeout_releases_the_reader_and_records_timing(self):
        engine = OCREngine()
        reader = FakeReader()
        reader.readtext = Mock(side_effect=TimeoutError("budget exceeded"))
        engine._reader = reader
        engine._reader_key = ("large", ("en",), "auto")
        with patch.object(engine, "_get_reader", return_value=reader):
            with self.assertRaises(TimeoutError):
                engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions(ocr_profile="large"))
        reader.close.assert_called_once_with()
        self.assertIsNone(engine._reader)
        self.assertGreaterEqual(engine.last_timing["total_seconds"], engine.last_timing["analysis_seconds"])

    def test_reader_setup_reuse_and_processing_times_are_distinct(self):
        engine = OCREngine()
        reader = FakeReader()
        factory = Mock(return_value=reader)
        times = [10.0, 12.0, 12.0, 15.0, 20.0, 20.1, 20.1, 22.1]
        with patch.dict(sys.modules, {"ocr_readers": SimpleNamespace(IndustrialOCRReader=factory)}), patch(
            "ocr_engine.time.perf_counter", side_effect=times
        ):
            cold = engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions())
            warm = engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions())
        factory.assert_called_once_with()
        self.assertFalse(cold["timing"]["model_reused"])
        self.assertEqual(cold["timing"]["model_setup_seconds"], 2.0)
        self.assertEqual(cold["timing"]["analysis_seconds"], 3.0)
        self.assertEqual(cold["timing"]["total_seconds"], 5.0)
        self.assertTrue(warm["timing"]["model_reused"])
        self.assertAlmostEqual(warm["timing"]["model_setup_seconds"], 0.1)
        self.assertAlmostEqual(warm["timing"]["analysis_seconds"], 2.0)
        self.assertAlmostEqual(warm["timing"]["total_seconds"], 2.1)
        engine.close()
        reader.close.assert_called_once_with()

    def test_switching_ocr_profiles_closes_previous_reader(self):
        engine = OCREngine()
        enhanced, large = FakeReader(), FakeReader(confidence=None)
        with patch.dict(sys.modules, {
            "ocr_readers": SimpleNamespace(IndustrialOCRReader=Mock(return_value=enhanced)),
            "large_ocr": SimpleNamespace(LargeOCRReader=Mock(return_value=large)),
        }):
            engine._get_reader(AnalysisOptions())
            engine._get_reader(AnalysisOptions(ocr_profile="large"))
        enhanced.close.assert_called_once_with()
        self.assertIs(engine._reader, large)
        engine.close()
        large.close.assert_called_once_with()

    def test_image_without_text_returns_empty_result(self):
        engine = OCREngine()
        reader = FakeReader()
        reader.readtext = lambda image, **kwargs: []
        with patch.object(engine, "_get_reader", return_value=reader):
            result = engine.analyze(Image.new("RGB", (50, 50), "white"), AnalysisOptions())
        self.assertEqual(result["detections"], [])
        self.assertEqual(extracted_text(result), "")

    def test_invalid_large_budget_is_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            OCREngine().analyze(Image.new("RGB", (20, 20)), AnalysisOptions(large_max_seconds=121))


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


if __name__ == "__main__":
    unittest.main()
