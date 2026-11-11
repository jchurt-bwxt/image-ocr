"""Enhanced OCR adapter regressions without loading or downloading a model."""

import builtins
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ocr_engine import AnalysisOptions, OCREngine
from ocr_readers import IndustrialOCRReader, PREPROCESSING, image_variants, rank_pass


POLYGON = [[1.25, 2.5], [18.5, 2.5], [18.5, 13.75], [1.25, 13.75]]


def response(text=None, confidence=None, polygon=POLYGON):
    return SimpleNamespace(
        boxes=None if text is None else np.asarray([polygon]),
        txts=None if text is None else [text],
        scores=None if confidence is None else [confidence],
    )


class PreprocessingTests(unittest.TestCase):
    def test_all_variants_keep_source_dimensions_and_rgb(self):
        for mode in ("RGB", "L", "RGBA"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (77, 43))
                variants = list(image_variants(image))
                self.assertEqual([name for name, _ in variants], list(PREPROCESSING))
                for _name, variant in variants:
                    self.assertEqual(variant.size, image.size)
                    self.assertEqual(variant.mode, "RGB")

    def test_original_pixels_are_retained(self):
        pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        original = list(image_variants(Image.fromarray(pixels)))[0][1]
        np.testing.assert_array_equal(np.asarray(original), pixels)


class RankingTests(unittest.TestCase):
    def test_rank_uses_text_length_and_confidence_only(self):
        line = (POLYGON, "ＡＢ-１２", 0.8)
        moved = ([[80, 90], [90, 90], [90, 99], [80, 99]], "ab 12", 0.8)
        self.assertAlmostEqual(rank_pass([line]), 1.6)
        self.assertEqual(rank_pass([line]), rank_pass([moved]))
        self.assertGreater(rank_pass([(POLYGON, "AB12", 0.9)]), rank_pass([line]))
        self.assertGreater(rank_pass([(POLYGON, "AB123456", 0.8)]), rank_pass([line]))

    def test_rank_handles_nonfinite_confidence_and_punctuation(self):
        self.assertEqual(rank_pass([(POLYGON, "---# /", 0.99)]), 0)
        self.assertEqual(rank_pass([(POLYGON, "AB12", math.nan)]), 0)
        self.assertEqual(rank_pass([(POLYGON, "AB12", math.inf)]), 0)
        self.assertEqual(rank_pass([(POLYGON, "AB12", -0.2)]), 0)
        self.assertEqual(rank_pass([(POLYGON, "AB12", 1.5)]), 2)


class AdapterTests(unittest.TestCase):
    def reader(self, outputs):
        reader = IndustrialOCRReader.__new__(IndustrialOCRReader)
        reader.engine = Mock(side_effect=outputs)
        reader.last_details = {}
        return reader

    def test_best_whole_pass_keeps_raw_text_and_source_coordinates(self):
        raw_winner = 'pl-53I2? 2 1/2" ht#'
        reader = self.reader([
            response("raw uncorrected", 0.1),
            response(raw_winner, 0.95),
            response("other pass", 0.15),
        ])
        result = reader.readtext(np.full((32, 80, 3), 220, dtype=np.uint8))
        self.assertEqual(result, [(POLYGON, raw_winner, 0.95)])
        self.assertEqual(reader.engine.call_count, 3)
        for call in reader.engine.call_args_list:
            variant = call.args[0]
            self.assertIsInstance(variant, Image.Image)
            self.assertEqual(variant.size, (80, 32))
        details = reader.last_details
        self.assertEqual(details["selected_preprocessing"], "clahe")
        self.assertEqual([candidate["preprocessing"] for candidate in details["candidates"]], list(PREPROCESSING))
        self.assertEqual(
            [candidate["lines"][0]["text"] for candidate in details["candidates"]],
            ["raw uncorrected", raw_winner, "other pass"],
        )
        self.assertEqual(details["candidates"][1]["lines"][0]["polygon"], POLYGON)

    def test_equal_rank_stably_favors_original(self):
        reader = self.reader([response("A1", 0.9), response("B2", 0.9), response("C3", 0.9)])
        result = reader.readtext(np.zeros((30, 60, 3), dtype=np.uint8))
        self.assertEqual(result[0][1], "A1")
        self.assertEqual(reader.last_details["selected_preprocessing"], "original")

    def test_empty_model_outputs_return_empty_lines_and_details(self):
        reader = self.reader([response(), response(), response()])
        self.assertEqual(reader.readtext(np.zeros((30, 60, 3), dtype=np.uint8)), [])
        self.assertEqual(reader.last_details["selected_preprocessing"], "original")
        self.assertEqual(reader.last_details["selection_score"], 0)
        self.assertEqual(len(reader.last_details["candidates"]), 3)
        self.assertTrue(all(candidate["lines"] == [] for candidate in reader.last_details["candidates"]))


class ReaderLoadingTests(unittest.TestCase):
    def test_telemetry_is_disabled_before_rapidocr_sessions_are_created(self):
        order = []
        disable_telemetry = Mock(side_effect=lambda: order.append("telemetry_disabled"))
        engine = Mock()

        def construct(**_kwargs):
            self.assertEqual(order, ["telemetry_disabled"])
            order.append("rapidocr_constructed")
            return engine

        factory = Mock(side_effect=construct)
        fake_rapidocr = SimpleNamespace(
            RapidOCR=factory,
            OCRVersion=SimpleNamespace(PPOCRV6="v6", PPOCRV4="v4"),
            ModelType=SimpleNamespace(MEDIUM="medium", MOBILE="mobile"),
            LangRec=SimpleNamespace(CH="ch"),
            EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            paths = {stage: cache / f"verified-{stage}.onnx" for stage in ("det", "rec", "cls")}
            with patch.dict(sys.modules, {
                "onnxruntime": SimpleNamespace(disable_telemetry_events=disable_telemetry),
                "rapidocr": fake_rapidocr,
            }), patch("ocr_readers.model_cache_directory", return_value=cache), patch(
                "ocr_readers.get_enhanced_model_paths", return_value=paths
            ) as model_paths:
                reader = IndustrialOCRReader()
            model_paths.assert_called_once_with(cache)
        self.assertIs(reader.engine, engine)
        self.assertEqual(order, ["telemetry_disabled", "rapidocr_constructed"])
        disable_telemetry.assert_called_once_with()
        self.assertFalse(factory.call_args.kwargs["params"]["EngineConfig.onnxruntime.use_cuda"])
        for stage, key in (("det", "Det"), ("rec", "Rec"), ("cls", "Cls")):
            self.assertEqual(factory.call_args.kwargs["params"][f"{key}.model_path"], str(paths[stage]))

    def test_enhanced_does_not_import_easyocr_or_torch(self):
        enhanced = Mock()
        factory = Mock(return_value=enhanced)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in ("easyocr", "torch"):
                raise AssertionError(f"Enhanced OCR unexpectedly imported {name}")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"ocr_readers": SimpleNamespace(IndustrialOCRReader=factory)}), patch(
            "builtins.__import__", side_effect=guarded_import
        ):
            engine = OCREngine()
            options = AnalysisOptions(ocr_profile="enhanced", device="cpu")
            self.assertIs(engine._get_reader(options), enhanced)
            self.assertIs(engine._get_reader(options), enhanced)
        factory.assert_called_once_with()

    def test_profile_switch_replaces_reader_and_each_profile_reuses_it(self):
        enhanced_readers = [Mock(), Mock()]
        enhanced_factory = Mock(side_effect=enhanced_readers)
        baseline_reader = Mock()
        baseline_factory = Mock(return_value=baseline_reader)
        with patch.dict(sys.modules, {
            "ocr_readers": SimpleNamespace(IndustrialOCRReader=enhanced_factory),
            "easyocr": SimpleNamespace(Reader=baseline_factory),
        }), patch("ocr_engine.select_device", return_value="cpu"):
            engine = OCREngine()
            enhanced = AnalysisOptions(ocr_profile="enhanced", device="cpu")
            baseline = AnalysisOptions(ocr_profile="baseline", device="cpu")
            self.assertIs(engine._get_reader(enhanced), enhanced_readers[0])
            self.assertIs(engine._get_reader(enhanced), enhanced_readers[0])
            self.assertIs(engine._get_reader(baseline), baseline_reader)
            self.assertIs(engine._get_reader(baseline), baseline_reader)
            self.assertIs(engine._get_reader(enhanced), enhanced_readers[1])
        self.assertEqual(enhanced_factory.call_count, 2)
        baseline_factory.assert_called_once_with(["en"], gpu=False)

    def test_enhanced_rejects_unsupported_language_before_model_loading(self):
        engine = OCREngine()
        with patch.object(engine, "_get_reader") as loader:
            with self.assertRaisesRegex(ValueError, "English.*baseline"):
                engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions(languages=("en", "fr")))
        loader.assert_not_called()

    def test_enhanced_rejects_cuda_with_clear_cpu_explanation(self):
        engine = OCREngine()
        with patch.object(engine, "_get_reader") as loader:
            with self.assertRaisesRegex(ValueError, "CPU inference"):
                engine.analyze(Image.new("RGB", (20, 20)), AnalysisOptions(device="cuda"))
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
