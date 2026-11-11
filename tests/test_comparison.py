"""Independent model failures, timing presentation, and comparison exports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ocr_comparison import compare_image, run_model, save_comparison
from ocr_engine import AnalysisOptions
from ocr_profiles import OCR_PROFILES


def payload(text="Café 42", confidence=0.9, reused=False):
    return {
        "mode": "all", "image_size": [40, 30],
        "detections": [{"id": 1, "label": "text", "text": text, "confidence": confidence,
                        "box_xyxy": [1, 2, 20, 15], "polygon": [[1, 2], [20, 2], [20, 15], [1, 15]],
                        "ocr_lines": []}],
        "timing": {"model_setup_seconds": 1.25, "analysis_seconds": 2.75,
                   "total_seconds": 4.0, "model_reused": reused},
    }


class ComparisonTests(unittest.TestCase):
    def test_comparison_rejects_mismatched_hardware_or_languages_before_inference(self):
        engine = Mock()
        for options in (AnalysisOptions(ocr_profile="baseline", device="cuda"),
                        AnalysisOptions(ocr_profile="baseline", languages=("fr",))):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "English and CPU"):
                compare_image(Image.new("RGB", (10, 10)), options, engine, progress=lambda _: None)
        engine.analyze.assert_not_called()

    def test_one_model_failure_does_not_prevent_other_results(self):
        engine = Mock()
        engine.last_timing = {"model_setup_seconds": 0.5, "analysis_seconds": 1.5, "model_reused": False}
        seen = []
        devices = []

        def analyze(image, options, **kwargs):
            seen.append(options.ocr_profile)
            devices.append(options.device)
            if options.ocr_profile == "enhanced":
                raise RuntimeError("model download failed")
            return payload(options.ocr_profile, None if options.ocr_profile == "large" else 0.9)

        engine.analyze.side_effect = analyze
        report = compare_image(Image.new("RGB", (40, 30)), AnalysisOptions(), engine, progress=lambda _: None)
        self.assertEqual(tuple(seen), OCR_PROFILES)
        self.assertEqual(devices, ["cpu", "cpu", "cpu"])
        self.assertEqual([row["status"] for row in report["models"]], ["ok", "error", "ok"])
        self.assertEqual(report["models"][0]["text"], "baseline")
        self.assertIn("model download failed", report["models"][1]["error"])
        self.assertIsNone(report["models"][1]["payload"])
        self.assertEqual(report["models"][2]["text"], "large")
        self.assertIsNone(report["models"][2]["payload"]["detections"][0]["confidence"])
        engine.close.assert_not_called()  # The caller owns this reusable engine.

    def test_large_timeout_is_distinguished_from_general_error(self):
        engine = Mock()
        engine.analyze.side_effect = TimeoutError("worker terminated after deadline")
        engine.last_timing = {"model_setup_seconds": 3.0, "analysis_seconds": 120.0, "model_reused": False}
        with patch("ocr_comparison.time.perf_counter", side_effect=[10.0, 133.0]):
            row = run_model(Image.new("RGB", (10, 10)), AnalysisOptions(ocr_profile="large"), engine)
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["total_seconds"], 123.0)
        self.assertEqual(row["model_setup_seconds"], 3.0)
        self.assertEqual(row["analysis_seconds"], 120.0)
        self.assertEqual(row["text"], "")

    def test_timing_row_keeps_cold_start_processing_and_reuse_separate(self):
        engine = Mock()
        engine.analyze.return_value = payload(reused=True)
        with patch("ocr_comparison.time.perf_counter", side_effect=[10.0, 14.0]):
            row = run_model(Image.new("RGB", (40, 30)), AnalysisOptions(), engine)
        self.assertEqual(row["total_seconds"], 4.0)
        self.assertEqual(row["model_setup_seconds"], 1.25)
        self.assertEqual(row["analysis_seconds"], 2.75)
        self.assertTrue(row["model_reused"])

    def test_comparison_closes_an_engine_it_created(self):
        engine = Mock()
        engine.analyze.return_value = payload()
        with patch("ocr_comparison.OCREngine", return_value=engine):
            report = compare_image(Image.new("RGB", (40, 30)), AnalysisOptions(), progress=lambda _: None)
        engine.close.assert_called_once_with()
        self.assertEqual(len(report["models"]), 3)
        self.assertEqual(report["large_inference_budget_seconds"], 120)
        self.assertIn("model setup/download", report["timing_scope"])

    def test_export_saves_each_successful_profile_and_failure_metadata(self):
        baseline, large = payload("small result"), payload("larger result", confidence=None)
        report = {"models": [
            {"profile": "baseline", "status": "ok", "payload": baseline},
            {"profile": "enhanced", "status": "error", "payload": None, "error": "offline"},
            {"profile": "large", "status": "ok", "payload": large},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            save_comparison(Image.new("RGB", (40, 30)), report, directory)
            root = Path(directory)
            stored = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["models"][1]["error"], "offline")
            self.assertFalse((root / "enhanced").exists())
            for profile, text in (("baseline", "small result"), ("large", "larger result")):
                self.assertTrue((root / profile / "annotated.png").is_file())
                self.assertTrue((root / profile / "results.json").is_file())
                self.assertEqual((root / profile / "text.txt").read_text(encoding="utf-8"), text)
            larger = json.loads((root / "large" / "results.json").read_text(encoding="utf-8"))
            self.assertIsNone(larger["detections"][0]["confidence"])

    def test_overwrite_removes_stale_successes_for_failed_and_skipped_models(self):
        image = Image.new("RGB", (40, 30))
        initial = {"models": [{"profile": profile, "status": "ok", "payload": payload("old " + profile)}
                              for profile in OCR_PROFILES]}
        updated = {"models": [
            {"profile": "baseline", "status": "ok", "payload": payload("new baseline")},
            {"profile": "enhanced", "status": "error", "payload": None, "error": "offline"},
            # Large is absent because this comparison was stopped before it ran.
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_comparison(image, initial, root)
            notes = root / "large" / "user_notes.txt"
            notes.write_text("keep my notes", encoding="utf-8")
            save_comparison(image, updated, root)
            for profile in ("enhanced", "large"):
                for name in ("annotated.png", "results.json", "text.txt"):
                    self.assertFalse((root / profile / name).exists(), f"Stale {profile}/{name} remained")
            self.assertEqual(notes.read_text(encoding="utf-8"), "keep my notes")
            self.assertEqual((root / "baseline" / "text.txt").read_text(encoding="utf-8"), "new baseline")
            self.assertEqual(json.loads((root / "comparison.json").read_text(encoding="utf-8")), updated)


if __name__ == "__main__":
    unittest.main()
