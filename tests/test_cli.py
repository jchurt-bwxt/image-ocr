"""CLI profile and comparison tests without loading model weights."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image
import image_ocr_main as cli
from ocr_profiles import OCR_PROFILES, PROFILE_LABELS


def model_row(profile="enhanced", status="ok"):
    return {"profile": profile, "label": PROFILE_LABELS[profile], "status": status, "model_setup_seconds": 1,
            "analysis_seconds": 2, "total_seconds": 3, "model_reused": False, "text": "AB12" if status == "ok" else "",
            "payload": {"detections": []} if status == "ok" else None, "error": None if status == "ok" else "Processing limit reached"}


class EntryPointTests(unittest.TestCase):
    def test_no_arguments_launch_desktop_without_engine(self):
        launch = Mock()
        with patch.dict(sys.modules, {"image_ocr_gui": SimpleNamespace(launch_gui=launch)}), patch.object(cli, "OCREngine") as engine:
            cli.main([])
        launch.assert_called_once_with()
        engine.assert_not_called()

    def run_cli(self, extra):
        engine = Mock()
        with patch.object(cli, "load_image", return_value=Image.new("RGB", (20, 20))), patch.object(cli, "OCREngine", return_value=engine), patch.object(cli, "run_model", return_value=model_row()) as run, patch.object(cli, "save_outputs") as save, redirect_stdout(io.StringIO()) as output:
            cli.main(["sample.png", *extra])
        engine.close.assert_called_once_with()
        self.assertEqual(save.call_args.args[2], Path("sample_results"))
        self.assertIn("startup 1.00 s, processing 2.00 s, total 3.00 s", output.getvalue())
        return run.call_args.args[1]

    def test_image_defaults_to_all_text_enhanced(self):
        options = self.run_cli([])
        self.assertEqual(options.mode, "all")
        self.assertEqual(options.ocr_profile, "enhanced")

    def test_geometry_large_profile_and_budget_forwarded(self):
        options = self.run_cli(["--mode", "rectangles", "--ocr-profile", "large", "--large-max-seconds", "90"])
        self.assertEqual(options.mode, "rectangles")
        self.assertEqual(options.ocr_profile, "large")
        self.assertEqual(options.large_max_seconds, 90)

    def test_baseline_preserves_languages(self):
        self.assertEqual(self.run_cli(["--ocr-profile", "baseline", "--languages", "en,fr"]).languages, ("en", "fr"))

    def test_removed_detector_flags_are_rejected(self):
        for flag, value in (("--prompt", "box"), ("--model", "checkpoint"), ("--threshold", "0.3")):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(["sample.png", flag, value])

    def test_compare_exports_every_model_row_and_closes_engine(self):
        engine = Mock()
        rows = [model_row(p, "timeout" if p == "large" else "ok") for p in OCR_PROFILES]
        report = {"models": rows, "wall_seconds": 9}
        with patch.object(cli, "load_image", return_value=Image.new("RGB", (20, 20))), patch.object(cli, "OCREngine", return_value=engine), patch.object(cli, "compare_image", return_value=report) as compare, patch.object(cli, "save_comparison") as save, patch.object(cli, "save_outputs") as single_save, redirect_stdout(io.StringIO()) as output:
            cli.main(["sample.png", "--compare-models"])
        self.assertIs(compare.call_args.kwargs["engine"], engine)
        self.assertEqual(save.call_args.args[1]["models"], rows)
        self.assertIn("timeout", output.getvalue())
        single_save.assert_not_called()
        engine.close.assert_called_once_with()

    def test_invalid_compare_language_device_never_load_engine(self):
        for extra in (("--ocr-profile", "baseline", "--languages", "fr"), ("--ocr-profile", "baseline", "--device", "cuda")):
            with self.subTest(extra=extra), patch.object(cli, "OCREngine") as engine, self.assertRaisesRegex(SystemExit, "English|CPU"):
                cli.main(["sample.png", "--compare-models", *extra])
            engine.assert_not_called()

    def test_single_timeout_has_no_success_export(self):
        engine = Mock()
        with patch.object(cli, "load_image", return_value=Image.new("RGB", (20, 20))), patch.object(cli, "OCREngine", return_value=engine), patch.object(cli, "run_model", return_value=model_row("large", "timeout")), patch.object(cli, "save_outputs") as save, redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "timeout"):
            cli.main(["sample.png", "--ocr-profile", "large"])
        save.assert_not_called()
        engine.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
