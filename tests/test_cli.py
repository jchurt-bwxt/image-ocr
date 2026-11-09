"""CLI entry-point regressions with model inference and GUI startup mocked."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import image_ocr_main


class EntryPointTests(unittest.TestCase):
    def test_no_arguments_launches_desktop_without_creating_engine(self):
        launch = Mock()
        with patch.dict(sys.modules, {"image_ocr_gui": SimpleNamespace(launch_gui=launch)}), patch.object(
            image_ocr_main, "OCREngine", side_effect=AssertionError("GUI startup should not load models")
        ):
            image_ocr_main.main([])
        launch.assert_called_once_with()

    def run_cli(self, extra_arguments):
        image = Image.new("RGB", (50, 40), "white")
        engine = Mock()
        engine.analyze.return_value = {"mode": "all", "detections": []}
        with patch.object(image_ocr_main, "load_image", return_value=image), patch.object(
            image_ocr_main, "OCREngine", return_value=engine
        ), patch.object(image_ocr_main, "save_outputs") as save, redirect_stdout(io.StringIO()):
            image_ocr_main.main(["sample.png", *extra_arguments])
        options = engine.analyze.call_args.args[1]
        self.assertEqual(save.call_args.args[2], Path("sample_results"))
        return options

    def test_image_without_mode_or_prompt_reads_all_text(self):
        self.assertEqual(self.run_cli([]).mode, "all")

    def test_explicit_all_mode_does_not_require_prompt(self):
        options = self.run_cli(["--mode", "all"])
        self.assertEqual(options.mode, "all")
        self.assertEqual(options.prompt, "")

    def test_legacy_prompt_argument_selects_prompt_mode(self):
        options = self.run_cli(["--prompt", "rectangular sign"])
        self.assertEqual(options.mode, "prompt")
        self.assertEqual(options.prompt, "rectangular sign")


if __name__ == "__main__":
    unittest.main()
