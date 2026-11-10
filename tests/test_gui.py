"""Model-free desktop worker and event handling tests; no Tk display required."""

import queue
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from image_ocr_gui import OCRApp
from ocr_engine import AnalysisOptions


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.app = OCRApp.__new__(OCRApp)
        self.app.events = queue.Queue()
        self.app.stop_event = threading.Event()
        self.app.engine = Mock()
        self.image = Image.new("RGB", (20, 20), "white")
        self.options = AnalysisOptions()

    def events(self):
        events = []
        while not self.app.events.empty():
            events.append(self.app.events.get_nowait())
        return events

    @patch("image_ocr_gui.load_image")
    def test_success_queues_progress_and_result(self, load):
        load.return_value = self.image

        def analyze(image, options, progress):
            self.assertIs(image, self.image)
            self.assertIs(options, self.options)
            progress("Reading text")
            return {"detections": []}

        self.app.engine.analyze.side_effect = analyze
        self.app._worker([("one", Path("test.png"))], self.options)
        events = self.events()
        self.assertEqual([event[0] for event in events], ["running", "progress", "done", "complete"])
        self.assertEqual(events[1][1], "test.png: Reading text")
        self.assertIs(events[2][2], self.image)
        self.assertEqual(events[2][3]["image"], "test.png")
        self.assertEqual(events[-1], ("complete", 1, 0, False))

    @patch("image_ocr_gui.load_image")
    def test_failed_file_does_not_stop_following_file(self, load):
        load.side_effect = [ValueError("Unsupported image"), self.image]
        self.app.engine.analyze.return_value = {"detections": []}
        self.app._worker([("bad", Path("bad.png")), ("good", Path("good.png"))], self.options)
        events = self.events()
        self.assertEqual([event[0] for event in events], ["running", "error", "running", "done", "complete"])
        self.assertEqual(events[1], ("error", "bad", "Unsupported image"))
        self.assertEqual(events[3][1], "good")
        self.assertEqual(events[-1], ("complete", 1, 1, False))
        self.app.engine.analyze.assert_called_once()

    @patch("image_ocr_gui.load_image")
    def test_analysis_exception_is_reported(self, load):
        load.return_value = self.image
        self.app.engine.analyze.side_effect = RuntimeError("Model unavailable")
        self.app._worker([("one", Path("test.png"))], self.options)
        self.assertEqual(self.events(), [
            ("running", "one", "test.png"),
            ("error", "one", "Model unavailable"),
            ("complete", 0, 1, False),
        ])

    @patch("image_ocr_gui.load_image")
    def test_stop_before_first_file_skips_loading(self, load):
        self.app.stop_event.set()
        self.app._worker([("one", Path("test.png"))], self.options)
        self.assertEqual(self.events(), [("complete", 0, 0, True)])
        load.assert_not_called()
        self.app.engine.analyze.assert_not_called()

    @patch("image_ocr_gui.load_image")
    def test_stop_finishes_current_file_and_skips_rest(self, load):
        load.return_value = self.image

        def analyze(*_args, **_kwargs):
            self.app.stop_event.set()
            return {"detections": []}

        self.app.engine.analyze.side_effect = analyze
        self.app._worker([("one", Path("first.png")), ("two", Path("second.png"))], self.options)
        events = self.events()
        self.assertEqual([event[0] for event in events], ["running", "done", "complete"])
        self.assertEqual(events[-1], ("complete", 1, 0, True))
        load.assert_called_once_with(Path("first.png"))


class EventTests(unittest.TestCase):
    def test_failed_rerun_clears_previous_preview_and_result(self):
        app = OCRApp.__new__(OCRApp)
        app.events = queue.Queue()
        app.root = Mock()
        app.file_tree = Mock()
        app.items = {"one": {
            "image": Image.new("RGB", (20, 20)),
            "annotated": Image.new("RGB", (20, 20)),
            "payload": {"detections": [{"text": "old output"}]},
            "error": None,
        }}
        app._selected = Mock(return_value=("one", app.items["one"]))
        app._show_selected = Mock()
        app.events.put(("error", "one", "Unreadable file"))
        app._poll_events()
        self.assertIsNone(app.items["one"]["image"])
        self.assertIsNone(app.items["one"]["annotated"])
        self.assertIsNone(app.items["one"]["payload"])
        self.assertEqual(app.items["one"]["error"], "Unreadable file")
        app.file_tree.set.assert_called_once_with("one", "status", "Error")
        app._show_selected.assert_called_once()
        app.root.after.assert_called_once()


class ProfileSettingsTests(unittest.TestCase):
    def app(self, profile="Difficult markings (enhanced)", languages="en"):
        app = OCRApp.__new__(OCRApp)
        for attribute, value in {
            "mode": "All text in image", "ocr_profile": profile, "languages": languages,
            "threshold": "0.30", "min_area": "0.1", "prompt": "", "model": "model-id", "device": "cpu",
        }.items():
            variable = Mock()
            variable.get.return_value = value
            setattr(app, attribute, variable)
        return app

    def test_options_send_enhanced_profile(self):
        options = self.app()._options()
        self.assertEqual(options.ocr_profile, "enhanced")
        self.assertEqual(options.languages, ("en",))

    def test_original_profile_preserves_selected_languages(self):
        options = self.app("Original EasyOCR (baseline)", "en,fr")._options()
        self.assertEqual(options.ocr_profile, "baseline")
        self.assertEqual(options.languages, ("en", "fr"))

    def test_enhanced_other_languages_raise_without_silent_changes(self):
        app = self.app(languages="fr")
        with self.assertRaisesRegex(ValueError, "Original EasyOCR"):
            app._options()
        app.languages.set.assert_not_called()

    def test_profile_selector_is_disabled_during_analysis(self):
        app = self.app()
        app.busy = True
        app.items = {}
        app.stop_event = threading.Event()
        app._selected = Mock(return_value=(None, None))
        for attribute in (
            "add_button", "language_entry", "remove_button", "clear_button", "mode_combo", "profile_combo",
            "device_combo", "prompt_entry", "model_entry", "model_button", "threshold_entry", "area_entry",
            "analyze_button", "stop_button", "export_button", "copy_button",
        ):
            setattr(app, attribute, Mock())
        app._set_controls()
        app.profile_combo.configure.assert_called_once_with(state="disabled")


if __name__ == "__main__":
    unittest.main()
