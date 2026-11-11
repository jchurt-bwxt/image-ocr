"""Desktop model selection/comparison tests; no Tk display or models required."""
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from image_ocr_gui import MODES, OCRApp
from ocr_engine import AnalysisOptions
from ocr_profiles import OCR_PROFILES, PROFILE_LABELS


def model_row(profile="enhanced", status="ok", text="AB12"):
    return {
        "profile": profile, "label": PROFILE_LABELS[profile], "status": status,
        "model_setup_seconds": 1.5, "analysis_seconds": 2.5, "total_seconds": 4.0,
        "model_reused": False, "text": text if status == "ok" else "",
        "payload": {"detections": [{"id": 1, "label": "text", "confidence": 0.9, "text": text, "ocr_lines": []}]} if status == "ok" else None,
        "error": None if status == "ok" else "No model result",
    }


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.app = OCRApp.__new__(OCRApp)
        self.app.events = queue.Queue()
        self.app.stop_event = threading.Event()
        self.app.engine = Mock()
        self.image = Image.new("RGB", (20, 20), "white")
        self.options = AnalysisOptions()

    def events(self):
        return list(self.app.events.queue)

    @patch("image_ocr_gui.load_image")
    @patch("image_ocr_gui.run_model")
    def test_single_result_progress_timing_and_cleanup(self, run, load):
        load.return_value = self.image
        def analyze(image, options, engine, progress):
            progress("Reading text")
            return model_row(options.ocr_profile)
        run.side_effect = analyze
        self.app._worker([("one", Path("test.png"))], self.options)
        events = self.events()
        self.assertEqual([e[0] for e in events], ["running", "progress", "model_done", "complete"])
        self.assertEqual(events[1][1], "test.png: Reading text")
        self.assertIs(events[2][2], self.image)
        self.assertEqual(events[2][3]["payload"]["image"], "test.png")
        self.assertEqual(events[-1], ("complete", {"ok": 1, "error": 0, "timeout": 0}, False))
        self.app.engine.close.assert_called_once_with()

    @patch("image_ocr_gui.load_image")
    @patch("image_ocr_gui.run_model")
    def test_bad_image_continues_to_next_image(self, run, load):
        load.side_effect = [ValueError("Unsupported image"), self.image]
        run.return_value = model_row()
        self.app._worker([("bad", Path("bad.png")), ("good", Path("good.png"))], self.options)
        events = self.events()
        self.assertEqual(events[1][3]["status"], "error")
        self.assertIn("Unsupported image", events[1][3]["error"])
        self.assertIsNone(events[1][2])
        self.assertEqual(events[-1][1], {"ok": 1, "error": 1, "timeout": 0})
        run.assert_called_once()

    @patch("image_ocr_gui.load_image")
    @patch("image_ocr_gui.run_model")
    def test_timeout_stays_visible_and_next_image_runs(self, run, load):
        load.return_value = self.image
        run.side_effect = [model_row(status="timeout"), model_row()]
        self.app._worker([("first", Path("first.png")), ("second", Path("second.png"))], self.options)
        self.assertEqual([e[3]["status"] for e in self.events() if e[0] == "model_done"], ["timeout", "ok"])
        self.assertEqual(self.events()[-1][1], {"ok": 1, "error": 0, "timeout": 1})

    @patch("image_ocr_gui.load_image")
    @patch("image_ocr_gui.run_model")
    def test_stop_before_first_file_skips_loading(self, run, load):
        self.app.stop_event.set()
        self.app._worker([("one", Path("test.png"))], self.options)
        self.assertEqual(self.events(), [("complete", {"ok": 0, "error": 0, "timeout": 0}, True)])
        load.assert_not_called()
        run.assert_not_called()
        self.app.engine.close.assert_called_once()

    @patch("image_ocr_gui.load_image")
    @patch("image_ocr_gui.run_model")
    def test_stop_after_current_analysis_skips_remaining_models(self, run, load):
        load.return_value = self.image
        def analyze(*_args, **_kwargs):
            self.app.stop_event.set()
            return model_row("baseline")
        run.side_effect = analyze
        self.app._worker([("one", Path("a.png")), ("two", Path("b.png"))], self.options, compare=True)
        self.assertEqual([e[0] for e in self.events()], ["running", "model_done", "complete"])
        self.assertEqual(self.events()[-1], ("complete", {"ok": 1, "error": 0, "timeout": 0}, True))
        run.assert_called_once()

    @patch("image_ocr_gui.load_image")
    @patch("image_ocr_gui.run_model")
    def test_comparison_runs_model_first_to_reuse_warm_cache(self, run, load):
        load.side_effect = lambda path: path.name
        order = []
        def analyze(image, options, engine, progress):
            order.append((options.ocr_profile, image))
            return model_row(options.ocr_profile)
        run.side_effect = analyze
        self.app._worker([("one", Path("a.png")), ("two", Path("b.png"))], self.options, compare=True)
        self.assertEqual(order, [(p, f) for p in OCR_PROFILES for f in ("a.png", "b.png")])
        self.assertEqual(self.events()[-1], ("complete", {"ok": 6, "error": 0, "timeout": 0}, False))
        self.app.engine.close.assert_called_once()


class ResultTests(unittest.TestCase):
    def result_app(self, item):
        app = OCRApp.__new__(OCRApp)
        app._selected = Mock(return_value=("one", item))
        app.region_tree = Mock()
        app.region_tree.get_children.return_value = ()
        app.text_output, app.json_output, app.timing_caption = Mock(), Mock(), Mock()
        app._render_preview, app._set_controls = Mock(), Mock()
        return app

    def test_active_model_controls_text_payload_json_and_timing(self):
        item = {"models": {"baseline": model_row("baseline", text="small"), "large": model_row("large", text="large")}, "active_profile": "large"}
        app = self.result_app(item)
        app._show_active_result()
        self.assertIs(item["payload"], item["models"]["large"]["payload"])
        app.text_output.insert.assert_called_with("1.0", "large")
        self.assertIn('"profile": "large"', app.json_output.insert.call_args.args[1])
        self.assertIn("startup 1.50 s", app.timing_caption.set.call_args.args[0])
        self.assertIsNone(item["annotated"])
        app._render_preview.assert_called_once()

    def test_failed_active_row_clears_previous_payload(self):
        for status in ("error", "timeout"):
            with self.subTest(status=status):
                item = {"models": {"large": model_row("large", status=status)}, "active_profile": "large", "payload": {"detections": []}}
                app = self.result_app(item)
                app._show_active_result()
                self.assertIsNone(item["payload"])
                self.assertIn(status, app.text_output.insert.call_args.args[1])

    def test_comparison_export_includes_failed_model_row(self):
        app = OCRApp.__new__(OCRApp)
        rows = {p: model_row(p, status="timeout" if p == "large" else "ok") for p in OCR_PROFILES}
        item = {"path": Path("test.png"), "image": Image.new("RGB", (20, 20)), "models": rows, "comparison": True, "payload": None}
        app._selected = Mock(return_value=("one", item))
        app.root, app.status = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, patch("image_ocr_gui.filedialog.askdirectory", return_value=directory), patch("image_ocr_gui.save_comparison") as save:
            app._export()
        report = save.call_args.args[1]
        self.assertEqual([r["profile"] for r in report["models"]], list(OCR_PROFILES))
        self.assertEqual(report["models"][-1]["status"], "timeout")
        self.assertEqual(report["wall_seconds"], 12)

    def test_copy_uses_active_result(self):
        app = OCRApp.__new__(OCRApp)
        app._selected = Mock(return_value=("one", {"payload": model_row(text="active result")["payload"]}))
        app.root, app.status = Mock(), Mock()
        app._copy_text()
        app.root.clipboard_append.assert_called_once_with("active result")


class SettingsTests(unittest.TestCase):
    def app(self, profile="enhanced", languages="en", device="cpu", budget="120"):
        app = OCRApp.__new__(OCRApp)
        for name, value in {"mode": "All text in image", "ocr_profile": PROFILE_LABELS[profile], "languages": languages, "min_area": "0.1", "device": device, "large_budget": budget}.items():
            variable = Mock()
            variable.get.return_value = value
            setattr(app, name, variable)
        return app

    def test_only_geometry_modes_and_three_profiles(self):
        self.assertEqual(set(MODES.values()), {"all", "rectangles", "triangles"})
        self.assertEqual(tuple(OCR_PROFILES), ("baseline", "enhanced", "large"))
        self.assertEqual(self.app()._options().ocr_profile, "enhanced")
        self.assertEqual(self.app("large")._options().large_max_seconds, 120)

    def test_baseline_preserves_languages_and_cuda(self):
        options = self.app("baseline", "en,fr", "cuda")._options()
        self.assertEqual(options.languages, ("en", "fr"))
        self.assertEqual(options.device, "cuda")

    def test_enhanced_large_and_compare_require_english_cpu(self):
        for profile in ("enhanced", "large"):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(ValueError, "baseline"):
                    self.app(profile, "fr")._options()
                with self.assertRaisesRegex(ValueError, "CPU"):
                    self.app(profile, device="cuda")._options()
        with self.assertRaisesRegex(ValueError, "English"):
            self.app("baseline", "fr")._options(compare=True)

    def test_large_budget_is_bounded(self):
        for budget in ("0", "121", "nan"):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "120"):
                self.app("large", budget=budget)._options()

    def test_selectors_and_compare_disabled_while_busy(self):
        app = self.app()
        app.busy, app.items = True, {}
        app.stop_event = threading.Event()
        app._selected = Mock(return_value=(None, None))
        for name in ("add_button", "language_entry", "remove_button", "clear_button", "mode_combo", "profile_combo", "device_combo", "area_entry", "budget_entry", "analyze_button", "compare_button", "stop_button", "export_button", "copy_button"):
            setattr(app, name, Mock())
        app._set_controls()
        app.profile_combo.configure.assert_called_once_with(state="disabled")
        app.compare_button.configure.assert_called_once_with(state="disabled")


if __name__ == "__main__":
    unittest.main()
