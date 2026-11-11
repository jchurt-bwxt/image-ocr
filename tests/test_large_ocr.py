"""Large-model process isolation and decoder contracts, without model weights."""
from contextlib import nullcontext
import builtins
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import large_ocr


def stalled_worker(tasks, results):
    """A real spawn target that simulates native inference which never returns."""
    results.send({"status": "ready"})
    tasks.recv()
    while True:
        time.sleep(0.1)


class ProcessDouble:
    def __init__(self, stubborn=False, start_error=None):
        self.pid = None
        self.alive = False
        self.stubborn = stubborn
        self.start_error = start_error
        self.start = Mock(side_effect=self._start)
        self.is_alive = Mock(side_effect=lambda: self.alive)
        self.terminate = Mock(side_effect=self._terminate)
        self.kill = Mock(side_effect=self._kill)
        self.join = Mock()

    def _start(self):
        if self.start_error:
            raise self.start_error
        self.pid, self.alive = 123, True

    def _terminate(self):
        if not self.stubborn:
            self.alive = False

    def _kill(self):
        self.alive = False


def fake_context(messages, *, stubborn=False, start_error=None):
    process = ProcessDouble(stubborn, start_error)
    child_tasks, tasks, results, child_results = [Mock() for _ in range(4)]
    results.poll.return_value = True
    results.recv.side_effect = messages
    context = SimpleNamespace(
        Pipe=Mock(side_effect=[(child_tasks, tasks), (results, child_results)]),
        Process=Mock(return_value=process),
    )
    return context, process, (child_tasks, tasks, results, child_results)


class LargeReaderTests(unittest.TestCase):
    def make_reader(self, messages, **kwargs):
        context, process, pipes = fake_context(messages, **kwargs)
        with patch("large_ocr.mp.get_context", return_value=context):
            reader = large_ocr.LargeOCRReader(startup_timeout=2)
        self.addCleanup(reader.close)
        return reader, process, pipes

    def test_ready_worker_returns_original_labels_and_no_fabricated_confidence(self):
        polygon = [[1.0, 2.0], [10.0, 2.0], [10.0, 8.0], [1.0, 8.0]]
        lines = [(polygon, "O0-I1 / #5", None)]
        reader, process, pipes = self.make_reader([
            {"status": "ready"},
            {"status": "ok", "lines": lines, "details": {"confidence_available": False, "raw_generation": "raw"}},
        ])
        actual = reader.readtext(np.zeros((20, 30, 3), dtype=np.uint8), timeout=7)
        self.assertEqual(actual, lines)
        self.assertIsNone(actual[0][2])
        self.assertEqual(reader.last_details["inference_budget_seconds"], 7)
        self.assertEqual(reader.last_details["raw_generation"], "raw")
        encoded = pipes[1].send.call_args.args[0]["image"]
        with Image.open(BytesIO(encoded)) as image:
            self.assertEqual(image.size, (30, 20))
        process.terminate.assert_not_called()
        pipes[0].close.assert_called_once_with()
        pipes[3].close.assert_called_once_with()

    def test_inference_timeout_terminates_and_joins_native_worker(self):
        reader, process, pipes = self.make_reader([{"status": "ready"}])
        pipes[2].poll.return_value = False
        with self.assertRaisesRegex(TimeoutError, "worker terminated"):
            reader.readtext(np.zeros((8, 8, 3), dtype=np.uint8), timeout=0.1)
        process.terminate.assert_called_once_with()
        process.join.assert_called_with(timeout=2)
        process.kill.assert_not_called()
        self.assertFalse(process.is_alive())
        pipes[1].close.assert_called()
        pipes[2].close.assert_called()

    def test_stubborn_worker_is_killed_after_terminate_does_not_stop_it(self):
        reader, process, pipes = self.make_reader([{"status": "ready"}], stubborn=True)
        pipes[2].poll.return_value = False
        with self.assertRaises(TimeoutError):
            reader.readtext(np.zeros((8, 8, 3), dtype=np.uint8), timeout=0.1)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual([call.kwargs["timeout"] for call in process.join.call_args_list], [2, 1])
        self.assertFalse(process.is_alive())

    def test_model_startup_error_closes_worker_and_all_pipe_endpoints(self):
        context, process, pipes = fake_context([{"status": "error", "error": "not enough RAM"}])
        with patch("large_ocr.mp.get_context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "not enough RAM"):
                large_ocr.LargeOCRReader(startup_timeout=2)
        process.terminate.assert_called_once_with()
        self.assertFalse(process.is_alive())
        for pipe in pipes:
            pipe.close.assert_called()

    def test_os_process_start_failure_also_closes_all_pipes(self):
        context, process, pipes = fake_context([], start_error=OSError("cannot spawn"))
        with patch("large_ocr.mp.get_context", return_value=context):
            with self.assertRaisesRegex(OSError, "cannot spawn"):
                large_ocr.LargeOCRReader(startup_timeout=2)
        process.join.assert_not_called()
        for pipe in pipes:
            pipe.close.assert_called_once_with()

    def test_crashed_worker_reports_error_and_is_cleaned_up(self):
        reader, process, pipes = self.make_reader([{"status": "ready"}, EOFError()])
        with self.assertRaisesRegex(RuntimeError, "worker stopped"):
            reader.readtext(np.zeros((8, 8, 3), dtype=np.uint8))
        process.terminate.assert_called_once_with()
        self.assertFalse(process.is_alive())

    def test_real_spawned_inference_process_is_dead_after_deadline(self):
        # This exercises OS process termination, not just a soft timeout flag.
        with patch("large_ocr._worker", stalled_worker):
            reader = large_ocr.LargeOCRReader(startup_timeout=10)
        try:
            process = reader._process
            self.assertTrue(process.is_alive())
            with self.assertRaises(TimeoutError):
                reader.readtext(np.zeros((8, 8, 3), dtype=np.uint8), timeout=0.05)
            self.assertFalse(process.is_alive())
            self.assertIsNotNone(process.exitcode)
        finally:
            reader.close()


class WorkerDecoderTests(unittest.TestCase):
    def test_native_worker_uses_safe_loaders_and_parse_compatible_decoding(self):
        processor, model = Mock(), Mock()
        processor.return_value = {}
        processor.batch_decode.return_value = ["raw OCR with location tokens"]
        processor.post_process_generation.return_value = {
            large_ocr.TASK: {"labels": ["O0-I1 / #5"], "quad_boxes": [[1, 2, 10, 2, 10, 8, 1, 8]]}
        }
        model.to.return_value = model
        model.eval.return_value = model
        generated = SimpleNamespace(shape=(1, 23))
        model.generate.return_value = generated
        processor_loader, model_loader = Mock(return_value=processor), Mock(return_value=model)
        fake_torch = SimpleNamespace(
            set_num_threads=Mock(), set_num_interop_threads=Mock(), float32=object(),
            inference_mode=lambda: nullcontext(),
        )
        fake_transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            Florence2ForConditionalGeneration=SimpleNamespace(from_pretrained=model_loader),
        )
        encoded = BytesIO()
        Image.new("RGB", (30, 20)).save(encoded, format="PNG")
        tasks, results = Mock(), Mock()
        tasks.recv.side_effect = [{"image": encoded.getvalue()}, None]
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}), patch(
            "large_ocr.get_large_model_directory", return_value="/verified/local-florence-model"
        ):
            large_ocr._worker(tasks, results)
        self.assertEqual(processor_loader.call_args.args[0], "/verified/local-florence-model")
        self.assertEqual(model_loader.call_args.args[0], "/verified/local-florence-model")
        self.assertTrue(processor_loader.call_args.kwargs["local_files_only"])
        self.assertTrue(model_loader.call_args.kwargs["local_files_only"])
        self.assertFalse(processor_loader.call_args.kwargs["trust_remote_code"])
        self.assertFalse(model_loader.call_args.kwargs["trust_remote_code"])
        self.assertTrue(model_loader.call_args.kwargs["use_safetensors"])
        processor.batch_decode.assert_called_once_with(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
        self.assertEqual(model.generate.call_args.kwargs["num_beams"], 1)
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], large_ocr.MAX_NEW_TOKENS)
        ok = results.send.call_args_list[1].args[0]
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["lines"][0][1], "O0-I1 / #5")
        self.assertIsNone(ok["lines"][0][2])
        self.assertFalse(ok["details"]["confidence_available"])


class ModelSourceTests(unittest.TestCase):
    @staticmethod
    def native_model_directory(directory):
        path = Path(directory)
        for filename in (
            "config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
        ):
            (path / filename).write_text("{}", encoding="utf-8")
        return path

    def test_explicit_local_large_model_uses_no_hub_or_tls_network_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.native_model_directory(directory)
            download = Mock(side_effect=AssertionError("Local model must not access the Hugging Face hub"))
            with patch.dict(os.environ, {"IMAGE_OCR_LARGE_MODEL_DIR": str(source)}, clear=True), patch.dict(
                sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}
            ), patch("ocr_network.configure_tls") as tls:
                actual = large_ocr.get_large_model_directory()
            self.assertEqual(Path(actual), source.resolve())
            download.assert_not_called()
            tls.assert_not_called()

    def test_tls_is_configured_before_pinned_hub_snapshot_download(self):
        order = []
        with tempfile.TemporaryDirectory() as directory:
            source = self.native_model_directory(directory)
            tls = Mock(side_effect=lambda: order.append("tls"))
            def snapshot(**kwargs):
                self.assertEqual(order, ["tls"])
                order.append("snapshot")
                return str(source)
            download = Mock(side_effect=snapshot)
            original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name == "huggingface_hub":
                    self.assertEqual(order, ["tls"], "TLS must be configured before importing the Hugging Face client")
                return original_import(name, *args, **kwargs)
            with patch.dict(os.environ, {}, clear=True), patch.dict(
                sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}
            ), patch("ocr_network.configure_tls", tls), patch("builtins.__import__", side_effect=guarded_import):
                actual = large_ocr.get_large_model_directory()
            self.assertEqual(Path(actual), source.resolve())
        self.assertEqual(order, ["tls", "snapshot"])
        parameters = download.call_args.kwargs
        self.assertEqual(parameters["repo_id"], large_ocr.MODEL_REPOSITORY)
        self.assertEqual(parameters["revision"], large_ocr.MODEL_REVISION)
        self.assertEqual(len(parameters["revision"]), 40)
        allowlist = parameters["allow_patterns"]
        self.assertIn("config.json", allowlist)
        self.assertTrue(any("safetensors" in filename for filename in allowlist))
        self.assertFalse(any(filename == "*" or ".py" in filename for filename in allowlist))
        self.assertNotIn("verify", parameters)


if __name__ == "__main__":
    unittest.main()
