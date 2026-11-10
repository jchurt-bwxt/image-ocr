"""Pinned model download/cache regressions using tiny fixtures and no network."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ocr_model_assets as assets


class InterruptedResponse:
    def __init__(self):
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        self.reads += 1
        if self.reads == 1:
            return b"partial"
        raise OSError("Connection interrupted")


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name) / "cache"
        self.contents = {
            "det": b"tiny detection checkpoint",
            "rec": b"tiny recognition checkpoint",
            "cls": b"tiny classification checkpoint",
        }
        self.fixtures = {
            stage: {
                "filename": f"{stage}.onnx",
                "source_path": f"test-fixtures/{stage}.onnx",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for stage, content in self.contents.items()
        }
        self.asset_patch = patch.object(assets, "ASSETS", self.fixtures)
        self.asset_patch.start()
        self.network_patch = patch.object(assets, "urlopen", side_effect=AssertionError("Real network is forbidden in model asset tests"))
        self.urlopen = self.network_patch.start()
        self.addCleanup(self.network_patch.stop)
        self.addCleanup(self.asset_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def write_cached(self, stages=("det", "rec", "cls")):
        self.cache.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            (self.cache / self.fixtures[stage]["filename"]).write_bytes(self.contents[stage])

    def assert_no_partial_files(self):
        self.assertEqual(list(self.cache.glob("*.part")), [])

    def test_all_three_verified_cached_models_work_offline(self):
        self.write_cached()
        paths = assets.get_enhanced_model_paths(self.cache)
        self.assertEqual(set(paths), {"det", "rec", "cls"})
        for stage, path in paths.items():
            self.assertEqual(path, self.cache / self.fixtures[stage]["filename"])
            self.assertEqual(path.read_bytes(), self.contents[stage])
        self.urlopen.assert_not_called()
        self.assert_no_partial_files()

    def test_missing_models_download_from_pinned_hugging_face_urls(self):
        def response(request, timeout):
            self.assertEqual(timeout, 60)
            for stage, asset in self.fixtures.items():
                expected = (
                    f"https://huggingface.co/{assets.MODEL_REPOSITORY}/resolve/"
                    f"{assets.MODEL_REVISION}/{asset['source_path']}"
                )
                if request.full_url == expected:
                    return io.BytesIO(self.contents[stage])
            self.fail(f"Unexpected download URL: {request.full_url}")

        self.urlopen.side_effect = response
        paths = assets.get_enhanced_model_paths(self.cache)
        self.assertEqual(self.urlopen.call_count, 3)
        self.assertEqual(len(assets.MODEL_REVISION), 40)
        for stage, path in paths.items():
            self.assertTrue(assets.valid_model(path, self.fixtures[stage]))
        self.assert_no_partial_files()

    def test_corrupt_existing_model_is_redownloaded_after_hash_check(self):
        self.write_cached()
        destination = self.cache / self.fixtures["det"]["filename"]
        corrupt = b"X" * len(self.contents["det"])
        destination.write_bytes(corrupt)

        def response(request, timeout):
            self.assertEqual(destination.read_bytes(), corrupt)
            self.assertEqual(request.full_url, assets.asset_url(self.fixtures["det"]))
            return io.BytesIO(self.contents["det"])

        self.urlopen.side_effect = response
        paths = assets.get_enhanced_model_paths(self.cache)
        self.assertEqual(paths["det"].read_bytes(), self.contents["det"])
        self.urlopen.assert_called_once()
        self.assert_no_partial_files()

    def _assert_failed_download_preserves_existing(self, response, message):
        self.write_cached()
        old = b"old invalid checkpoint"
        destination = self.cache / self.fixtures["det"]["filename"]
        destination.write_bytes(old)
        self.urlopen.side_effect = None
        self.urlopen.return_value = response
        with self.assertRaisesRegex(RuntimeError, message):
            assets.get_enhanced_model_paths(self.cache)
        self.assertEqual(destination.read_bytes(), old)
        self.assert_no_partial_files()

    def test_bad_hash_does_not_replace_existing_file_and_removes_partial(self):
        bad = b"X" * len(self.contents["det"])
        self._assert_failed_download_preserves_existing(io.BytesIO(bad), "SHA256")

    def test_oversize_download_does_not_replace_existing_file_and_removes_partial(self):
        self._assert_failed_download_preserves_existing(io.BytesIO(self.contents["det"] + b"extra"), "exceeds")

    def test_truncated_download_does_not_replace_existing_file_and_removes_partial(self):
        self._assert_failed_download_preserves_existing(io.BytesIO(self.contents["det"][:3]), "size or SHA256")

    def test_interrupted_download_does_not_replace_existing_file_and_removes_partial(self):
        self._assert_failed_download_preserves_existing(InterruptedResponse(), "Connection interrupted")

    def test_validation_checks_both_file_size_and_sha256(self):
        self.cache.mkdir()
        destination = self.cache / self.fixtures["det"]["filename"]
        self.assertFalse(assets.valid_model(destination, self.fixtures["det"]))
        destination.write_bytes(self.contents["det"][:-1])
        self.assertFalse(assets.valid_model(destination, self.fixtures["det"]))
        destination.write_bytes(b"X" * len(self.contents["det"]))
        self.assertFalse(assets.valid_model(destination, self.fixtures["det"]))
        destination.write_bytes(self.contents["det"])
        self.assertTrue(assets.valid_model(destination, self.fixtures["det"]))


if __name__ == "__main__":
    unittest.main()
