"""TLS configuration regressions without contacting any external service."""
import os
from pathlib import Path
import ssl
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ocr_network


class TLSConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.inject = Mock()
        self.truststore = patch.dict(sys.modules, {"truststore": SimpleNamespace(inject_into_ssl=self.inject)})
        self.truststore.start()
        self.addCleanup(self.truststore.stop)
        self.injected_state = patch.object(ocr_network, "_injected", False)
        self.injected_state.start()
        self.addCleanup(self.injected_state.stop)

    def test_system_trust_is_installed_without_disabling_certificate_checks(self):
        original_unverified = ssl._create_unverified_context
        original_https_context = ssl._create_default_https_context
        with patch.dict(os.environ, {}, clear=True):
            ocr_network.configure_tls()
            self.assertNotIn("PYTHONHTTPSVERIFY", os.environ)
            self.assertNotIn("HF_HUB_DISABLE_SSL_VERIFY", os.environ)
            for key in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
                self.assertNotIn(key, os.environ)
        self.inject.assert_called_once_with()
        self.assertIs(ssl._create_unverified_context, original_unverified)
        self.assertIs(ssl._create_default_https_context, original_https_context)

    def test_missing_truststore_dependency_fails_without_disabling_verification(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"truststore": None}):
            with self.assertRaises((ImportError, RuntimeError)):
                ocr_network.configure_tls()
            self.assertNotIn("PYTHONHTTPSVERIFY", os.environ)
            self.assertNotIn("HF_HUB_DISABLE_SSL_VERIFY", os.environ)

    def test_existing_bundle_is_preserved_and_synced_instead_of_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "existing.pem"
            bundle.touch()
            with patch.dict(os.environ, {"REQUESTS_CA_BUNDLE": str(bundle)}, clear=True), patch(
                "ocr_network.ssl.SSLContext", return_value=Mock()
            ):
                ocr_network.configure_tls()
                for key in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
                    self.assertEqual(os.environ[key], str(bundle.resolve()))

    def test_custom_bundle_is_validated_and_forwarded_to_every_http_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "company-ca.pem"
            bundle.write_text("unit-test certificate placeholder", encoding="ascii")
            context = Mock()
            with patch.dict(os.environ, {"IMAGE_OCR_CA_BUNDLE": str(bundle)}, clear=True), patch(
                "ocr_network.ssl.SSLContext", return_value=context
            ) as create_context:
                ocr_network.configure_tls()
                create_context.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
                context.load_verify_locations.assert_called_once_with(cafile=str(bundle.resolve()))
                for key in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
                    self.assertEqual(os.environ[key], str(bundle.resolve()))
        self.inject.assert_called_once_with()

    def test_missing_custom_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"IMAGE_OCR_CA_BUNDLE": str(Path(directory) / "missing.pem")}, clear=True
        ):
            with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                ocr_network.configure_tls()
            self.assertNotIn("REQUESTS_CA_BUNDLE", os.environ)
            self.assertNotIn("CURL_CA_BUNDLE", os.environ)
            self.assertNotIn("SSL_CERT_FILE", os.environ)
        self.inject.assert_not_called()

    def test_invalid_ca_content_fails_ssl_validation_without_installing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "invalid.pem"
            bundle.write_text("This is not a CA certificate", encoding="ascii")
            with patch.dict(os.environ, {"IMAGE_OCR_CA_BUNDLE": str(bundle)}, clear=True):
                with self.assertRaises((ValueError, ssl.SSLError, RuntimeError)):
                    ocr_network.configure_tls()
                self.assertNotIn("REQUESTS_CA_BUNDLE", os.environ)
                self.assertNotIn("CURL_CA_BUNDLE", os.environ)
                self.assertNotIn("SSL_CERT_FILE", os.environ)
        self.inject.assert_not_called()

    def test_real_system_ca_bundle_is_accepted_with_verified_ssl_context(self):
        # Exercise real PEM parsing and CERT_REQUIRED locally, without a socket.
        system_ca = ssl.get_default_verify_paths().cafile
        if not system_ca or not Path(system_ca).is_file():
            self.skipTest("No readable system CA bundle on this platform")
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "system-ca-copy.pem"
            bundle.write_bytes(Path(system_ca).read_bytes())
            with patch.dict(os.environ, {"IMAGE_OCR_CA_BUNDLE": str(bundle)}, clear=True):
                result = ocr_network.configure_tls()
                context = ssl.create_default_context(cafile=result["ca_bundle"])
            self.assertEqual(result["ca_bundle"], str(bundle.resolve()))
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)

    def test_explicit_bundle_wins_over_conflicting_existing_client_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "approved.pem"
            bundle.touch()
            with patch.dict(os.environ, {"IMAGE_OCR_CA_BUNDLE": str(bundle), "REQUESTS_CA_BUNDLE": "old-missing.pem"}, clear=True), patch(
                "ocr_network.ssl.SSLContext", return_value=Mock()
            ):
                ocr_network.configure_tls()
                for key in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
                    self.assertEqual(os.environ[key], str(bundle.resolve()))

    def test_native_trust_is_injected_once_and_python_transport_keeps_verification(self):
        with patch.dict(os.environ, {}, clear=True):
            ocr_network.configure_tls()
            ocr_network.configure_tls()
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
            self.assertEqual(os.environ["HF_HUB_ENABLE_HF_TRANSFER"], "0")
        self.inject.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
