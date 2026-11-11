"""Application-level HTTPS trust for model downloads; verification stays on."""
from __future__ import annotations

import os
from pathlib import Path
import ssl

_injected = False
CA_VARIABLES = ("IMAGE_OCR_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")


def configure_tls() -> dict:
    """Use native OS trust, optionally augmented by an approved PEM CA bundle.

    Call before importing Hugging Face/requests in each application process.
    An explicit bundle is selected in CA_VARIABLES order and applied to both
    requests and urllib. No global machine certificate settings are modified.
    """
    global _injected
    configured = next((os.environ[name] for name in CA_VARIABLES if os.environ.get(name)), None)
    bundle = None
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise ValueError(f"CA bundle file does not exist: {path}. Use an IT-approved PEM CA bundle.")
        bundle = str(path.resolve())
        # Parse the bundle before changing process settings; invalid trust fails closed.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        try:
            context.load_verify_locations(cafile=bundle)
        except (ssl.SSLError, OSError) as exc:
            raise ValueError(f"Invalid PEM CA bundle: {path}. Request an approved CA bundle from IT.") from exc
    if not _injected:
        try:
            import truststore
        except ImportError as exc:
            raise ImportError("Model-download TLS support requires truststore. Install the updated requirements.txt.") from exc
        truststore.inject_into_ssl()
        _injected = True
    if bundle:
        for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ[name] = bundle
    # Keep all HF downloads on the Python transport governed by this trust setup.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return {"system_trust": True, "ca_bundle": bundle, "transport": "verified Python HTTPS"}


def describe_download_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    lower = message.lower()
    if any(marker in lower for marker in ("certificate_verify_failed", "certificate verify failed", "unable to get local issuer", "sslcertverificationerror")):
        return (
            "HTTPS certificate verification failed before model inference. "
            "Install the updated requirements and restart. If this network uses an enterprise certificate, "
            "ask IT to install its approved root/intermediate CA in Windows or provide a PEM CA bundle. "
            "Set IMAGE_OCR_CA_BUNDLE to that file and launch the app from the same environment. "
            "A complete local model snapshot can also be used via IMAGE_OCR_LARGE_MODEL_DIR. "
            "Run python ocr_network.py --check to test verified model-download access.\n\n" + message
        )
    return message


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check verified urllib and requests access to a small Hugging Face config file.")
    args = parser.parse_args()
    if not args.check:
        parser.print_help()
        return 0
    try:
        settings = configure_tls()
        # Import clients only after native trust is configured.
        from urllib.request import urlopen
        import requests
        from large_ocr import MODEL_REPOSITORY, MODEL_REVISION
        url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/config.json"
        with urlopen(url, timeout=30) as response:
            response.read(1024)
        with requests.get(url, timeout=30, stream=True) as response:
            response.raise_for_status()
        print("Verified HTTPS succeeded with urllib and requests.")
        print("Trust: operating system" + (f" plus CA bundle {settings['ca_bundle']}" if settings["ca_bundle"] else ""))
        return 0
    except Exception as exc:
        print(describe_download_error(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
