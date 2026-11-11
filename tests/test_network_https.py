"""Real certificate-chain/hostname verification against a temporary local CA."""
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL CLI is needed for temporary test certificates")
class HTTPSVerificationTests(unittest.TestCase):
    def test_approved_bundle_accepts_chain_but_untrusted_ca_and_wrong_host_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)

            def openssl(*arguments):
                subprocess.run(["openssl", *arguments], cwd=folder, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.pem",
                    "-days", "1", "-subj", "/CN=Image-OCR-Test-CA", "-addext", "basicConstraints=critical,CA:TRUE",
                    "-addext", "keyUsage=critical,keyCertSign,cRLSign")
            openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", "server.key", "-out", "server.csr", "-subj", "/CN=localhost")
            (folder / "extensions.cnf").write_text(
                "subjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\n"
                "keyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n", encoding="utf-8")
            openssl("x509", "-req", "-in", "server.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
                    "-out", "server.pem", "-days", "1", "-extfile", "extensions.cnf")

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"verified")

                def log_message(self, *_args):
                    pass

            try:
                server = HTTPServer(("127.0.0.1", 0), Handler)
            except OSError as exc:
                self.skipTest(f"Local test server is unavailable: {exc}")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(folder / "server.pem", folder / "server.key")
            server.socket = context.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            environment = os.environ.copy()
            for key in ("IMAGE_OCR_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
                environment.pop(key, None)
            environment["NO_PROXY"] = environment["no_proxy"] = "localhost,127.0.0.1"
            # An isolated client process keeps native SSL injection out of the
            # rest of the test runner. No machine trust settings are modified.
            script = '''
import json, os, sys
sys.path.insert(0, sys.argv[3])
from ocr_network import configure_tls
configure_tls()
import requests
from urllib.request import urlopen
from urllib.error import URLError
url = sys.argv[1]
checks = {}
try:
    requests.get(url, timeout=5)
except requests.exceptions.SSLError:
    checks['untrusted_requests_rejected'] = True
try:
    urlopen(url, timeout=5)
except URLError:
    checks['untrusted_urllib_rejected'] = True
os.environ['IMAGE_OCR_CA_BUNDLE'] = sys.argv[2]
configure_tls()
checks['approved_requests'] = requests.get(url, timeout=5).status_code == 200
with urlopen(url, timeout=5) as response:
    checks['approved_urllib'] = response.status == 200
try:
    requests.get(url.replace('localhost', '127.0.0.1'), timeout=5)
except requests.exceptions.SSLError:
    checks['wrong_hostname_rejected'] = True
print(json.dumps(checks))
'''
            try:
                result = subprocess.run(
                    [sys.executable, "-c", script, f"https://localhost:{server.server_port}",
                     str(folder / "ca.pem"), str(Path(__file__).resolve().parents[1])],
                    env=environment, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                checks = json.loads(result.stdout)
                self.assertEqual(len(checks), 5)
                self.assertTrue(all(checks.values()), checks)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
