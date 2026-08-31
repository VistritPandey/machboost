from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from machboost.relay import LoopbackRelayServer


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self.server.authorization = self.headers.get("Authorization")
        body = json.dumps({"data": [{"id": "shared-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.server.authorization = self.headers.get("Authorization")
        self.server.request_body = self.rfile.read(length)
        chunks = (b'{"message":{"content":"hel"}}\n', b'{"message":{"content":"lo"}}\n')
        body = b"".join(chunks)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        return


class ClaudeLoopbackRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        cls.relay = LoopbackRelayServer(
            ("127.0.0.1", 0),
            upstream=f"http://127.0.0.1:{cls.upstream.server_port}",
            upstream_token="studio-secret",
            local_token="claude-local-secret",
        )
        cls.relay_thread = threading.Thread(target=cls.relay.serve_forever, daemon=True)
        cls.relay_thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.relay.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.relay.shutdown()
        cls.relay.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def test_health_is_available_without_credentials(self):
        with urlopen(self.endpoint + "/health") as response:
            self.assertEqual(response.status, 200)

    def test_gateway_requires_its_private_local_token(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.endpoint + "/v1/models")
        self.assertEqual(error.exception.code, 401)

    def test_gateway_exchanges_local_credentials_for_host_credentials(self):
        request = Request(
            self.endpoint + "/v1/models",
            headers={"Authorization": "Bearer claude-local-secret"},
        )
        with urlopen(request) as response:
            payload = json.load(response)

        self.assertEqual(payload["data"][0]["id"], "shared-model")
        self.assertEqual(self.upstream.authorization, "Bearer studio-secret")

    def test_post_body_and_streamed_response_are_forwarded(self):
        body = b'{"model":"shared-model","stream":true}'
        request = Request(
            self.endpoint + "/api/chat",
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer claude-local-secret",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request) as response:
            received = response.read()

        self.assertEqual(self.upstream.request_body, body)
        self.assertEqual(received.count(b'"content"'), 2)


if __name__ == "__main__":
    unittest.main()
