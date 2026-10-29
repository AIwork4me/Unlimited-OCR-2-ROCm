"""Exercise acquisition against a real local HTTP server, without model downloads."""

import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def downloader():
    path = Path(__file__).parents[1] / "scripts/g05_acquire_datasets.py"
    assert path.exists(), "dataset downloader has not been implemented"
    spec = importlib.util.spec_from_file_location("g05_acquire", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def endpoint():
    body = b"original dataset bytes\x00" * 30
    ranges = []
    batch_attempts = []
    auth_headers = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            batch_attempts.append(self.path)
            if self.path == "/batch-retry" and len(batch_attempts) == 1:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            objects = request["objects"]
            if self.path == "/batch-wrong":
                objects[0]["oid"] = "b" * 64
            for item in objects:
                item["actions"] = {
                    "download": {"href": "https://us.aws.cdn.hf.co/exact-object?signature=secret"}
                }
            payload = json.dumps({"objects": objects}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            auth_headers.append(self.headers.get("Authorization"))
            if self.path == "/redirect":
                if self.headers.get("Authorization") != "Bearer synthetic-test-token":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(302)
                self.send_header(
                    "Location", f"http://localhost:{self.server.server_port}/redirected"
                )
                self.end_headers()
                return
            if self.path == "/redirected" and self.headers.get("Authorization") is not None:
                self.send_response(400)
                self.end_headers()
                return
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1][:-1])
            ranges.append(start)
            if self.path == "/ignored":
                start = 0
            self.send_response(206 if start or self.path == "/wrong-range" else 200)
            if self.path == "/wrong-range":
                self.send_header("Content-Range", "bytes 1-2/3")
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            payload = body[start:]
            if self.path == "/short":
                payload = payload[:-1]
            elif self.path == "/overlong":
                payload += b"x"
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", body, ranges
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("route", ["/range", "/ignored"])
def test_resume_when_range_is_honored_or_ignored(tmp_path, endpoint, route):
    module = downloader()
    base, body, ranges = endpoint
    dest = tmp_path / "page.png"
    dest.with_suffix(".png.part").write_bytes(body[:19])
    module.download_file(base + route, dest, len(body), hashlib.sha256(body).hexdigest(), 1)
    assert dest.read_bytes() == body
    assert ranges == [19]
    assert not dest.with_suffix(".png.part").exists()


@pytest.mark.parametrize(
    ("route", "error"),
    [
        ("/wrong-range", "unexpected Content-Range"),
        ("/short", "size mismatch"),
        ("/overlong", "payload exceeds expected size"),
    ],
)
def test_protocol_failures_never_publish_a_page(tmp_path, endpoint, route, error):
    module = downloader()
    base, body, _ = endpoint
    dest = tmp_path / "page.png"
    with pytest.raises(ValueError, match=error):
        module.download_file(base + route, dest, len(body), hashlib.sha256(body).hexdigest(), 1)
    assert not dest.exists()


def test_lfs_hashes_are_bound_to_frozen_document_manifest():
    module = downloader()
    paths = ["images/doc/page_1.png"]
    gt = [{"page_info": {"image_path": "doc.pdf", "images_list": paths}}]
    canonical = lambda value: hashlib.sha256(  # noqa: E731
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    rows = {
        "doc": {
            "pages": 1,
            "annotation_sha256": canonical(gt[0]),
            "ordered_paths_sha256": canonical(paths),
            "published_page_hashes_sha256": canonical(["a" * 64]),
        }
    }
    with pytest.raises(ValueError, match="published LFS hashes"):
        module.validate_frozen_scope(gt, {paths[0]: {"lfs": {"sha256": "b" * 64}}}, rows)
    module.validate_frozen_scope(gt, {paths[0]: {"lfs": {"sha256": "a" * 64}}}, rows)


def test_hash_mismatch_never_publishes_page(tmp_path, endpoint):
    module = downloader()
    base, body, _ = endpoint
    dest = tmp_path / "page.png"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        module.download_file(base, dest, len(body), "0" * 64, 1)
    assert not dest.exists()


def test_existing_corruption_is_replaced_only_with_verified_bytes(tmp_path, endpoint):
    module = downloader()
    base, body, _ = endpoint
    dest = tmp_path / "page.png"
    dest.write_bytes(b"x" * len(body))
    module.download_file(base, dest, len(body), hashlib.sha256(body).hexdigest(), 1)
    assert dest.read_bytes() == body


def test_lfs_batch_rejects_objects_outside_frozen_oid_set(endpoint):
    module = downloader()
    base, _, _ = endpoint
    pages = [{"rfilename": "images/doc/page_1.png", "lfs": {"sha256": "a" * 64, "size": 4}}]
    with pytest.raises(ValueError, match="batch identity mismatch"):
        module.batch_download_urls(pages, base + "/batch-wrong")
    urls = module.batch_download_urls(pages, base + "/batch-valid")
    assert urls[pages[0]["rfilename"]].startswith("https://us.aws.cdn.hf.co/exact-object?")


def test_lfs_batch_recovers_from_transient_http_failure(endpoint):
    module = downloader()
    base, _, _ = endpoint
    pages = [{"rfilename": "images/doc/page_1.png", "lfs": {"sha256": "a" * 64, "size": 4}}]
    result = module.batch_download_urls(pages, base + "/batch-retry", attempts=2)
    assert set(result) == {"images/doc/page_1.png"}


def test_authorization_is_not_forwarded_to_redirect_download_host(tmp_path, endpoint):
    module = downloader()
    base, body, _ = endpoint
    dest = tmp_path / "page.png"
    module.download_file(
        base + "/redirect",
        dest,
        len(body),
        hashlib.sha256(body).hexdigest(),
        1,
        token="synthetic-test-token",
    )
    assert dest.read_bytes() == body
