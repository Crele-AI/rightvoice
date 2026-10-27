import io

import pytest

from rightvoice import sources


def test_registry_groups_cover_every_source():
    assert {s.group for s in sources.REGISTRY} == set(sources.GROUPS)
    assert len({s.name for s in sources.REGISTRY}) == len(sources.REGISTRY)


def test_every_source_has_licence_and_https():
    for s in sources.REGISTRY:
        assert s.url.startswith("https://"), s.name
        assert s.licence and s.licence_url, s.name


def test_select_all_and_unknown():
    assert sources.select(["all"]) == list(sources.REGISTRY)
    ordered = sources.select(["but-reverbdb", "libritts-r"])
    assert ordered[0].group == "but-reverbdb" and ordered[-1].group == "libritts-r"
    with pytest.raises(KeyError):
        sources.select(["nope"])


def test_plan_prints_without_fetching(tmp_path, monkeypatch):
    monkeypatch.setenv(sources.DATA_ENV, str(tmp_path))
    out = io.StringIO()
    sources.plan(["libritts-r"], out=out)
    text = out.getvalue()
    assert "openslr.org/resources/141" in text
    assert "total" in text
    assert not list(tmp_path.iterdir())


def test_unpinned_fetch_refused_without_pin(tmp_path):
    s = sources.Source(name="x", group="x", url="https://example.invalid/x.zip", size_bytes=1,
                       size_note="test", licence="CC0", licence_url="https://example.invalid/")
    with pytest.raises(RuntimeError):
        sources.fetch(s, tmp_path)
    assert not (tmp_path / "sources").exists()


def test_speakers_list_is_fixed_and_sane():
    assert len(sources.SPEAKERS) == 91
    assert all(s.isdigit() for s in sources.SPEAKERS)


def test_fsd50k_licence_filter():
    info = {
        "1": {"license": "http://creativecommons.org/publicdomain/zero/1.0/"},
        "2": {"license": "http://creativecommons.org/licenses/by/4.0/"},
        "3": {"license": "http://creativecommons.org/licenses/by/3.0/"},
        "4": {"license": "http://creativecommons.org/licenses/by-nc/3.0/"},
        "5": {"license": "http://creativecommons.org/licenses/sampling+/1.0/"},
        "6": {},
    }
    kept = sources.fsd50k_allowed_clips(info)
    assert set(kept) == {"1", "2", "3"}


def test_download_resumes_a_partial_file(tmp_path):
    """A Range-aware local server; the first attempt is cut short and the second resumes."""
    import http.server
    import threading

    payload = bytes(range(256)) * 400  # 102,400 bytes
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            calls.append(rng)
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                body = payload[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
            else:
                body, start = payload, 0
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if start == 0:
                self.wfile.write(body[:30_000])  # cut the first transfer short
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/blob.bin"
        target = tmp_path / "blob.bin"
        sources.time.sleep = lambda *_: None  # no waiting between attempts in the test
        sources._download(url, target, retries=4)
    finally:
        server.shutdown()
    assert target.read_bytes() == payload
    assert calls[0] is None and any(c and c.startswith("bytes=") for c in calls[1:])


def test_parallel_download_with_resume(tmp_path):
    import http.server
    import threading

    payload = bytes(range(256)) * 2000  # 512,000 bytes
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            rng = self.headers["Range"].split("=")[1]
            a, b = (int(x) for x in rng.split("-"))
            hits.append((a, b))
            body = payload[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/blob.bin"
        target = tmp_path / "blob.bin"
        total, ranges = sources._head(url)
        assert (total, ranges) == (len(payload), True)
        # pretend two chunks are already done from an earlier run
        partial = target.with_suffix(".bin.part")
        with partial.open("wb") as fh:
            fh.truncate(total)
            fh.seek(0)
            fh.write(payload[:200_000])
        target.with_suffix(".bin.chunks").write_text("[0, 1]")
        sources._download_parallel(url, target, total, streams=3, chunk=100_000)
    finally:
        server.shutdown()
    assert target.read_bytes() == payload
    assert not target.with_suffix(".bin.chunks").exists()
    chunks = [(a, b) for a, b in hits if (a, b) != (0, 0)]  # the probe asks for byte 0 alone
    assert min(a for a, _ in chunks) == 200_000 and len(chunks) == 4
