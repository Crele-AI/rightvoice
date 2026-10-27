"""Pinned sources: where each comes from, its licence, and how it is verified.

Every source is fetched from its owner, checked against a sha256 and recorded. Nothing is
downloaded until the operator runs ``rightvoice sources fetch``; ``rightvoice sources plan``
prints every URL and size first. A source whose sha256 is still ``None`` is fetched only with
``--pin``, which records the digest of what arrived for review.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_ENV = "RIGHTVOICE_DATA"
USER_AGENT = "rightvoice-sources/0.1 (benchmark build; +https://www.crele.co/benchmarks/rightvoice)"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ENV, "data")).expanduser()


@dataclass(frozen=True)
class Source:
    name: str
    group: str
    url: str
    size_bytes: int           # approximate until pinned; the fetch records the real size
    size_note: str            # where the size estimate came from
    licence: str
    licence_url: str
    sha256: str | None = None  # None until the first verified fetch pins it
    note: str = ""

    @property
    def archive_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


REGISTRY: tuple[Source, ...] = (
    *[
        Source(
            name=f"libritts-r-{part.replace('_', '-')}",
            group="libritts-r",
            url=f"https://www.openslr.org/resources/141/{part}.tar.gz",
            size_bytes=size,
            size_note="Content-Length on 2026-10-22",
            licence="CC BY 4.0",
            licence_url="https://creativecommons.org/licenses/by/4.0/",
            sha256=sha,
            note="24 kHz wav with normalized text per utterance; resampled to 16 kHz by the build. "
                 "Pinned 2026-10-22 after the archive matched OpenSLR's md5sum.txt.",
        )
        for part, size, sha in (
            ("dev_clean", 1_357_352_348, "02ea52de47ca670d7ad86e714a6c58b207660a3f8a371ad102156776d2262a0c"),
            ("test_clean", 1_294_768_623, "d4a2a7cdeeb68a6cfd628559739879bcb29556364cc13f38413930d59369a7f1"),
            ("dev_other", 975_450_334, "17addecd7a58d68d4b22284aae64b81ca81a60317a3e4f7705e35feac37194f8"),
            ("test_other", 1_011_954_678, "a87cc23789526fe93237cd71f0781607f806bb110601acfdb27047617ab266e4"),
        )
    ],
    Source(
        name="libritts-r-doc",
        group="libritts-r",
        url="https://www.openslr.org/resources/141/doc.tar.gz",
        size_bytes=318_318,
        size_note="Content-Length on 2026-10-22",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        sha256="b59b03462911e559a4ab84f7229752b3578b715d534a31d17b88054eb3cc7389",
        note="speakers.tsv (reader, sex, subset), chapter and book tables, licence and readme. "
             "Pinned 2026-10-22 after matching OpenSLR's md5sum.txt.",
    ),
    Source(
        name="ami-annotations",
        group="ami-annotations",
        url="https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/resolve/main/annotations/ami_public_manual_1.6.2.zip",
        size_bytes=22887865,
        size_note="Content-Length on 2026-10-22",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        sha256="56e29112cd89cfba76fbf8f10215eb245cc89b0d3d27355ba14b406a7ea7776c",
        note="AMI public manual annotations 1.6.2: segments and words XML, timing only, no AMI "
             "audio. The archive is the one the AMI Consortium publishes at "
             "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip; "
             "that server did not answer on 2026-10-22, so this copy comes from a public mirror "
             "that redistributes it unchanged under the same licence.",
    ),
    Source(
        name="fsd50k-eval-audio",
        group="fsd50k",
        url="https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip",
        size_bytes=3_037_675_767,
        size_note="Content-Length on 2026-10-22",
        licence="per clip: CC0 1.0 or CC BY 3.0/4.0 kept; BY-NC and Sampling+ dropped",
        licence_url="https://zenodo.org/records/4060432",
        sha256="e3d78e77115ff8d8df1ff3dae7ca5cd5864924c054561f61ba836764254bf50e",
        note="10,231 clips; the licence filter reads the metadata archive below",
    ),
    Source(
        name="fsd50k-eval-audio-part1",
        group="fsd50k",
        url="https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01",
        size_bytes=3_221_225_472,
        size_note="Content-Length on 2026-10-22",
        licence="per clip, as above",
        licence_url="https://zenodo.org/records/4060432",
        sha256="a1b776a5a9466a0cf17d0cb9a2a98f641454084f0292f511718aacb48e6c401d",
        note="first part of the split zip; FSD50K.eval_audio.zip is the last part. "
             "Join with: zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio.joined.zip",
    ),
    Source(
        name="fsd50k-ground-truth",
        group="fsd50k",
        url="https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip",
        size_bytes=334_701,
        size_note="Content-Length on 2026-10-22",
        licence="CC BY 4.0 (labels)",
        licence_url="https://zenodo.org/records/4060432",
        sha256="db2396260a7b1fb06feb6ccac71685794fc89917d71c3a8d64a1de85c2cc0ebf",
    ),
    Source(
        name="fsd50k-metadata",
        group="fsd50k",
        url="https://zenodo.org/records/4060432/files/FSD50K.metadata.zip",
        size_bytes=6_700_838,
        size_note="Content-Length on 2026-10-22",
        licence="CC BY 4.0 (metadata)",
        licence_url="https://zenodo.org/records/4060432",
        sha256="9a738e032546f9a2c6e3d04566928d04a65fb79b422cc9d78bb781723537bd19",
        note="eval_clips_info_FSD50K.json carries each clip's licence URL and uploader",
    ),
    Source(
        name="but-reverbdb-rir",
        group="but-reverbdb",
        url="https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
        size_bytes=9_308_593_693,
        size_note="Content-Length on 2026-10-22",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        sha256="d0d14cdd626b4b05401027c5a774b4f24369f4ea38f57480c6fc6e04e9f3f5b5",
        note="only */RIR/* members and microphone metadata are kept",
    ),
)

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(s.group for s in REGISTRY))

# The LibriTTS-R test and dev readers RightVoice uses. Fixed, and intersected with what the
# archives hold at build time; readers outside this list are never placed in an item.
SPEAKERS: frozenset[str] = frozenset((
    "61", "84", "116", "260", "367", "533", "672", "700", "1089", "1188", "1221", "1255",
    "1272", "1284", "1585", "1630", "1650", "1651", "1686", "1688", "1701", "1998", "2033", "2086",
    "2094", "2277", "2414", "2506", "2609", "2830", "2902", "3005", "3080", "3331", "3528", "3538",
    "3575", "3576", "3660", "3663", "3764", "3915", "3997", "4153", "4198", "4294", "4323", "4350",
    "4507", "4515", "4570", "4572", "4831", "4852", "4970", "5105", "5442", "5484", "5543", "5639",
    "5764", "5849", "6070", "6123", "6128", "6241", "6267", "6432", "6455", "6467", "6599", "6841",
    "6938", "7018", "7021", "7105", "7127", "7601", "7641", "7697", "7729", "7902", "7975", "8131",
    "8173", "8188", "8230", "8254", "8280", "8288", "8461",
))

# FSD50K licence URLs that may be used. Everything else, including BY-NC and Sampling+, is dropped.
FSD50K_ALLOWED_LICENCES: tuple[str, ...] = (
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "https://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/4.0/",
)


def fsd50k_allowed_clips(clips_info: dict[str, dict]) -> dict[str, dict]:
    """Keep the clips whose licence is CC0 or CC BY, from FSD50K's clips_info JSON."""
    allowed = {u.rstrip("/") for u in FSD50K_ALLOWED_LICENCES}
    return {
        clip_id: info
        for clip_id, info in clips_info.items()
        if str(info.get("license", "")).strip().rstrip("/") in allowed
    }


def select(groups: Iterable[str] | None = None) -> list[Source]:
    """The sources of the given groups, in the order the groups were given."""
    order = list(dict.fromkeys(groups or GROUPS))
    if "all" in order:
        order = list(GROUPS)
    unknown = set(order) - set(GROUPS)
    if unknown:
        raise KeyError(f"unknown source group(s) {sorted(unknown)}; known: {GROUPS}")
    return [s for group in order for s in REGISTRY if s.group == group]


def human(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def plan(groups: Iterable[str] | None = None, out=sys.stdout) -> None:
    sources = select(groups)
    total = sum(s.size_bytes for s in sources)
    print(f"{'source':28} {'size':>9}  licence                      url", file=out)
    for s in sources:
        pinned = "pinned" if s.sha256 else "unpinned"
        print(f"{s.name:28} {human(s.size_bytes):>9}  {s.licence[:28]:28} {s.url}  [{pinned}, size {s.size_note}]", file=out)
    print(f"{'total':28} {human(total):>9}  into {data_root() / 'sources'}", file=out)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


RETRIES = 6
TIMEOUT_S = 60


class IncompleteDownload(OSError):
    """The connection closed before the announced length arrived."""


def _expected_total(response, have: int) -> int | None:
    """Total length of the resource from Content-Range, else Content-Length plus the offset."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = response.headers.get("Content-Length")
    return have + int(length) if length and length.isdigit() else None


def _download(url: str, target: Path, expected: int | None = None, *, retries: int = RETRIES) -> None:
    """Stream ``url`` into ``target``, resuming a ``.part`` file with a Range request.

    Retries on network errors with a growing pause. A server that ignores Range answers 200
    and the download restarts from zero.
    """
    partial = target.with_suffix(target.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        have = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                status = getattr(response, "status", 200)
                if have and status == 206:
                    mode = "ab"
                elif status == 416:
                    mode = None
                else:
                    mode, have = "wb", 0
                if mode is not None:
                    expected = _expected_total(response, have)
                    with partial.open(mode) as fh:
                        for chunk in iter(lambda: response.read(1 << 20), b""):
                            fh.write(chunk)
                    got = partial.stat().st_size
                    if expected is not None and got < expected:
                        raise IncompleteDownload(f"{got} of {expected} bytes arrived")
            partial.rename(target)
            return
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and partial.exists():
                partial.rename(target)
                return
            if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code != 416:
                raise
            print(f"fetch: attempt {attempt}/{retries} failed for {url}: {exc}", file=sys.stderr)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"gave up after {retries} attempts: {url}") from last


PARALLEL_STREAMS = 6
CHUNK_BYTES = 64 << 20


def _head(url: str) -> tuple[int | None, bool]:
    """(total length, server honours Range) from a one-byte ranged GET.

    Some servers answer Range requests without advertising Accept-Ranges on a HEAD, so the
    probe asks for the first byte and reads the total from Content-Range.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        status = getattr(response, "status", 200)
        content_range = response.headers.get("Content-Range", "")
        length = response.headers.get("Content-Length")
        response.read()
    if status == 206 and "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
        return int(content_range.rsplit("/", 1)[1]), True
    return (int(length) if length and length.isdigit() else None), False


def _fetch_range(url: str, start: int, end: int) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        if getattr(response, "status", 200) != 206:
            raise IncompleteDownload("server ignored the Range request")
        data = response.read()
    if len(data) != end - start + 1:
        raise IncompleteDownload(f"range {start}-{end}: {len(data)} bytes")
    return data


def _download_parallel(url: str, target: Path, total: int, *, streams: int = PARALLEL_STREAMS,
                       chunk: int = CHUNK_BYTES, retries: int = RETRIES) -> None:
    """Fetch ``url`` as fixed chunks over several connections into a preallocated ``.part``.

    Finished chunks are recorded in ``<target>.chunks`` so an interrupted download resumes.
    Servers that cap a single stream often allow a higher total this way.
    """
    partial = target.with_suffix(target.suffix + ".part")
    state = target.with_suffix(target.suffix + ".chunks")
    n_chunks = (total + chunk - 1) // chunk
    done: set[int] = set()
    if state.exists() and partial.exists() and partial.stat().st_size == total:
        done = set(json.loads(state.read_text()))
    else:
        with partial.open("wb") as fh:
            fh.truncate(total)
        state.write_text("[]")
    todo = [i for i in range(n_chunks) if i not in done]
    lock = threading.Lock()
    failures: list[Exception] = []

    def worker() -> None:
        while True:
            with lock:
                if not todo or failures:
                    return
                index = todo.pop(0)
            start, end = index * chunk, min(total, (index + 1) * chunk) - 1
            for attempt in range(1, retries + 1):
                try:
                    data = _fetch_range(url, start, end)
                    break
                except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
                    if attempt == retries:
                        with lock:
                            failures.append(exc)
                        return
                    time.sleep(min(60, 5 * attempt))
            with lock:
                fd = os.open(partial, os.O_WRONLY)
                try:
                    os.pwrite(fd, data, start)
                finally:
                    os.close(fd)
                done.add(index)
                state.write_text(json.dumps(sorted(done)))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(streams, len(todo)) or 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise RuntimeError(f"parallel download failed: {failures[0]}") from failures[0]
    if len(done) != n_chunks:
        raise IncompleteDownload(f"{len(done)} of {n_chunks} chunks")
    state.unlink()
    partial.rename(target)


def fetch(source: Source, root: Path | None = None, *, pin: bool = False, parallel: bool = False) -> Path:
    """Download one source into ``<root>/sources/archives``, verify, and record it."""
    root = root or data_root()
    if source.sha256 is None and not pin:
        raise RuntimeError(
            f"{source.name} has no pinned sha256; run with --pin to fetch and record the digest")
    archives = root / "sources" / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    target = archives / source.archive_name
    if not target.exists():
        total, ranges = (None, False)
        if parallel:
            total, ranges = _head(source.url)
        if parallel and total and ranges:
            _download_parallel(source.url, target, total)
        else:
            _download(source.url, target)
    digest = sha256_of(target)
    if source.sha256 is not None and digest != source.sha256:
        raise RuntimeError(f"{source.name}: sha256 {digest} does not match pinned {source.sha256}")
    record = {**asdict(source), "sha256_fetched": digest, "size_fetched": target.stat().st_size}
    pins = root / "sources" / "pins.json"
    existing = json.loads(pins.read_text()) if pins.exists() else {}
    existing[source.name] = record
    pins.write_text(json.dumps(existing, indent=1, sort_keys=True))
    return target
