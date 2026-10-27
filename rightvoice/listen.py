"""A static listening page for a built set.

One HTML file with, for a few items spread over the capture values, the item id, its bucket
labels and one player per stem. Plain HTML and CSS: no scripts and no external resources.
Audio is linked by relative paths, so the page works when it sits inside the build folder.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

PAGE_NAME = "listen.html"

STEM_NOTES: tuple[tuple[str, str], ...] = (
    ("mixture", "what a system receives: the conversation as captured"),
    ("sample", "the voice sample of the target, from another chapter of the same reader"),
    ("target", ("the clean target stem, after the room if any, at the mixture's level; "
                "silence when the target is absent")),
    ("partner", "the conversation partner, after the room if any, at the mixture's level"),
    ("others", ("everything else in the mixture: bystanders, noise and music, and what the "
                "device chain changed")),
)

BUCKET_ORDER = ("capture", "overlap", "reentry", "tir_db", "bystanders", "same_sex",
                "sample_s", "sample_device", "target_absent", "long")

CSS = """
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 64rem;
       padding: 1.5rem 1rem; color: #1d1d1f; background: #fafafa; line-height: 1.4; }
h1 { font-size: 1.4rem; margin: 0 0 0.5rem; }
dl.stems { display: grid; grid-template-columns: max-content 1fr; gap: 0.2rem 1rem;
           margin: 0 0 1.5rem; font-size: 0.92rem; }
dl.stems dt { font-weight: 600; }
dl.stems dd { margin: 0; }
section.item { background: #fff; border: 1px solid #ddd; border-radius: 6px;
               padding: 0.8rem 1rem; margin: 0 0 1rem; }
section.item h2 { font-size: 1.05rem; margin: 0 0 0.4rem; font-family: ui-monospace, monospace; }
ul.labels { list-style: none; padding: 0; margin: 0 0 0.6rem; display: flex; flex-wrap: wrap;
            gap: 0.3rem; }
ul.labels li { font-size: 0.8rem; background: #eef1f5; border-radius: 4px;
               padding: 0.1rem 0.45rem; }
div.players { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 0.8rem;
              align-items: center; }
div.players span { font-size: 0.9rem; font-weight: 600; }
audio { width: 100%; max-width: 32rem; }
p.note { font-size: 0.85rem; color: #555; }
"""


def read_manifest(build_dir: Path | str) -> list[dict]:
    path = Path(build_dir) / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose(rows: list[dict], count: int, seed: int = 0) -> list[dict]:
    """Up to ``count`` rows, drawn with ``seed`` and spread over the capture values.

    Rows are grouped by capture value and each group is shuffled; the groups are then visited
    in turn, one row from each, until ``count`` rows are taken. The result is sorted by id.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        groups.setdefault(str(row["buckets"]["capture"]), []).append(row)
    queues = {k: [groups[k][i] for i in rng.permutation(len(groups[k])).tolist()]
              for k in sorted(groups)}
    picked: list[dict] = []
    while len(picked) < count and any(queues.values()):
        for k in sorted(queues):
            if queues[k] and len(picked) < count:
                picked.append(queues[k].pop(0))
    return sorted(picked, key=lambda r: r["id"])


def _label(name: str, value) -> str:
    if isinstance(value, bool):
        value = "yes" if value else "no"
    elif value is None:
        value = "n/a"
    return f"{name.replace('_', ' ')}: {value}"


def _src(build_dir: Path, rel: str, page_dir: Path) -> str:
    target = (build_dir / rel).resolve()
    return quote(Path(os.path.relpath(target, page_dir.resolve())).as_posix())


def render_page(rows: list[dict], build_dir: Path, page_dir: Path, title: str) -> str:
    esc = html.escape
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f"<style>{CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{esc(title)}</h1>",
        (f'<p class="note">{len(rows)} items, spread over the capture values. Every file is '
         "16 kHz mono. Mixture, target, partner and others share one gain, so the mixture is "
         "their sum; the sample has its own level.</p>"),
        '<dl class="stems">',
    ]
    for name, note in STEM_NOTES:
        parts.append(f"<dt>{esc(name)}</dt><dd>{esc(note)}</dd>")
    parts.append("</dl>")
    for row in rows:
        buckets = row.get("buckets", {})
        parts.append(f'<section class="item" id="{esc(row["id"])}">')
        parts.append(f"<h2>{esc(row['id'])}</h2>")
        parts.append('<ul class="labels">')
        for key in BUCKET_ORDER:
            if key in buckets:
                parts.append(f"<li>{esc(_label(key, buckets[key]))}</li>")
        parts.append("</ul>")
        parts.append('<div class="players">')
        for name, _ in STEM_NOTES:
            rel = row.get("files", {}).get(name)
            if rel is None:
                continue
            src = _src(build_dir, rel, page_dir)
            parts.append(f"<span>{esc(name)}</span>"
                         f'<audio controls preload="none" src="{esc(src)}"></audio>')
        parts.append("</div>")
        parts.append("</section>")
    parts += ["</body>", "</html>", ""]
    return "\n".join(parts)


def write_page(build_dir: Path | str, out_html: Path | str | None = None, count: int = 20,
               seed: int = 0) -> Path:
    """Write the listening page and return its path.

    ``out_html`` defaults to ``<build_dir>/listen.html``; audio paths are made relative to
    wherever the page is written.
    """
    build = Path(build_dir)
    page = Path(out_html) if out_html is not None else build / PAGE_NAME
    rows = choose(read_manifest(build), count, seed)
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render_page(rows, build, page.parent, "RightVoice listening page"),
                    encoding="utf-8")
    return page


__all__ = ["choose", "read_manifest", "render_page", "write_page"]
