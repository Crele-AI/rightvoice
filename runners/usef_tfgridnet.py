#!/usr/bin/env python3
"""USEF-TFGridNet: target speaker extraction from a mixture and an enrolment sample.

    python runners/usef_tfgridnet.py --fetch
    hf download ZBang/USEF-TSE --local-dir <data_root>/third_party/usef_tfgridnet/chkpt
    python runners/usef_tfgridnet.py --build data/build/v1 [--config F] [--checkpoint F]

Licence: the upstream code is CC BY-NC 4.0, which forbids commercial use. This script never
vendors any of it. ``--fetch`` clones the upstream repository into
``<data_root>/third_party/usef_tfgridnet`` and downloads nothing else; the checkpoint is
downloaded by hand from the weights repository into the clone's ``chkpt/`` folder or passed
with ``--checkpoint``. RightVoice runs it only to report its numbers.

The model is built the way the authors' ``eval.py`` builds it: the published HyperPyYAML config
(``chkpt/USEF-TFGridNet/config.yaml``) constructs ``modules.masknet``, the checkpoint's
``model_state_dict`` is loaded strictly after the same key renames ``eval.py`` applies, and
``model(mixture, enrolment)`` is called on two ``[1, samples]`` tensors. The default checkpoint
is the one published for noisy, reverberant mixtures (``whamr!``); ``--checkpoint`` picks another.
The model's sample rate comes from the config (8 kHz for the published checkpoints); the mixture
and the sample are resampled to it, and the output back to 16 kHz. Files longer than
``--chunk-s`` run in overlapping windows that each get the whole sample. ``--probe`` prints what
the runner reads from the clone and what it would use.
"""
from __future__ import annotations

import argparse
import ast
import functools
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import common
import numpy as np
import yaml

from rightvoice.sources import data_root

SYSTEM = "usef_tfgridnet"
LICENCE = "CC BY-NC 4.0: no commercial use; RightVoice runs it only to report numbers"
# The authors' code and weights.
UPSTREAM_URL = "https://github.com/ZBang/USEF-TSE"
WEIGHTS_URL = "https://huggingface.co/ZBang/USEF-TSE"
CHUNK_S = 30.0
OVERLAP_S = 1.0
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".tar")
RATE_KEYS = ("sample_rate", "sampling_rate", "sr", "fs", "samplerate")
MODEL_KEYS = ("model", "model_args", "model_conf", "model_config", "network", "net", "nnet_conf")
EVAL_LINES = re.compile(
    r"^\s*(import|from)\s|torch\.load|load_state_dict|add_argument|sample_rate|sampling_rate"
    r"|\bsr\b|\bfs\b|config|model\(|\.forward\(|gridnet|usef", re.IGNORECASE)

_LOADED: dict[tuple, dict] = {}


# The upstream clone

def repo_dir() -> Path:
    return data_root() / "third_party" / "usef_tfgridnet"


def fetch(dest: Path | None = None, run: Callable[..., Any] = subprocess.run) -> Path:
    """``git clone UPSTREAM_URL`` into ``dest`` (default ``repo_dir()``), unless it is there."""
    dest = Path(dest) if dest is not None else repo_dir()
    if (dest / ".git").exists():
        print(f"{SYSTEM}: {dest} already holds a clone; leaving it as it is")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", UPSTREAM_URL, str(dest)], check=True)
    print(f"{SYSTEM}: cloned {UPSTREAM_URL} into {dest}. Download the TFGridNet checkpoint "
          f"from {WEIGHTS_URL} into {dest / 'chkpt'} (or pass --checkpoint), then run --probe.")
    return dest


def commit(repo: Path) -> str:
    """The clone's commit, read from ``.git`` without running git."""
    git = Path(repo) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
    except OSError:
        return "unknown"
    if not head.startswith("ref:"):
        return head
    ref = head.split(None, 1)[1]
    try:
        return (git / ref).read_text().strip()
    except OSError:
        pass
    try:
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def need_repo(repo: Path | None) -> Path:
    repo = Path(repo) if repo is not None else repo_dir()
    if not (repo / "models").is_dir():
        raise RuntimeError(f"{SYSTEM} needs the USEF-TFGridNet code from {UPSTREAM_URL} at "
                           f"{repo}, which is not there; run this script with --fetch")
    return repo


# Config, checkpoint and model class

def _pick(found: list[Path], what: str, flag: str) -> Path:
    if len(found) == 1:
        return found[0]
    listed = ", ".join(str(p) for p in found) or "none"
    raise RuntimeError(f"choose the TFGridNet {what} with {flag}; candidates: {listed}")


def config_files(repo: Path) -> list[Path]:
    folder = repo / "config"
    return sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []


def _given(repo: Path, path: Path) -> Path:
    """A path given on the command line, read against the clone when it is not found as is."""
    path = Path(path)
    return path if path.exists() or path.is_absolute() else repo / path


PUBLISHED_CONFIG = Path("chkpt/USEF-TFGridNet/config.yaml")
PUBLISHED_CHECKPOINT = Path("chkpt/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar")


def find_config(repo: Path, config: Path | None = None) -> Path:
    """The config beside the published checkpoints when present, as eval.sh uses it."""
    if config is not None:
        return _given(repo, config)
    if (repo / PUBLISHED_CONFIG).is_file():
        return repo / PUBLISHED_CONFIG
    return _pick([p for p in config_files(repo) if "gridnet" in p.name.lower()],
                 "config", "--config")


def checkpoint_files(repo: Path) -> list[Path]:
    return sorted(p for p in repo.rglob("*")
                  if p.is_file() and p.suffix in CHECKPOINT_SUFFIXES
                  and ".git" not in p.relative_to(repo).parts)


def find_checkpoint(repo: Path, checkpoint: Path | None = None) -> Path:
    """The whamr! checkpoint by default: the published one made for noisy, reverberant mixtures,
    the closest of the three to this benchmark's recordings."""
    if checkpoint is not None:
        return _given(repo, checkpoint)
    if (repo / PUBLISHED_CHECKPOINT).is_file():
        return repo / PUBLISHED_CHECKPOINT
    return _pick([p for p in checkpoint_files(repo)
                  if "gridnet" in str(p.relative_to(repo)).lower()], "checkpoint", "--checkpoint")


def read_config(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    blob = json.loads(text) if Path(path).suffix == ".json" else yaml.safe_load(text)
    return blob if isinstance(blob, dict) else {}


def config_rate(cfg: Any) -> int | None:
    """The first sample rate found in the config, breadth first, under ``RATE_KEYS``."""
    queue = [cfg]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for key in RATE_KEYS:
                value = node.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and value >= 1000:
                    return int(value)
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return None


def model_classes(repo: Path) -> list[dict]:
    """Classes under ``models/`` whose name contains GridNet, read without importing them."""
    found = []
    for path in sorted((repo / "models").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module = ".".join(path.relative_to(repo).with_suffix("").parts)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and "gridnet" in node.name.lower():
                methods = {f.name: [a.arg for a in f.args.args[1:]] for f in node.body
                           if isinstance(f, ast.FunctionDef)}
                found.append({"module": module, "name": node.name,
                              "init": methods.get("__init__"), "forward": methods.get("forward")})
    # A class named for USEF first, then the rest in file order.
    return sorted(found, key=lambda c: "usef" not in c["name"].lower())


def model_kwargs(cfg: dict, cls: type) -> dict:
    """The config's model section, kept to the arguments the class takes."""
    section = next((cfg[k] for k in MODEL_KEYS if isinstance(cfg.get(k), dict)), cfg)
    params = inspect.signature(cls).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(section)
    return {k: v for k, v in section.items() if k in params}


def _to_numpy(out: Any) -> np.ndarray:
    if isinstance(out, dict):
        out = next(v for v in out.values() if hasattr(v, "shape"))
    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    y = np.asarray(out, dtype=np.float32)
    return y.squeeze() if y.ndim > 1 else y.reshape(-1)


def _state_dict(torch: Any, checkpoint: Path, trust_checkpoint: bool) -> dict:
    """The checkpoint's ``model_state_dict`` with the renames eval.py applies."""
    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001  the published file may hold more than tensors
        if not trust_checkpoint:
            print(f"{SYSTEM}: {checkpoint.name} is not plain weights ({type(exc).__name__}); "
                  f"unpickling it as eval.py does", file=sys.stderr)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob
    for name in ("model_state_dict", "state_dict", "model"):
        if isinstance(blob, dict) and isinstance(blob.get(name), dict):
            state = blob[name]
            break
    return {k.replace("module.", "").replace("convolution_", "convolution_module."): v
            for k, v in state.items()}


def load(device: str | None = None, repo: Path | None = None, config: Path | None = None,
         checkpoint: Path | None = None, model_rate: int | None = None,
         trust_checkpoint: bool = False) -> dict:
    """The model as eval.py builds it, loaded once per set of arguments.

    The config is HyperPyYAML: ``modules.masknet`` constructs the network with its STFT and
    attention block from the clone's ``models`` and ``utils`` packages, which is why the clone
    root goes on ``sys.path``. ``sample_rate`` in the same config gives the model's rate.
    """
    key = (device, repo, config, checkpoint, model_rate, trust_checkpoint)
    if key in _LOADED:
        return _LOADED[key]
    repo = need_repo(repo)
    try:
        import torch
        from hyperpyyaml import load_hyperpyyaml
    except ImportError as exc:
        raise RuntimeError(f"{SYSTEM} needs the 'torch' and 'hyperpyyaml' packages, which could "
                           f"not be imported ({exc})") from exc
    config = find_config(repo, config)
    checkpoint = find_checkpoint(repo, checkpoint)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    try:
        os.chdir(repo)  # relative imports inside the clone's yaml resolve from its root
        cfg = load_hyperpyyaml(config.read_text(encoding="utf-8"))
    finally:
        os.chdir(cwd)
    rate = model_rate or cfg.get("sample_rate") or config_rate(cfg)
    if rate is None:
        raise RuntimeError(f"no sample rate found in {config}; pass --model-rate")
    try:
        model = cfg["modules"]["masknet"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"{config} has no modules.masknet entry; read eval.py") from exc
    model.load_state_dict(_state_dict(torch, checkpoint, trust_checkpoint), strict=True)
    where = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(where).eval()
    _LOADED[key] = {"torch": torch, "model": model, "rate": int(rate), "device": where,
                    "repo": repo, "config": config, "checkpoint": checkpoint,
                    "class": f"{type(model).__module__}.{type(model).__name__}"}
    return _LOADED[key]


def process(mixture: np.ndarray, sample: np.ndarray, row: dict, *, device: str | None = None,
            repo: Path | None = None, config: Path | None = None,
            checkpoint: Path | None = None, model_rate: int | None = None,
            trust_checkpoint: bool = False, chunk_s: float = CHUNK_S) -> np.ndarray:
    """The target's speech extracted from the mixture, at 16 kHz."""
    m = load(device, repo, config, checkpoint, model_rate, trust_checkpoint)
    torch, model, rate = m["torch"], m["model"], m["rate"]
    x = common.resample(mixture, common.RATE, rate)
    enrolment = torch.from_numpy(common.resample(sample, common.RATE, rate))[None].to(m["device"])

    def run(segment: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            seg = torch.from_numpy(np.ascontiguousarray(segment))[None].to(m["device"])
            return _to_numpy(model(seg, enrolment))

    return common.resample(common.in_chunks(run, x, rate, chunk_s, OVERLAP_S), rate, common.RATE)


def info(m: dict, chunk_s: float) -> dict:
    return {"package": "USEF-TSE, upstream clone", "version": commit(m["repo"]),
            "code": UPSTREAM_URL, "weights": f"{m['checkpoint']} (from {WEIGHTS_URL})",
            "config": str(m["config"]), "class": m["class"], "model_rate": m["rate"],
            "device": m["device"], "chunk_s": chunk_s, "torch": m["torch"].__version__,
            "licence": LICENCE}


# Probe

def probe(repo: Path | None, config: Path | None, checkpoint: Path | None) -> int:
    """Print what this adapter reads from the clone, without importing it."""
    repo = need_repo(repo)
    print(f"clone: {repo} at commit {commit(repo)}")
    print("top level: " + ", ".join(sorted(p.name for p in repo.iterdir())))
    for name in ("eval.sh", "eval.py"):
        path = repo / name
        if not path.is_file():
            print(f"{name}: missing")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"{name} ({len(lines)} lines):")
        for n, line in enumerate(lines, start=1):
            if name == "eval.sh" or EVAL_LINES.search(line):
                print(f"  {n:4d}  {line}")
    print("config files:")
    for path in config_files(repo):
        try:
            cfg = read_config(path)
            print(f"  {path.relative_to(repo)}: rate {config_rate(cfg)}, keys {sorted(cfg)}")
        except (ValueError, yaml.YAMLError) as exc:
            print(f"  {path.relative_to(repo)}: unreadable ({exc})")
    print("model classes:")
    for c in model_classes(repo):
        print(f"  {c['module']}.{c['name']}: __init__{c['init']} forward{c['forward']}")
    print("checkpoint files:")
    for path in checkpoint_files(repo):
        print(f"  {path.relative_to(repo)} ({path.stat().st_size / 1e6:.1f} MB)")
    for what, choose in (("config", lambda: find_config(repo, config)),
                         ("checkpoint", lambda: find_checkpoint(repo, checkpoint))):
        try:
            print(f"would use {what}: {choose()}")
        except RuntimeError as exc:
            print(f"would use {what}: none, {exc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_arguments(parser, build_required=False)
    parser.add_argument("--fetch", action="store_true", help=f"clone {UPSTREAM_URL} and stop")
    parser.add_argument("--probe", action="store_true",
                        help="print what the adapter reads from the clone, then stop")
    parser.add_argument("--repo", type=Path, default=None,
                        help="the clone (default <data_root>/third_party/usef_tfgridnet)")
    parser.add_argument("--config", type=Path, default=None, help="the model's file in config/")
    parser.add_argument("--checkpoint", type=Path, default=None, help="the model's weights")
    parser.add_argument("--model-rate", type=int, default=None,
                        help="the model's sample rate, when the config does not say")
    parser.add_argument("--chunk-s", type=float, default=CHUNK_S,
                        help="longer files run in overlapping windows of this many seconds")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="unpickle a checkpoint that is not plain weights")
    args = parser.parse_args(argv)
    try:
        if args.fetch:
            fetch(args.repo)
            return 0
        if args.probe:
            return probe(args.repo, args.config, args.checkpoint)
        if args.build is None:
            parser.error("--build is required unless --fetch or --probe is given")
        m = load(args.device, args.repo, args.config, args.checkpoint, args.model_rate,
                 args.trust_checkpoint)  # a missing clone or package stops the run here
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 2
    head = info(m, args.chunk_s)
    print(f"{SYSTEM}: {head}")
    process_item = functools.partial(
        process, device=args.device, repo=args.repo, config=args.config,
        checkpoint=args.checkpoint, model_rate=args.model_rate,
        trust_checkpoint=args.trust_checkpoint, chunk_s=args.chunk_s)
    counts = common.run_system(args.build, SYSTEM, process_item, only=args.only, quick=args.quick,
                               limit=args.limit, resume=not args.no_resume, info=head)
    return common.finish(SYSTEM, counts, args.build)


if __name__ == "__main__":
    sys.exit(main())
