#!/usr/bin/env python3
"""DeepFilterNet3: a general speech enhancer that keeps every voice.

It removes noise and reverberation but has no idea which voice to keep, so it ignores the voice
sample by design. It is the reference for what plain enhancement does on RightVoice: the
partner's and the bystanders' words come through along with the target's.

    python runners/deepfilternet3.py --build data/build/v1 [--limit N] [--only clips|long]

Needs the ``deepfilternet`` package (imported as ``df``) and ``torch``. ``df.enhance.init_df()``
loads its default model, DeepFilterNet3, and downloads the weights into the package's cache on
first use. The model works at 48 kHz, so each mixture is resampled from 16 kHz to 48 kHz,
enhanced and resampled back. The package picks its own device (CUDA when it is available);
``--device cpu`` hides the GPU from it before it loads.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any

import common
import numpy as np

SYSTEM = "deepfilternet3"
PACKAGE = "deepfilternet"
WEIGHTS = "DeepFilterNet3, the default model of df.enhance.init_df, fetched by the package"
MODEL_RATE = 48000

_LOADED: dict[str, tuple[Any, Any, Any, Any]] = {}


def load(device: str | None = None) -> tuple[Any, Any, Any, Any]:
    """``(torch, enhance, model, df_state)``, loaded once per device."""
    key = device or "auto"
    if key not in _LOADED:
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import torch
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise RuntimeError(
                f"{SYSTEM} needs the '{PACKAGE}' package (imported as df) and torch, which "
                f"could not be imported ({exc}); install them where the runner runs") from exc
        loaded = init_df()  # (model, df_state, suffix) or, in later versions, with the epoch too
        _LOADED[key] = (torch, enhance, loaded[0], loaded[1])
    return _LOADED[key]


def process(mixture: np.ndarray, sample: np.ndarray, row: dict, *,
            device: str | None = None) -> np.ndarray:
    """The enhanced mixture at 16 kHz. ``sample`` and ``row`` are not used."""
    torch, enhance, model, df_state = load(device)
    rate = int(df_state.sr()) if hasattr(df_state, "sr") else MODEL_RATE
    x = common.resample(mixture, common.RATE, rate)
    y = enhance(model, df_state, torch.from_numpy(x)[None])
    y = np.asarray(y.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
    return common.resample(y, rate, common.RATE)


def info(device: str | None) -> dict:
    torch, _, model, df_state = load(device)
    try:
        where = str(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        where = "unknown"
    return {"package": PACKAGE, "version": common.package_version(PACKAGE), "weights": WEIGHTS,
            "model_rate": int(df_state.sr()) if hasattr(df_state, "sr") else MODEL_RATE,
            "device": where, "torch": torch.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        head = info(args.device)  # loads the model, so a missing package stops the run here
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{SYSTEM}: {head}")
    counts = common.run_system(args.build, SYSTEM, functools.partial(process, device=args.device),
                               only=args.only, quick=args.quick, limit=args.limit,
                               resume=not args.no_resume, info=head)
    return common.finish(SYSTEM, counts, args.build)


if __name__ == "__main__":
    sys.exit(main())
