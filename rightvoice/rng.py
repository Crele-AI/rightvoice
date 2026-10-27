"""Independent random streams per (seed, item index, stage).

Every item is a pure function of ``(seed, index)``. Each build stage draws from its own
stream, so changing how one stage draws never moves another stage's draws.
"""
from __future__ import annotations

import numpy as np

STAGES: tuple[str, ...] = (
    "conversation",  # which timeline, which readers
    "utterances",    # which utterance fills each turn
    "levels",        # partner gain, mixture level
    "rooms",         # which response, which positions
    "noise",         # which clip, SNR
    "music",         # whether music, which clip
    "bystanders",    # count, clips, placement, playback chain
    "chain",         # device stages and their parameters
    "sample",        # which stretch becomes the voice sample
    "absent",        # whether the target is removed
    "selection",     # quota-then-fill order
)


def stage_id(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError as exc:
        raise KeyError(f"unknown stage {stage!r}; one of {STAGES}") from exc


def stream(seed: int, index: int, stage: str) -> np.random.Generator:
    """A generator that depends only on ``(seed, index, stage)``."""
    if seed < 0 or index < 0:
        raise ValueError("seed and index must be non-negative")
    sequence = np.random.SeedSequence([int(seed), int(index), stage_id(stage)])
    return np.random.default_rng(sequence)
