import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
import controls


def test_controls_link_stems_and_leave_silence_empty(tmp_path):
    build = tmp_path / "b"
    rows = []
    for i in range(3):
        item = build / "items" / f"rv1-{i:05d}"
        item.mkdir(parents=True)
        for name in ("mixture.wav", "target.wav"):
            sf.write(item / name, np.full(1600, 0.1 * (i + 1), dtype=np.float32), 16000)
        rows.append({"id": item.name})
    (build / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    counts = controls.write_controls(build)
    assert counts == {"control_clean": 3, "control_mixture": 3, "control_silence": 0}
    clean = build / "outputs" / "control_clean" / "rv1-00001.wav"
    assert clean.is_symlink()
    x, sr = sf.read(clean)
    assert sr == 16000 and abs(float(x[0]) - 0.2) < 1e-3
    assert not any((build / "outputs" / "control_silence").iterdir())
    # idempotent
    assert controls.write_controls(build)["control_clean"] == 3
