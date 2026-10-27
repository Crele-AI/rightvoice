import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
import common

RATE = 16000


def _script(name):
    # Loaded under another module name, so a runner never stands in for the package it wraps.
    spec = importlib.util.spec_from_file_location(f"runner_{name}", RUNNERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tone(seconds, rate=RATE, freq=220.0):
    t = np.arange(round(seconds * rate)) / rate
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def build(tmp_path):
    root = tmp_path / "build"
    rows = []
    for i, long in enumerate((False, False, False, True)):
        item_id = f"rv1-{i:05d}"
        folder = root / "items" / item_id
        folder.mkdir(parents=True)
        sf.write(folder / "mixture.wav", _tone(2.0 if long else 1.0), RATE, subtype="PCM_16")
        sf.write(folder / "sample.wav", _tone(0.5, freq=330.0), RATE, subtype="PCM_16")
        rows.append({"id": item_id, "long": long,
                     "files": {"mixture": f"items/{item_id}/mixture.wav",
                               "sample": f"items/{item_id}/sample.wav"}})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _log(build, system):
    path = build / "outputs" / system / common.RUNS_LOG
    return [json.loads(line) for line in path.read_text().splitlines()]


# common

def test_iter_items_filters_and_limits(build):
    items = list(common.iter_items(build))
    assert [i[0] for i in items] == ["rv1-00000", "rv1-00001", "rv1-00002", "rv1-00003"]
    item_id, mixture, sample, _ = items[0]
    assert mixture == build / "items" / item_id / "mixture.wav" and sample.name == "sample.wav"
    assert [i[0] for i in common.iter_items(build, only="long")] == ["rv1-00003"]
    assert [i[0] for i in common.iter_items(build, only="clips", limit=2)] == ["rv1-00000",
                                                                               "rv1-00001"]
    with pytest.raises(ValueError):
        list(common.iter_items(build, only="short"))
    assert [i[0] for i in common.iter_items(build, quick=True)] == ["rv1-00000"]


def test_read_wav_16k_resamples_and_mixes_down(tmp_path):
    path = tmp_path / "x.wav"
    stereo = np.stack([_tone(1.0, 44100), _tone(1.0, 44100)], axis=1)
    sf.write(path, stereo, 44100)
    x = common.read_wav_16k(path)
    assert x.dtype == np.float32 and x.ndim == 1 and len(x) == RATE
    assert abs(float(np.max(np.abs(x))) - 0.1) < 0.01


def test_write_wav_16k_is_pcm16(tmp_path):
    path = tmp_path / "y.wav"
    common.write_wav_16k(path, np.array([0.5, np.nan, 2.0], dtype=np.float32))
    info = sf.info(path)
    assert info.samplerate == RATE and info.subtype == "PCM_16" and info.channels == 1
    y, _ = sf.read(path)
    assert y[1] == 0.0 and y[2] == pytest.approx(1.0, abs=1e-4)
    assert not (tmp_path / "y.wav.part").exists()


def test_as_output_fits_length_and_scales_loud_output():
    y, gain = common.as_output(np.ones((1, 5), dtype=np.float32) * 0.5, 8)
    assert y.shape == (8,) and gain == 0.0 and y[-1] == 0.0
    y, gain = common.as_output(np.full(10, 2.0), 4)
    assert y.shape == (4,) and float(np.max(y)) == pytest.approx(common.PEAK) and gain < 0
    with pytest.raises(ValueError):
        common.as_output(np.zeros((2, 5)), 5)


def test_in_chunks_covers_the_input_with_short_windows():
    x = np.random.default_rng(0).standard_normal(10_000).astype(np.float32)
    seen = []

    def identity(seg):
        seen.append(len(seg))
        return seg

    y = common.in_chunks(identity, x, 1000, chunk_s=3.0, overlap_s=0.5)
    assert y.shape == x.shape and np.allclose(y, x, atol=1e-5)
    assert max(seen) <= 3000 and len(seen) == 4
    assert len(common.in_chunks(identity, x[:2000], 1000, 3.0, 0.5)) == 2000


def test_run_system_logs_done_abstained_and_error(build):
    def process(mixture, sample, row):
        if row["id"] == "rv1-00000":
            return mixture[: len(mixture) // 2] * 0.5   # short: padded to the mixture
        if row["id"] == "rv1-00001":
            return None
        raise RuntimeError("model fell over")

    info = {"package": "fake", "version": "0.0", "weights": "none"}
    counts = common.run_system(build, "fake", process, only="clips", log=False, info=info)
    assert counts == {"done": 1, "abstained": 1, "error": 1, "skipped": 0}
    out = build / "outputs" / "fake"
    y, sr = sf.read(out / "rv1-00000.wav")
    assert sr == RATE and len(y) == RATE and sf.info(out / "rv1-00000.wav").subtype == "PCM_16"
    assert not (out / "rv1-00001.wav").exists() and not (out / "rv1-00002.wav").exists()

    head, *lines = _log(build, "fake")
    assert head["run"] == "start" and head["version"] == "0.0" and head["weights"] == "none"
    assert [(r["id"], r["status"]) for r in lines] == [
        ("rv1-00000", "done"), ("rv1-00001", "abstained"), ("rv1-00002", "error")]
    assert lines[2]["error"] == "RuntimeError: model fell over"
    assert lines[0]["error"] is None and lines[0]["seconds"] >= 0.0

    # Resuming skips the item that has an output and runs the others again.
    calls = []

    def again(mixture, sample, row):
        calls.append(row["id"])
        return mixture

    counts = common.run_system(build, "fake", again, only="clips", log=False)
    assert calls == ["rv1-00001", "rv1-00002"]
    assert counts == {"done": 2, "abstained": 0, "error": 0, "skipped": 1}
    assert len(_log(build, "fake")) == 4 + 3

    # Without resuming, an item that fails loses the older output.
    def broken(mixture, sample, row):
        raise ValueError("no")

    counts = common.run_system(build, "fake", broken, limit=1, resume=False, log=False)
    assert counts["error"] == 1 and not (out / "rv1-00000.wav").exists()


# Runner scripts

@pytest.mark.parametrize("name", ["deepfilternet3", "usef_tfgridnet"])
def test_scripts_print_help(name):
    with pytest.raises(SystemExit) as exc:
        _script(name).main(["--help"])
    assert exc.value.code == 0


def test_deepfilternet3_names_the_missing_package(build, monkeypatch):
    enhancer = _script("deepfilternet3")
    for name in ("df", "df.enhance", "torch"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(RuntimeError, match="'deepfilternet' package"):
        enhancer.process(_tone(1.0), _tone(0.5), {})
    assert enhancer.main(["--build", str(build)]) == 2
    assert not (build / "outputs").exists()


def test_usef_names_the_missing_code_and_torch(tmp_path, monkeypatch):
    usef = _script("usef_tfgridnet")
    with pytest.raises(RuntimeError, match="USEF-TFGridNet code"):
        usef.process(_tone(1.0), _tone(0.5), {}, repo=tmp_path / "absent")
    (tmp_path / "clone" / "models").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="'torch' and 'hyperpyyaml' packages"):
        usef.process(_tone(1.0), _tone(0.5), {}, repo=tmp_path / "clone")


# The adapters against fakes of their packages

class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, key):
        return FakeTensor(self.a[key])

    def __mul__(self, k):
        return FakeTensor(self.a * k)

    def squeeze(self, dim):
        return FakeTensor(self.a.squeeze(dim))

    def to(self, device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def _fake_torch(loaded=None):
    torch = types.ModuleType("torch")
    torch.__version__ = "fake"
    torch.from_numpy = FakeTensor
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.load = lambda path, map_location=None, weights_only=True: loaded
    return torch


def test_deepfilternet3_runs_at_48k_and_ignores_the_sample(build, monkeypatch):
    enhancer = _script("deepfilternet3")
    seen = {}

    class State:
        def sr(self):
            return 48000

    def enhance(model, state, audio):
        seen["shape"] = audio.a.shape
        return audio * 0.5

    enhance_module = types.ModuleType("df.enhance")
    enhance_module.init_df = lambda: ("model", State(), "suffix")
    enhance_module.enhance = enhance
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "df", types.ModuleType("df"))
    monkeypatch.setitem(sys.modules, "df.enhance", enhance_module)
    mixture = _tone(1.0)
    y = enhancer.process(mixture, np.zeros(10, dtype=np.float32), {})
    assert seen["shape"] == (1, 48000)
    assert len(y) == RATE and np.allclose(y[100:-100], 0.5 * mixture[100:-100], atol=2e-3)

    assert enhancer.main(["--build", str(build), "--limit", "1"]) == 0
    head, line = _log(build, "deepfilternet3")
    assert head["package"] == "deepfilternet" and "DeepFilterNet3" in head["weights"]
    assert "version" in head and line["status"] == "done"


def _fake_clone(root):
    (root / "models").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "chkpt").mkdir()
    (root / "models" / "usef_tfgridnet.py").write_text(
        "import numpy as np\n"
        "class USEF_TFGridNet:\n"
        "    def __init__(self, n_fft=128, layers=1):\n"
        "        self.n_fft, self.layers, self.state = n_fft, layers, None\n"
        "    def load_state_dict(self, state, strict=True):\n"
        "        self.state = state\n"
        "    def to(self, device):\n"
        "        return self\n"
        "    def eval(self):\n"
        "        return self\n"
        "    def forward(self, mix, enrolment):\n"
        "        return mix * 0.5\n"
        "    def __call__(self, mix, enrolment):\n"
        "        assert enrolment.a.shape[0] == 1\n"
        "        return (self.forward(mix, enrolment),)\n")
    (root / "config" / "usef_tfgridnet.yaml").write_text(
        "data:\n  sample_rate: 8000\nmodel:\n  n_fft: 256\n  layers: 2\n  unknown: 1\n")
    (root / "chkpt" / "usef_tfgridnet.pt").write_bytes(b"weights")
    (root / "eval.py").write_text("import torch\nmodel = build(config)\nest = model(mix, enr)\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "refs" / "heads" / "main").write_text("abc123\n")


@pytest.fixture
def forget_models():
    # The fake clone's models package must not stay imported for later tests.
    yield
    for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
        del sys.modules[name]


def _fake_hyperpyyaml():
    """Stands in for HyperPyYAML: builds modules.masknet from the fake clone's class."""
    import importlib
    import types

    import yaml

    def load_hyperpyyaml(text):
        cfg = yaml.safe_load(text)
        cls = importlib.import_module("models.usef_tfgridnet").USEF_TFGridNet
        args = {k: v for k, v in cfg["model"].items() if k in ("n_fft", "layers")}
        return {"sample_rate": cfg["data"]["sample_rate"], "modules": {"masknet": cls(**args)}}

    module = types.ModuleType("hyperpyyaml")
    module.load_hyperpyyaml = load_hyperpyyaml
    return module


@pytest.mark.usefixtures("forget_models")
def test_usef_builds_from_the_config_and_runs_in_chunks(build, tmp_path, monkeypatch):
    usef = _script("usef_tfgridnet")
    clone = tmp_path / "clone"
    _fake_clone(clone)
    weights = {"model_state_dict": {"module.w": 1, "module.convolution_x": 2}}
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(weights))
    monkeypatch.setitem(sys.modules, "hyperpyyaml", _fake_hyperpyyaml())
    monkeypatch.syspath_prepend(str(clone))
    m = usef.load(repo=clone)
    assert m["rate"] == 8000 and m["class"] == "models.usef_tfgridnet.USEF_TFGridNet"
    model = m["model"]
    # eval.py's renames: drop "module." and map "convolution_" to "convolution_module."
    assert (model.n_fft, model.layers) == (256, 2)
    assert model.state == {"w": 1, "convolution_module.x": 2}

    mixture = _tone(3.0)
    y = usef.process(mixture, _tone(0.5), {}, repo=clone, chunk_s=1.0)
    assert len(y) == len(mixture)
    assert np.allclose(y[200:-200], 0.5 * mixture[200:-200], atol=5e-3)

    assert usef.main(["--build", str(build), "--repo", str(clone), "--limit", "1"]) == 0
    head, line = _log(build, "usef_tfgridnet")
    assert head["version"] == "abc123" and head["model_rate"] == 8000
    assert "CC BY-NC 4.0" in head["licence"] and usef.WEIGHTS_URL in head["weights"]
    assert line["status"] == "done"


def test_usef_prefers_the_published_config_and_the_whamr_checkpoint(tmp_path):
    usef = _script("usef_tfgridnet")
    clone = tmp_path / "clone"
    _fake_clone(clone)
    published = clone / usef.PUBLISHED_CONFIG
    published.parent.mkdir(parents=True)
    published.write_text("sample_rate: 8000\n")
    whamr = clone / usef.PUBLISHED_CHECKPOINT
    whamr.parent.mkdir(parents=True)
    whamr.write_bytes(b"weights")
    assert usef.find_config(clone) == published
    assert usef.find_checkpoint(clone) == whamr
    assert "whamr!" in str(usef.PUBLISHED_CHECKPOINT)


def test_usef_probe_and_fetch(tmp_path, capsys):
    usef = _script("usef_tfgridnet")
    clone = tmp_path / "clone"
    _fake_clone(clone)
    assert usef.probe(clone, None, None) == 0
    text = capsys.readouterr().out
    assert "abc123" in text and "model(mix, enr)" in text and "rate 8000" in text
    assert "USEF_TFGridNet: __init__['n_fft', 'layers']" in text
    assert "would use checkpoint: " + str(clone / "chkpt" / "usef_tfgridnet.pt") in text
    given = Path("config") / "usef_tfgridnet.yaml"
    assert usef.find_config(clone, given) == clone / given

    calls = []
    dest = usef.fetch(tmp_path / "third_party" / "usef_tfgridnet",
                      run=lambda cmd, check: calls.append(cmd))
    assert calls == [["git", "clone", usef.UPSTREAM_URL, str(dest)]]
    assert usef.UPSTREAM_URL == "https://github.com/ZBang/USEF-TSE"
    assert usef.WEIGHTS_URL == "https://huggingface.co/ZBang/USEF-TSE"
    assert usef.fetch(clone, run=lambda cmd, check: calls.append(cmd)) == clone
    assert len(calls) == 1
