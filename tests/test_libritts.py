import numpy as np
import soundfile as sf

from rightvoice import libritts


def _utt(root, split, spk, chapter, idx, seconds, text="hello there"):
    d = root / split / spk / chapter
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{spk}_{chapter}_{idx:06d}_000000"
    sf.write(d / f"{stem}.wav", np.zeros(int(seconds * 24000), dtype=np.float32), 24000, subtype="PCM_16")
    (d / f"{stem}.normalized.txt").write_text(text + "\n")
    (d / f"{stem}.original.txt").write_text(text.upper() + "\n")


def _tree(tmp_path):
    root = tmp_path / "LibriTTS_R"
    # speaker 1: two chapters, 4 minutes -> conversation pool
    for i in range(12):
        _utt(root, "dev-clean", "1", "100" if i < 6 else "101", i, 20.0)
    # speaker 2: one chapter, 4 minutes -> only one chapter, so bystander
    for i in range(12):
        _utt(root, "test-other", "2", "200", i, 20.0)
    # speaker 3: 10 seconds -> dropped
    _utt(root, "test-clean", "3", "300", 0, 10.0)
    # speaker 9: not on the allow list
    _utt(root, "test-clean", "9", "900", 0, 60.0)
    return root


def test_scan_filters_and_reads_durations(tmp_path):
    root = _tree(tmp_path)
    rows = libritts.scan(root, speakers=frozenset({"1", "2", "3"}))
    assert {r.speaker for r in rows} == {"1", "2", "3"}
    assert all(r.sample_rate == 24000 for r in rows)
    one = [r for r in rows if r.speaker == "1"]
    assert len(one) == 12 and abs(one[0].seconds - 20.0) < 1e-6
    assert one[0].text == "hello there" and one[0].path.startswith("dev-clean/1/100/")
    assert rows == sorted(rows, key=lambda r: r.id)


def test_pools_and_summary(tmp_path):
    root = _tree(tmp_path)
    rows = libritts.scan(root, speakers=frozenset({"1", "2", "3"}))
    pools = libritts.eligible_speakers(rows)
    assert pools == {"conversation": ["1"], "bystander": ["2"]}
    s = libritts.summary(rows)
    assert s["speakers"] == 3 and s["conversation_speakers"] == 1 and s["bystander_speakers"] == 1
    assert s["splits"]["dev-clean"]["utterances"] == 12


def test_load_caches(tmp_path):
    root = _tree(tmp_path)
    first = libritts.load(root, speakers=frozenset({"1"}))
    assert len(first) == 12 and libritts.cache_path(root).exists()
    (root / "dev-clean" / "1" / "100" / "1_100_000000_000000.wav").unlink()
    again = libritts.load(root, speakers=frozenset({"1"}))
    assert [x.id for x in again] == [x.id for x in first]
    rebuilt = libritts.load(root, speakers=frozenset({"1"}), rebuild=True)
    assert len(rebuilt) == 11


def test_reader_table_and_sex_counts(tmp_path):
    root = _tree(tmp_path)
    (root / "speakers.tsv").write_text(
        "READER\tGENDER\tSUBSET NAME\n1\tF\tdev-clean\tA Reader\n2\tM\ttest-other\tB Reader\n"
        "3\tM\ttest-clean\tC\n")
    table = libritts.reader_table(root)
    assert table["1"] == {"sex": "F", "subset": "dev-clean", "name": "A Reader"}
    rows = libritts.scan(root, speakers=frozenset({"1", "2", "3"}))
    s = libritts.summary(rows, table)
    assert s["sex_by_pool"] == {"conversation": {"F": 1, "M": 0}, "bystander": {"F": 0, "M": 1}}
