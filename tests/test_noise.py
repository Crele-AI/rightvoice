import csv
import json

from rightvoice import noise

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
BY = "http://creativecommons.org/licenses/by/3.0/"
NC = "http://creativecommons.org/licenses/by-nc/3.0/"


def _root(tmp_path):
    (tmp_path / "FSD50K.metadata").mkdir()
    (tmp_path / "FSD50K.ground_truth").mkdir()
    info = {
        "1": {"title": "rain", "license": CC0, "uploader": "a"},
        "2": {"title": "guitar", "license": BY, "uploader": "b"},
        "3": {"title": "talk", "license": CC0, "uploader": "c"},
        "4": {"title": "crowd", "license": BY, "uploader": "d"},
        "5": {"title": "engine nc", "license": NC, "uploader": "e"},
    }
    (tmp_path / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").write_text(json.dumps(info))
    rows = [
        ("1", "Rain,Water", "/m/1,/m/2"),
        ("2", "Guitar,Musical_instrument,Music", "/m/3,/m/4,/m/5"),
        ("3", "Speech,Human_voice", "/m/6,/m/7"),
        ("4", "Crowd,Human_group_actions", "/m/8,/m/9"),
        ("5", "Engine", "/m/10"),
    ]
    with (tmp_path / "FSD50K.ground_truth" / "eval.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fname", "labels", "mids"])
        w.writerows(rows)
    return tmp_path


def test_pools_and_exclusions(tmp_path):
    clips = noise.catalogue(_root(tmp_path))
    by_id = {c.id: c for c in clips}
    assert by_id["1"].pool == "noise"
    assert by_id["2"].pool == "music"
    assert (by_id["3"].pool, by_id["3"].reason) == ("excluded", "voice")
    assert (by_id["4"].pool, by_id["4"].reason) == ("excluded", "voice")
    assert (by_id["5"].pool, by_id["5"].reason) == ("excluded", "licence")
    assert noise.summary(clips) == {"total": 5, "noise": 1, "music": 1,
                                    "excluded_licence": 1, "excluded_voice": 2}
    assert by_id["1"].path == "FSD50K.eval_audio/1.wav"


def test_attribution_lists_only_kept_clips(tmp_path):
    clips = noise.catalogue(_root(tmp_path))
    rows = noise.attribution_rows(clips)
    assert [r["clip_id"] for r in rows] == ["1", "2"]
    assert rows[0]["url"] == "https://freesound.org/s/1/" and rows[0]["licence"] == CC0
    out = tmp_path / "attribution.csv"
    noise.write_attribution(clips, out)
    with out.open(newline="") as fh:
        assert [r["uploader"] for r in csv.DictReader(fh)] == ["a", "b"]
