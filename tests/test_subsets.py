import sys
from pathlib import Path

import pytest

from rightvoice import subsets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
import common


def _rows():
    captures = ("room", "call", "phone_g711")
    rows = []
    for i in range(100):
        long = i >= 96
        rows.append({"id": f"rv1-{i:05d}", "long": long,
                     "target_absent": not long and i % 11 == 0,
                     "buckets": {"capture": captures[i % 3]}})
    return rows


def test_quick_keeps_every_nth_item_of_each_stratum_in_manifest_order():
    rows = _rows()
    ids = subsets.quick_ids(rows, every=5)
    for key in {subsets.stratum(r) for r in rows} - {"long"}:
        members = [r["id"] for r in rows if subsets.stratum(r) == key]
        assert [i for i in ids if i in members] == members[::5]
    assert ids == [r["id"] for r in rows if r["id"] in set(ids)]
    assert not any(r["long"] for r in rows if r["id"] in ids)


def test_quick_covers_every_stratum_and_is_fixed():
    rows = _rows()
    ids = subsets.quick_ids(rows)
    strata = {subsets.stratum(r) for r in rows}
    assert strata == {"long", "target_absent", "clip:room", "clip:call", "clip:phone_g711"}
    assert {subsets.stratum(r) for r in rows if r["id"] in ids} == strata - {"long"}
    assert ids == subsets.quick_ids(list(rows))
    with pytest.raises(ValueError):
        subsets.quick_ids(rows, every=0)


def test_the_runner_helper_uses_the_same_rule():
    rows = _rows()
    for every in (1, 4, 15):
        assert common.quick_ids(rows, every) == subsets.quick_ids(rows, every)
