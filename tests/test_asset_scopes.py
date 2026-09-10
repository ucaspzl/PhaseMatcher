import json
from pathlib import Path

import pytest

from phasematcher.cli import _asset_records


@pytest.mark.parametrize("dataset", ["phasemix", "rruff"])
@pytest.mark.parametrize("scope,count", [("inference", 4), ("dataset", 16), ("all", 18)])
def test_asset_scope(dataset, scope, count):
    manifest = json.loads((Path(__file__).parents[1] / "assets.json").read_text())
    records = _asset_records(manifest, dataset, scope)
    paths = {r["path"] for r in records}
    assert len(paths) == count
    assert f"ckpt/{dataset}/last.pt" in paths
    assert all(f"/{dataset}/" in p for p in paths)
    if scope != "all":
        assert f"ckpt/{dataset}/single.pt" not in paths
        assert f"ckpt/{dataset}/phase.pt" not in paths
    if scope == "inference":
        assert not any("manifest/" in p or "observations/" in p for p in paths)


def test_unknown_scope_rejected():
    with pytest.raises(ValueError, match="Unknown asset scope"):
        _asset_records({"files": []}, "rruff", "unknown")
