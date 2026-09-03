from __future__ import annotations

import pytest

from services.projects.assets import build_assets
from services.projects.importer import material_chunk


def chunk(path, symbol):
    return material_chunk(
        project_id="orders", version="v1", module="orders", path=path, symbol=symbol,
        text=f"class {symbol}: pass", cloud_generation_allowed=False,
    )


def test_assets_are_deterministic_and_only_point_to_existing_project_evidence():
    one, two = chunk("services/orders/app.py", "CheckoutService"), chunk("services/orders/api.py", "CheckoutService")
    assets = build_assets([two, one])
    assert "项目 orders：2 个证据块" in assets.summary
    assert assets.aliases["CheckoutService"] == tuple(sorted((one.source_url, two.source_url)))
    assert assets.aliases["app.py"] == (one.source_url,)


def test_assets_reject_cross_project_mixing():
    one = chunk("a.py", "A")
    two = material_chunk(project_id="payments", version="v1", module=None, path="b.py", symbol="B", text="x", cloud_generation_allowed=True)
    with pytest.raises(ValueError, match="多个项目"):
        build_assets([one, two])
