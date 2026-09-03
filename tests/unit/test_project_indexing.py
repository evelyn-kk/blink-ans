from __future__ import annotations

from services.projects.importer import Material
from services.projects.indexing import rebuild_project_index
from services.projects.registry import Project
from services.retrieval import store as store_mod
from services.retrieval.embed import DIM
from services.retrieval.store import ChunkStore, IndexBuilder
from packages.schemas.chunk import Chunk, utc_now


class FakeEmbedder:
    model_id = "synthetic"

    def __init__(self):
        self.batches = []

    def encode(self, texts):
        self.batches.append(list(texts))
        return [[float(i + 1)] + [0.0] * (DIM - 1) for i, _ in enumerate(texts)]


def test_project_rebuild_replaces_only_its_old_chunks_and_carries_everything_else(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    official = Chunk(
        source_url="https://example.test/doc", source_project="official", version_or_commit="v1",
        license="Apache-2.0", retrieved_at=utc_now(), title_path=["Official"], technology="java",
        content_type="prose", locale="en", text="official evidence",
    )
    old_project = Chunk(
        source_url="project://orders/a.py", source_project="project:orders", version_or_commit="old",
        license="Proprietary", retrieved_at=utc_now(), title_path=["orders", "a.py"], technology="project",
        content_type="code", locale="en", text="old evidence", project_id="orders",
        cloud_generation_allowed=False,
    )
    for chunk in (official, old_project):
        chunk.validate()
    baseline = IndexBuilder("baseline")
    baseline.add([official, old_project], FakeEmbedder().encode([official.text, old_project.text]))
    baseline.finalize({"official": "v1", "project:orders": "old"}, "synthetic")
    baseline.activate()

    root = tmp_path / "project-root"
    root.mkdir()
    embedder = FakeEmbedder()
    stats = rebuild_project_index(
        Project("orders", "new", root, False),
        [Material("services/new.py", "new evidence", module="services", content_type="code")],
        embedder, source_index=tmp_path / "baseline.db", name="project-orders",
    )

    assert stats.added == 1 and stats.carried == 1
    store = ChunkStore(tmp_path / "project-orders.db")
    try:
        rows = store.execute("SELECT source_project, version_or_commit, text FROM chunks ORDER BY source_project")
        assert [tuple(row) for row in rows] == [
            ("official", "v1", "official evidence"),
            ("project:orders", "new", "new evidence"),
        ]
        assert '"project:orders": "new"' in store.meta["sources"]
    finally:
        store.close()


def test_project_rebuild_batches_large_materials_to_bound_embedding_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    source = IndexBuilder("baseline")
    seed = Chunk(
        source_url="https://example.test/doc", source_project="official", version_or_commit="v1",
        license="Apache-2.0", retrieved_at=utc_now(), title_path=["Official"], technology="java",
        content_type="prose", locale="en", text="official evidence",
    )
    seed.validate()
    source.add([seed], FakeEmbedder().encode([seed.text]))
    source.finalize({"official": "v1"}, "synthetic")
    source.activate()

    root = tmp_path / "project-root"
    root.mkdir()
    materials = [Material(f"docs/{n}.md", f"evidence {n}", content_type="prose") for n in range(20)]
    embedder = FakeEmbedder()
    rebuild_project_index(
        Project("orders", "v1", root, False), materials, embedder,
        source_index=tmp_path / "baseline.db", name="project-orders",
    )
    assert [len(batch) for batch in embedder.batches] == [16, 4]
