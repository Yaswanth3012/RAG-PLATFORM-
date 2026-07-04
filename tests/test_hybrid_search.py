"""
Unit tests for the parts of the system that are pure logic and don't need
live Qdrant/Postgres: RRF fusion math and RBAC subset-check filtering.
Run with: pytest tests/test_hybrid_search.py -v
"""

from app.retrieval.hybrid_search import _rrf_fuse, apply_metadata_filters, RetrievedChunk
from app.auth.rbac import document_is_visible, AccessContext


def test_rrf_fuse_prioritizes_agreement():
    dense = [{"chunk_id": "a", "score": 0.9, "payload": {}}, {"chunk_id": "b", "score": 0.8, "payload": {}}]
    sparse = [{"chunk_id": "b", "score": 5.0, "payload": {}}, {"chunk_id": "c", "score": 4.0, "payload": {}}]

    fused = _rrf_fuse(dense, sparse, k=60)
    # "b" appears in both lists at good ranks -> should be ranked first
    assert fused[0].chunk_id == "b"


def test_rrf_fuse_handles_disjoint_results():
    dense = [{"chunk_id": "a", "score": 0.9, "payload": {}}]
    sparse = [{"chunk_id": "b", "score": 5.0, "payload": {}}]
    fused = _rrf_fuse(dense, sparse, k=60)
    assert {c.chunk_id for c in fused} == {"a", "b"}


def test_metadata_filter_matches_exact_value():
    chunks = [
        RetrievedChunk(chunk_id="1", score=1.0, payload={"department": "finance"}),
        RetrievedChunk(chunk_id="2", score=1.0, payload={"department": "hr"}),
    ]
    filtered = apply_metadata_filters(chunks, {"department": "finance"})
    assert [c.chunk_id for c in filtered] == ["1"]


def test_metadata_filter_matches_any_in_list():
    chunks = [
        RetrievedChunk(chunk_id="1", score=1.0, payload={"department": "finance"}),
        RetrievedChunk(chunk_id="2", score=1.0, payload={"department": "hr"}),
        RetrievedChunk(chunk_id="3", score=1.0, payload={"department": "legal"}),
    ]
    filtered = apply_metadata_filters(chunks, {"department": ["finance", "hr"]})
    assert {c.chunk_id for c in filtered} == {"1", "2"}


def test_rbac_document_visible_when_tags_subset_of_allowed():
    ctx = AccessContext(user_id="u1", allowed_tags={"dept:finance", "classification:internal"})
    assert document_is_visible(["dept:finance"], ctx) is True
    assert document_is_visible(["dept:finance", "classification:confidential"], ctx) is False


def test_rbac_admin_sees_everything():
    ctx = AccessContext(user_id="u1", allowed_tags=set(), is_admin=True)
    assert document_is_visible(["dept:finance", "classification:restricted"], ctx) is True
