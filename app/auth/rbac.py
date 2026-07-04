"""
Role-Based Access Control for retrieval.

Design principle: default-deny. A document is visible to a user only if
the UNION of access_tags granted by all of the user's roles is a SUPERSET
of the document's required access_tags. A document tagged
{classification:confidential, dept:finance} requires the user to hold
BOTH tags via their roles — not just one.

This filter is applied in TWO places for defense in depth:
  1. At the vector-store level (Qdrant filter on access_tags) so restricted
     chunks are never even retrieved, not just hidden later.
  2. At the Postgres level when hydrating chunk text/citations, as a
     belt-and-suspenders check in case the vector store filter is stale
     (e.g. a role's permissions changed after the point was indexed).
"""

from dataclasses import dataclass
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Role


@dataclass
class AccessContext:
    user_id: str
    allowed_tags: set[str]
    is_admin: bool = False


async def build_access_context(db: AsyncSession, user: User) -> AccessContext:
    """Resolve a user's roles into the flat set of access tags they may read."""
    role_names = {r.name for r in user.roles}
    if "admin" in role_names:
        return AccessContext(user_id=str(user.id), allowed_tags=set(), is_admin=True)

    from app.db.models import role_access_tags as rat_table  # avoid import cycle

    allowed: set[str] = set()
    for role in user.roles:
        rows = await db.execute(
            select(rat_table.c.access_tag).where(rat_table.c.role_id == role.id)
        )
        allowed.update(r[0] for r in rows)

    return AccessContext(user_id=str(user.id), allowed_tags=allowed, is_admin=False)


def document_is_visible(doc_tags: Sequence[str], ctx: AccessContext) -> bool:
    """A doc is visible iff every tag it requires is covered by the user's grants."""
    if ctx.is_admin:
        return True
    return set(doc_tags).issubset(ctx.allowed_tags)


def qdrant_access_filter(ctx: AccessContext):
    """
    Build a Qdrant filter that enforces the subset check above at the vector-
    store layer. Qdrant has no native "point.tags subset_of allowed" operator,
    so at ingest time we store, per point, a payload field `denied_tags` that
    is precomputed as (ALL_KNOWN_TAGS - point.access_tags) is NOT what we want
    either. The approach that works cleanly with Qdrant's filter DSL is:

      - Store `access_tags` (the doc's required tags) on every point.
      - At query time, require that access_tags is a subset of allowed_tags.
        Qdrant supports MatchAny / MatchExcept but not "all values in set",
        so we instead invert: reject any point carrying a tag NOT in the
        user's allowed set, expressed as `must_not` MatchAny over the
        complement of allowed_tags within our known tag vocabulary.

    For a bounded, known tag vocabulary (departments x classifications,
    typically <200 distinct tags in an enterprise), computing the complement
    is cheap and exact. See retrieval/hybrid_search.py::KNOWN_TAG_VOCAB.
    """
    from qdrant_client.http import models as qm
    from app.retrieval.hybrid_search import KNOWN_TAG_VOCAB

    if ctx.is_admin:
        return None  # no filter — admin sees everything

    forbidden = KNOWN_TAG_VOCAB - ctx.allowed_tags
    if not forbidden:
        return None

    return qm.Filter(
        must_not=[qm.FieldCondition(key="access_tags", match=qm.MatchAny(any=list(forbidden)))]
    )
