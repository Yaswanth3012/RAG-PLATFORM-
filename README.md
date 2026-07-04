# Enterprise RAG Platform

A production-shaped Retrieval-Augmented Generation backend built for
enterprise scale: 100k+ documents, hybrid search, metadata filtering,
role-based access control, source citations, table/image understanding,
and streaming responses.

This is not a PDF-chatbot demo. It's structured the way a real deployment
is structured: Postgres as the metadata/RBAC system of record, Qdrant as
the vector store, a proper ingestion pipeline with idempotent re-ingestion,
and defense-in-depth access control enforced at both the vector-store and
database layers.

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │              FastAPI (app/main.py)        │
                     └───────────────┬───────────────────────────┘
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             /auth/token       /ingest            /query/stream
             (JWT login)     (upload doc)      (SSE streamed answer)
                    │                │                │
                    ▼                ▼                ▼
              Postgres users   Ingestion pipeline   Hybrid search
              + roles           │                     │
                                 ▼                     ▼
                    ┌─────────────────────┐   ┌──────────────────┐
                    │ loaders.py           │   │ dense: Qdrant     │
                    │ (unstructured, OCR,  │   │ sparse: BM25/ES   │
                    │  Claude vision)       │   │ fused: RRF        │
                    │ chunking.py           │   │ RBAC-filtered     │
                    │ table_extraction.py   │   │ reranked (opt.)   │
                    │ image_understanding.py│   └────────┬─────────┘
                    └──────────┬────────────┘            │
                               ▼                          ▼
                      Postgres (chunks,          generation/rag_engine.py
                      tables, images,            (Claude, streamed,
                      documents, ACL)             cited [1][2]...)
                               │
                               ▼
                         Qdrant (vectors +
                         filterable payload)
```

## Why these design choices

**Hybrid search, fused with RRF, not a weighted blend.**
Dense (embedding) search wins on semantic/paraphrase queries; BM25/keyword
search wins on exact terms — SKUs, ticket IDs, acronyms, names — which is
what enterprise users actually search for constantly. Reciprocal Rank
Fusion combines the two rankings without needing to calibrate incomparable
score scales, and it's robust across very different query shapes.

**RBAC enforced twice, not once.**
Every document carries `access_tags` (e.g. `dept:finance`,
`classification:confidential`). A user's roles resolve to an allowed-tag
set; a document is visible only if its tags are a *subset* of what the
user is allowed (default-deny composition of AND-conditions, not OR). This
filter runs first inside the Qdrant query itself — so restricted chunks are
never retrieved, not just hidden after the fact — and is checked again at
the Postgres layer when citations are hydrated, in case permissions changed
after a point was indexed.

**Tables and images are first-class citizens, not flattened text.**
Tables are parsed into structured JSON (via `unstructured`'s HTML table
output, falling back to Camelot for lattice-style PDFs) so the model can
cite specific rows instead of garbled run-on numbers. Images get both OCR
(literal text) and a vision-model caption (semantic content), concatenated
into a single searchable chunk, with the original stored separately for
UI display.

**Citations are resolved after generation, never trusted from the model's
memory.** Retrieved chunks are numbered `[1]`, `[2]`... in the prompt; the
model is instructed to cite using those markers; after the stream
completes, we scan the answer for which markers actually appear and return
only those as sources — so the UI never shows an unused "source."

**Idempotent, resumable bulk ingestion.** Each file is checksummed; a
document that's already indexed with a matching checksum is skipped. The
bulk ingestion script writes failures to a manifest so a 100k-document
batch job can be safely re-run without redoing successful work.

## Project layout

```
app/
  auth/            JWT auth + RBAC (access-tag resolution & filters)
  db/               Postgres schema (init_db.sql) + SQLAlchemy models
  ingestion/        loaders, chunking, table extraction, image understanding, pipeline
  retrieval/        embeddings, Qdrant store, sparse/BM25 search, RRF fusion, reranker
  generation/       citation building/resolution, streaming RAG engine
  api/              FastAPI routes: auth, ingest, query
scripts/
  bulk_ingest.py    concurrent bulk ingestion with failure manifest
tests/
  test_hybrid_search.py   unit tests for RRF fusion & RBAC logic (no infra needed)
docker-compose.yml  Postgres + Qdrant + Redis + Elasticsearch(optional) + API
```

## Running it

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY
docker-compose up --build
```

This brings up Postgres (schema auto-applied from `init_db.sql`), Qdrant,
Redis, Elasticsearch, and the API on `localhost:8000`.

Create a user + role directly in Postgres for a first login (a proper
signup/admin-invite flow is the next thing you'd add before shipping this
to real users — deliberately out of scope here so the RAG mechanics stay
front and center):

```sql
INSERT INTO users (email, hashed_password, department)
VALUES ('alice@company.com', '<bcrypt hash>', 'finance');

INSERT INTO user_roles (user_id, role_id)
SELECT u.id, r.id FROM users u, roles r
WHERE u.email = 'alice@company.com' AND r.name = 'finance_analyst';
```

Then:

```bash
# Log in
curl -X POST localhost:8000/auth/token -d "username=alice@company.com&password=..."

# Ingest a document
curl -X POST localhost:8000/ingest \
  -H "Authorization: Bearer <token>" \
  -F "file=@quarterly_report.pdf" \
  -F "department=finance" \
  -F "classification=confidential"

# Ask a streamed, cited question
curl -N -X POST localhost:8000/query/stream \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"query": "What was Q3 revenue in EMEA?"}'
```

For 100k+ document corpora, use the bulk script instead of the single-file
endpoint:

```bash
python scripts/bulk_ingest.py --root /data/corpus --department finance --classification confidential
```

## What's stubbed vs. production-ready

Honest about scope:

- **Production-ready as-is:** RBAC model, RRF fusion logic, citation
  resolution, chunking strategy, schema design, streaming — these are
  fully implemented and unit-tested (`tests/test_hybrid_search.py`).
- **Needs scaling work before 100k+ docs in practice:** the BM25 fallback
  (`retrieval_backend=bm25`) is in-process and rebuilt from scratch — swap
  to `elasticsearch` (already wired) for real scale. Qdrant should be run
  as a sharded cluster, not a single node, at that document count.
- **Deliberately out of scope:** user signup/invite flows, a frontend,
  fine-grained per-chunk ACLs beyond document-level tags, and multi-tenant
  isolation — each is a real project in itself.

## Talking about this in interviews

The things worth being able to explain clearly: why RRF over score
blending, why RBAC is checked at the vector-store layer *and* the DB layer,
why tables/images get separate structured extraction instead of naive text
flattening, and why citations are resolved post-hoc from the model's actual
output rather than requested from the model directly (an LLM will
confidently cite a source it didn't use if you just ask it to name one).
