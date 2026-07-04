-- ============================================================
-- Production RAG Platform — core schema
-- Postgres is the system of record for metadata, RBAC and audit.
-- Qdrant only stores vectors + a *copy* of filterable metadata
-- (chunk_id, doc_id, acl_tags) needed for fast filtered search.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------- Identity & RBAC ----------

CREATE TABLE roles (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(64) UNIQUE NOT NULL,          -- e.g. 'finance_analyst', 'admin'
    description TEXT
);

CREATE TABLE users (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email          VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    department     VARCHAR(128),
    is_active      BOOLEAN DEFAULT TRUE,
    created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);

-- Permissions attach a role to an "access tag" it may read.
-- Access tags are arbitrary strings like 'dept:finance', 'classification:confidential',
-- 'region:eu'. A document carries a list of tags; a user can see it iff their
-- roles collectively cover ALL required tags on that document (default-deny).
CREATE TABLE role_access_tags (
    role_id    UUID REFERENCES roles(id) ON DELETE CASCADE,
    access_tag VARCHAR(128) NOT NULL,
    PRIMARY KEY (role_id, access_tag)
);

-- ---------- Documents & chunks ----------

CREATE TABLE documents (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_uri       TEXT NOT NULL,                 -- original path / URL / bucket key
    title            TEXT,
    doc_type         VARCHAR(32),                   -- pdf, docx, image, html, ...
    department       VARCHAR(128),
    classification   VARCHAR(32) DEFAULT 'internal', -- public | internal | confidential | restricted
    access_tags      TEXT[] DEFAULT '{}',            -- e.g. {dept:finance, classification:confidential}
    version          INT DEFAULT 1,
    checksum         VARCHAR(64),                    -- sha256, used for idempotent re-ingest
    status           VARCHAR(32) DEFAULT 'pending',  -- pending | processing | indexed | failed
    ingested_by      UUID REFERENCES users(id),
    ingested_at      TIMESTAMPTZ DEFAULT now(),
    extra_metadata   JSONB DEFAULT '{}'::jsonb        -- author, publish_date, tags, custom fields
);

CREATE INDEX idx_documents_access_tags ON documents USING GIN (access_tags);
CREATE INDEX idx_documents_metadata    ON documents USING GIN (extra_metadata);
CREATE INDEX idx_documents_dept        ON documents (department);
CREATE INDEX idx_documents_checksum    ON documents (checksum);

-- One row per chunk. chunk vectors themselves live in Qdrant;
-- this table is the source of truth for text + provenance used in citations.
CREATE TABLE chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    content_type  VARCHAR(16) DEFAULT 'text',        -- text | table | image_caption
    page_number   INT,
    bbox          JSONB,                              -- bounding box for source highlighting
    token_count   INT,
    qdrant_point_id UUID,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_chunks_document_id ON chunks (document_id);

-- Extracted tables kept structured (not just flattened text) so the
-- generation layer can render them faithfully and cite exact cells.
CREATE TABLE extracted_tables (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id  UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id     UUID REFERENCES chunks(id) ON DELETE CASCADE,
    page_number  INT,
    table_json   JSONB NOT NULL,                      -- list-of-rows representation
    caption      TEXT
);

-- Extracted images with model-generated descriptions (for image understanding).
CREATE TABLE extracted_images (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id  UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id     UUID REFERENCES chunks(id) ON DELETE CASCADE,
    page_number  INT,
    storage_path TEXT NOT NULL,
    description  TEXT,                                -- caption from vision model
    ocr_text     TEXT
);

-- ---------- Query audit / citations ----------

CREATE TABLE query_log (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       UUID REFERENCES users(id),
    query_text    TEXT NOT NULL,
    filters       JSONB,
    retrieved_chunk_ids UUID[],
    answer_text   TEXT,
    latency_ms    INT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Seed baseline roles
INSERT INTO roles (name, description) VALUES
    ('admin', 'Full access to all documents'),
    ('finance_analyst', 'Finance department access'),
    ('hr_analyst', 'HR department access'),
    ('general_employee', 'Public + internal only');

INSERT INTO role_access_tags (role_id, access_tag)
SELECT id, 'classification:public' FROM roles WHERE name = 'general_employee'
UNION ALL
SELECT id, 'classification:internal' FROM roles WHERE name = 'general_employee'
UNION ALL
SELECT id, tag FROM roles, (VALUES ('classification:public'),('classification:internal'),
    ('classification:confidential'),('dept:finance')) AS t(tag) WHERE roles.name = 'finance_analyst'
UNION ALL
SELECT id, tag FROM roles, (VALUES ('classification:public'),('classification:internal'),
    ('classification:confidential'),('dept:hr')) AS t(tag) WHERE roles.name = 'hr_analyst';
