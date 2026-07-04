"""SQLAlchemy ORM models mirroring init_db.sql. Kept 1:1 with the SQL schema
so the schema file remains the readable source of truth for DB reviewers,
while these models give us type-safe access in code."""

import uuid
from sqlalchemy import Column, String, Integer, ForeignKey, TIMESTAMP, ARRAY, Table
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

user_roles = Table(
    "user_roles", Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id")),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id")),
)


class Role(Base):
    __tablename__ = "roles"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String)
    users = relationship("User", secondary=user_roles, back_populates="roles")


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    department = Column(String(128))
    is_active = Column(String, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    roles = relationship("Role", secondary=user_roles, back_populates="users")


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_uri = Column(String, nullable=False)
    title = Column(String)
    doc_type = Column(String(32))
    department = Column(String(128))
    classification = Column(String(32), default="internal")
    access_tags = Column(ARRAY(String), default=list)
    version = Column(Integer, default=1)
    checksum = Column(String(64))
    status = Column(String(32), default="pending")
    ingested_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    ingested_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    extra_metadata = Column(JSONB, default=dict)
    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    chunk_index = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    content_type = Column(String(16), default="text")
    page_number = Column(Integer)
    bbox = Column(JSONB)
    token_count = Column(Integer)
    qdrant_point_id = Column(UUID(as_uuid=True))
    document = relationship("Document", back_populates="chunks")
