from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class MemoryType(str, enum.Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    AVERSION = "aversion"
    PERSON_PLACE = "person_place"
    HABIT = "habit"
    EVENT = "event"
    COMMUNICATION = "communication"
    PLAN = "plan"


class MemoryStatus(str, enum.Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    DISABLED = "disabled"


class ImportJob(Base):
    __tablename__ = "imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_format: Mapped[str] = mapped_column(String(64), default="wechat_json")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    error_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    field_mapping: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    subject_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    partner_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    messages: Mapped[list[Message]] = relationship(back_populates="import_job")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("fingerprint", name="uq_message_fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("imports.id"), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sender: Mapped[str] = mapped_column(String(128), nullable=False)
    sender_role: Mapped[str] = mapped_column(String(32), default="unknown")  # subject|partner|other
    content: Mapped[str] = mapped_column(Text, nullable=False)
    msg_type: Mapped[str] = mapped_column(String(32), default="text")
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    import_job: Mapped[ImportJob] = relationship(back_populates="messages")


class MessageChunk(Base):
    __tablename__ = "message_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    import_id: Mapped[int] = mapped_column(ForeignKey("imports.id"), index=True)
    start_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    end_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    message_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    redacted_text: Mapped[str] = mapped_column(Text, nullable=False)
    original_char_count: Mapped[int] = mapped_column(Integer, default=0)
    redacted_char_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    import_id: Mapped[int | None] = mapped_column(ForeignKey("imports.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunk_total: Mapped[int] = mapped_column(Integer, default=0)
    chunk_done: Mapped[int] = mapped_column(Integer, default=0)
    candidates_created: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(32), default=MemoryType.FACT.value, index=True)
    status: Mapped[str] = mapped_column(String(32), default=MemoryStatus.CANDIDATE.value, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Real-world validity: when this fact held true in the real world
    valid_from: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_inference: Mapped[bool] = mapped_column(Boolean, default=False)
    conflict_group: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Replacement link: a superseded memory points at the newer one via supersedes_id
    supersedes_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Retired at: when this memory stopped representing the current fact (system
    # time, not real-world time)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Pending replacement proposal: conflicts inferred by the model are never
    # rewritten directly, they go through review first
    proposed_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    proposed_target_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_runs.id"), nullable=True)
    source_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("message_chunks.id"), nullable=True)
    source_diary_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # When the system learned this (the moment it was first written to the database)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    evidence: Mapped[list[MemoryEvidence]] = relationship(
        back_populates="memory", cascade="all, delete-orphan"
    )


class MemoryEvidence(Base):
    __tablename__ = "memory_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(ForeignKey("memories.id"), index=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), index=True)
    is_counterexample: Mapped[bool] = mapped_column(Boolean, default=False)
    quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    memory: Mapped[Memory] = relationship(back_populates="evidence")
    message: Mapped[Message] = relationship()


class OutboundEvent(Base):
    """Local audit trail for every request that leaves the machine.

    Records metadata only — never prompt text or API keys — so the user can
    answer "what left my computer, to whom, and was it redacted?".
    """

    __tablename__ = "outbound_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    host: Mapped[str] = mapped_column(String(256), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    purpose: Mapped[str] = mapped_column(String(32), default="chat", index=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    redacted: Mapped[bool] = mapped_column(Boolean, default=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChatThread(Base):
    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    turns: Mapped[list["ChatTurn"]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ChatTurn.id",
    )


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("chat_threads.id"), index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), default="ask")
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    thread: Mapped[ChatThread] = relationship(back_populates="turns")


class DiaryEntry(Base):
    __tablename__ = "diary_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entry_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    mood: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    candidates_created: Mapped[int] = mapped_column(Integer, default=0)
    last_extracted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


def create_db_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {
            "check_same_thread": False,
            "timeout": 30,
        }
    engine = create_engine(database_url, connect_args=connect_args)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
            except Exception:
                # Another connection may hold the lock; ignore and continue.
                pass
            cursor.close()

    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
