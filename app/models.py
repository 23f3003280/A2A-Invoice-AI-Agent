from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class TaskRecord(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    principal_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    message_id: Mapped[str] = mapped_column(String(256), nullable=False)
    semantic_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    batch_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    state: Mapped[str] = mapped_column(String(64), nullable=False)
    task_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        UniqueConstraint("principal_id", "message_id", name="uq_principal_message"),
        Index("ix_task_owner_created", "principal_id", "created_at"),
    )


class MessageReplay(Base):
    """Durable idempotency record for initial and continuation messages."""
    __tablename__ = "message_replays"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    message_id: Mapped[str] = mapped_column(String(256), nullable=False)
    semantic_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("principal_id", "message_id", name="uq_replay_principal_message"),)


class SemanticDecision(Base):
    __tablename__ = "semantic_decisions"
    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
