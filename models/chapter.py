from sqlalchemy import create_engine, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .base import Base

from typing import Optional

class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[int] = mapped_column(primary_key=True)
    chapter_id: Mapped[int] = mapped_column()
    comic_name: Mapped[str] = mapped_column(String(500))
    canvas_url_before: Mapped[str] = mapped_column(String(500))
    canvas_url_after: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped["User"] = relationship(back_populates="chapters")