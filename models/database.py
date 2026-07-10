# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .base import Base
from app.config import settings

engine = None
SessionLocal = None

def init_engine():
    global engine, SessionLocal
    if engine is not None:
        return  
    engine = create_engine(settings.db_url, echo=True, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()