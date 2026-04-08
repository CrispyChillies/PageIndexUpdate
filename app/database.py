from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def initialize_database() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def sync_document_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS folder_id VARCHAR(255)"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_text TEXT"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(128)"))
        conn.execute(
            text(
                f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding vector({settings.embedding_dimensions})"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_documents_embedding_hnsw "
                "ON documents USING hnsw (embedding vector_cosine_ops)"
            )
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
