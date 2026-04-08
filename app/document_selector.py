import math
import os
import re
from typing import Any

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocumentRecord

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "what",
    "which",
    "with",
}


def get_openai_client() -> OpenAI:
    api_key = os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("CHATGPT_API_KEY or OPENAI_API_KEY must be set")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def extract_top_titles(nodes: list[dict[str, Any]], limit: int = 8) -> list[str]:
    titles: list[str] = []
    for node in nodes:
        title = node.get("title")
        if title:
            titles.append(str(title))
        if len(titles) >= limit:
            break
    return titles[:limit]


def extract_top_summaries(nodes: list[dict[str, Any]], limit: int = 5) -> list[str]:
    summaries: list[str] = []
    stack = list(nodes)
    while stack and len(summaries) < limit:
        node = stack.pop(0)
        summary = node.get("summary")
        if summary:
            summaries.append(str(summary))
        stack.extend(node.get("nodes", []))
    return summaries[:limit]


def infer_page_count(raw_tree: dict[str, Any] | None) -> int | None:
    if not raw_tree:
        return None

    max_page = 0

    def walk(nodes: list[dict[str, Any]]) -> None:
        nonlocal max_page
        for node in nodes:
            end_page = node.get("end_index")
            if isinstance(end_page, int):
                max_page = max(max_page, end_page)
            walk(node.get("nodes", []))

    walk(raw_tree.get("structure", []))
    return max_page or None


def build_document_embedding_text(record: DocumentRecord) -> str:
    raw_tree = record.raw_tree or {}
    structure = raw_tree.get("structure", [])
    title = record.title or record.doc_name or "Untitled"
    description = record.doc_description or ""
    source_type = record.source_type or ""
    top_titles = extract_top_titles(structure)
    top_summaries = extract_top_summaries(structure)

    parts = [
        f"Title: {title}",
        f"Source type: {source_type}",
    ]
    if description:
        parts.append(f"Description: {description}")
    if top_titles:
        parts.append("Top sections: " + " | ".join(top_titles))
    if top_summaries:
        parts.append("Key summaries: " + " | ".join(top_summaries))
    return "\n".join(parts)


def generate_embedding(client: OpenAI, text_value: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text_value,
    )
    return list(response.data[0].embedding)


def ensure_document_embedding(db: Session, record: DocumentRecord, client: OpenAI | None = None) -> None:
    if record.status != "completed" or not record.raw_tree:
        return

    page_count = infer_page_count(record.raw_tree)
    changed = False
    if record.page_count != page_count:
        record.page_count = page_count
        changed = True

    embedding_text = build_document_embedding_text(record)
    if record.embedding_text != embedding_text:
        record.embedding_text = embedding_text
        changed = True

    if record.embedding is None or record.embedding_model != settings.embedding_model:
        if client is None:
            client = get_openai_client()
        record.embedding = generate_embedding(client, embedding_text)
        record.embedding_model = settings.embedding_model
        changed = True

    if changed:
        db.add(record)
        db.commit()
        db.refresh(record)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    }


def lexical_score(query: str, record: DocumentRecord) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    doc_text = " ".join(
        filter(
            None,
            [
                record.title,
                record.doc_name,
                record.doc_description,
                record.embedding_text,
            ],
        )
    )
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / max(1, len(query_tokens))


def format_selector_doc(record: DocumentRecord, score: float) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.doc_name or record.title,
        "status": record.status,
        "pageNum": record.page_count,
        "folderId": record.folder_id,
        "createdAt": record.created_at.isoformat() if record.created_at else None,
        "description": record.doc_description,
        "score": round(score, 4),
    }


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    return list(value)


def find_relevant_documents(
    db: Session,
    query: str,
    top_k: int,
    status: str = "completed",
) -> dict[str, Any]:
    client = get_openai_client()

    records = list(
        db.scalars(
            select(DocumentRecord).where(DocumentRecord.status == status).order_by(DocumentRecord.created_at.desc())
        )
    )

    if not records:
        return {
            "docs": [],
            "success": True,
            "has_more": False,
            "next_steps": {
                "options": [f'No documents are available for "{query}".'],
                "auto_retry": "Index a document first",
                "summary": f'Vector search for "{query}": 0 document(s) ranked by relevance',
            },
            "search_mode": "vector",
            "total_returned": 0,
        }

    for record in records:
        if record.embedding is None or record.embedding_model != settings.embedding_model:
            ensure_document_embedding(db, record, client=client)

    query_embedding = generate_embedding(client, query)
    candidates: list[tuple[DocumentRecord, float]] = []
    for record in records:
        embedding = normalize_embedding(record.embedding)
        if embedding is None:
            continue
        vector_score = _cosine_similarity(query_embedding, embedding)
        text_score = lexical_score(query, record)
        score = 0.8 * vector_score + 0.2 * text_score
        candidates.append((record, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = candidates[:top_k]

    docs = [format_selector_doc(record, score) for record, score in selected]
    total_returned = len(docs)

    return {
        "docs": docs,
        "success": True,
        "has_more": len(candidates) > top_k,
        "next_steps": {
            "options": [
                f'Found {total_returned} document(s) ranked by vector relevance to "{query}".',
                f'{total_returned} document(s) are ready for analysis. Use /search with the returned document ids.',
            ],
            "auto_retry": "Proceed to search the most relevant document",
            "summary": f'Vector search for "{query}": {total_returned} document(s) ranked by relevance',
        },
        "search_mode": "vector",
        "total_returned": total_returned,
    }
