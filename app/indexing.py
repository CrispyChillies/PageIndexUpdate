import asyncio
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.config import settings
from app.database import SessionLocal
from app.document_selector import ensure_document_embedding, infer_page_count
from app.models import DocumentRecord
from pageindex import page_index_main
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader


def persist_upload(upload: UploadFile) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    upload_name = Path(upload.filename or "upload.bin").name
    target = settings.upload_dir / f"{uuid.uuid4()}_{upload_name}"
    with target.open("wb") as output:
        shutil.copyfileobj(upload.file, output)
    return target.resolve()


def build_index_options(payload) -> dict:
    return {
        "model": payload.model or settings.default_model,
        "toc_check_pages": payload.toc_check_pages or settings.toc_check_pages,
        "max_pages_per_node": payload.max_pages_per_node or settings.max_pages_per_node,
        "max_tokens_per_node": payload.max_tokens_per_node or settings.max_tokens_per_node,
        "if_add_node_id": payload.if_add_node_id or settings.if_add_node_id,
        "if_add_node_summary": payload.if_add_node_summary or settings.if_add_node_summary,
        "if_add_doc_description": payload.if_add_doc_description or settings.if_add_doc_description,
        "if_add_node_text": payload.if_add_node_text or settings.if_add_node_text,
        "if_thinning": payload.if_thinning or settings.if_thinning,
        "thinning_threshold": payload.thinning_threshold or settings.thinning_threshold,
        "summary_token_threshold": payload.summary_token_threshold
        or settings.summary_token_threshold,
    }


def index_document(document_id: str) -> None:
    with SessionLocal() as db:
        record = db.get(DocumentRecord, document_id)
        if record is None:
            return

        options = record.index_options or {}
        source_path = record.source_path
        if not source_path:
            record.status = "failed"
            record.error_message = "Missing source_path"
            record.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        if not os.path.exists(source_path):
            record.status = "failed"
            record.error_message = f"Source file not found: {source_path}"
            record.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        try:
            suffix = Path(source_path).suffix.lower()
            if suffix == ".pdf":
                user_opt = {
                    "model": options["model"],
                    "toc_check_page_num": options["toc_check_pages"],
                    "max_page_num_each_node": options["max_pages_per_node"],
                    "max_token_num_each_node": options["max_tokens_per_node"],
                    "if_add_node_id": options["if_add_node_id"],
                    "if_add_node_summary": options["if_add_node_summary"],
                    "if_add_doc_description": options["if_add_doc_description"],
                    "if_add_node_text": options["if_add_node_text"],
                }
                opt = ConfigLoader().load(user_opt)
                result = page_index_main(source_path, opt)
            elif suffix in {".md", ".markdown"}:
                md_result = asyncio.run(
                    md_to_tree(
                        md_path=source_path,
                        if_thinning=options["if_thinning"].lower() == "yes",
                        min_token_threshold=options["thinning_threshold"],
                        if_add_node_summary=options["if_add_node_summary"],
                        summary_token_threshold=options["summary_token_threshold"],
                        model=options["model"],
                        if_add_doc_description=options["if_add_doc_description"],
                        if_add_node_text=options["if_add_node_text"],
                        if_add_node_id=options["if_add_node_id"],
                    )
                )
                result = {
                    "doc_name": md_result.get("doc_name", Path(source_path).stem),
                    "structure": md_result.get("structure", md_result),
                }
                if "doc_description" in md_result:
                    result["doc_description"] = md_result["doc_description"]
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            result["source_path"] = os.path.abspath(source_path)
            result["source_type"] = "pdf" if suffix == ".pdf" else "markdown"
            result["generated_at"] = datetime.now(timezone.utc).isoformat()
            result["index_options"] = options

            record.status = "completed"
            record.source_type = result.get("source_type")
            record.pageindex_doc_id = result.get("doc_id") or result.get("doc_name")
            record.doc_name = result.get("doc_name")
            record.doc_description = result.get("doc_description")
            record.page_count = infer_page_count(result)
            record.raw_tree = result
            record.completed_at = datetime.now(timezone.utc)
            record.error_message = None
            db.commit()
            ensure_document_embedding(db, record)
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            record.completed_at = datetime.now(timezone.utc)
            db.commit()
