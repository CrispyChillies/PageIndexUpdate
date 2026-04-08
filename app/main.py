from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, engine, get_db, initialize_database, sync_document_schema
from app.document_selector import find_relevant_documents
from app.indexing import build_index_options, index_document, persist_upload
from app.models import DocumentRecord
from app.retrieval import expand_node, retrieve_full_content, search_documents
from app.schemas import (
    DocumentCreate,
    DocumentCreateResponse,
    DocumentResponse,
    ExpandNodeRequest,
    ExpandNodeResponse,
    FindRelevantDocumentsRequest,
    FindRelevantDocumentsResponse,
    RetrieveFullContentRequest,
    RetrieveFullContentResponse,
    SearchRequest,
    SearchResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    Base.metadata.create_all(bind=engine)
    sync_document_schema()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


def derive_document_title(file: UploadFile | None, file_path: str | None) -> str | None:
    if file is not None and file.filename:
        return Path(file.filename).stem
    if file_path:
        return Path(file_path).stem
    return None


@app.post("/documents", response_model=DocumentCreateResponse, status_code=202)
async def create_document(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile | None = File(default=None),
):
    content_type = request.headers.get("content-type", "")
    payload_dict = {}
    if "application/json" in content_type:
        payload_dict = await request.json()
    else:
        form = await request.form()
        payload_dict = dict(form)

    if file is not None:
        payload_dict["file_path"] = str(persist_upload(file))

    if not file and not payload_dict.get("file_path"):
        raise HTTPException(status_code=422, detail="Provide either file or file_path")

    if not payload_dict.get("title"):
        payload_dict["title"] = derive_document_title(file, payload_dict.get("file_path"))

    try:
        payload = DocumentCreate.model_validate(payload_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not payload.title:
        raise HTTPException(status_code=422, detail="Could not derive title from file name")

    record = DocumentRecord(
        title=payload.title,
        status="processing",
        source_path=payload.file_path,
        index_options=build_index_options(payload),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    background_tasks.add_task(index_document, record.id)
    return DocumentCreateResponse(document_id=record.id, status=record.status)


@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    record = db.get(DocumentRecord, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        document_id=record.id,
        title=record.title,
        status=record.status,
        source_path=record.source_path,
        source_type=record.source_type,
        pageindex_doc_id=record.pageindex_doc_id,
        doc_name=record.doc_name,
        doc_description=record.doc_description,
        raw_tree=record.raw_tree,
        index_options=record.index_options,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
    )


@app.post("/find_relevant_documents", response_model=FindRelevantDocumentsResponse)
def find_documents(payload: FindRelevantDocumentsRequest, db: Session = Depends(get_db)):
    try:
        result = find_relevant_documents(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
            status=payload.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FindRelevantDocumentsResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db)):
    if not payload.document_ids:
        raise HTTPException(status_code=422, detail="document_ids cannot be empty")
    hits = search_documents(
        db=db,
        query=payload.query,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        model=settings.retrieval_model,
    )
    return SearchResponse(hits=hits)


@app.post("/expand_node", response_model=ExpandNodeResponse)
def expand(payload: ExpandNodeRequest, db: Session = Depends(get_db)):
    try:
        result = expand_node(db, payload.document_id, payload.node_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExpandNodeResponse(**result)


@app.post("/retrieve-full-content", response_model=RetrieveFullContentResponse)
def retrieve_node_content(payload: RetrieveFullContentRequest, db: Session = Depends(get_db)):
    try:
        result = retrieve_full_content(
            db=db,
            document_id=payload.document_id,
            node_id=payload.node_id,
            start_page=payload.start_page,
            end_page=payload.end_page,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 422 if "Invalid page range" in detail else 404
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return RetrieveFullContentResponse(**result)
