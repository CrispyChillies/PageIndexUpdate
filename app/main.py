from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, engine, get_db, initialize_database, sync_document_schema
from app.document_selector import find_relevant_documents
from app.indexing import build_index_options, index_document, persist_upload
from app.models import DocumentRecord
from app.retrieval import (
    answer_with_pageindex,
    expand_node,
    retrieve_full_content,
    search_documents,
)
from app.schemas import (
    AnswerWithPageIndexRequest,
    AnswerWithPageIndexResponse,
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
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    Base.metadata.create_all(bind=engine)
    sync_document_schema()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "PageIndex retrieval toolkit. Recommended tool sequence for question answering: "
        "1) call find_relevant_documents if document ids are unknown, "
        "2) call search with selected document ids, "
        "3) call expand_node if a node is still broad, "
        "4) call retrieve_full_content to extract exact source text before answering."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "document-selection", "description": "Choose the best documents for a query."},
        {"name": "node-search", "description": "Find and refine relevant nodes inside documents."},
        {"name": "full-content", "description": "Extract exact raw text from PDF pages."},
        {"name": "documents", "description": "Create and inspect indexed documents."},
        {"name": "meta", "description": "Connector metadata and compatibility endpoints."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def derive_document_title(file: UploadFile | None, file_path: str | None) -> str | None:
    if file is not None and file.filename:
        return Path(file.filename).stem
    if file_path:
        return Path(file_path).stem
    return None


@app.post(
    "/documents",
    response_model=DocumentCreateResponse,
    status_code=202,
    tags=["documents"],
    summary="Register and index a document",
    description=(
        "Upload or register a local PDF/Markdown file, create a database record, and start "
        "background indexing. The response only confirms that indexing has started."
    ),
    operation_id="create_document",
)
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
        payload_dict["title"] = derive_document_title(
            file, payload_dict.get("file_path")
        )

    try:
        payload = DocumentCreate.model_validate(payload_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not payload.title:
        raise HTTPException(
            status_code=422, detail="Could not derive title from file name"
        )

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


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["documents"],
    summary="Get document status and indexed metadata",
    description=(
        "Return one indexed document record. Use this after POST /documents to check whether "
        "indexing is still processing, completed, or failed."
    ),
    operation_id="get_document",
)
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


@app.post(
    "/find_relevant_documents",
    response_model=FindRelevantDocumentsResponse,
    tags=["document-selection"],
    summary="Find the most relevant documents for a user query",
    description=(
        "First step of the retrieval workflow. Call this before /search when the user has not "
        "specified document ids. This returns the best candidate documents ranked by vector "
        "relevance and lexical overlap."
    ),
    operation_id="find_relevant_documents",
)
def find_documents(
    payload: FindRelevantDocumentsRequest, db: Session = Depends(get_db)
):
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


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["node-search"],
    summary="Search inside selected documents for relevant nodes",
    description=(
        "Second step of the retrieval workflow. After choosing candidate documents with "
        "/find_relevant_documents, call /search with those document ids to locate the most "
        "relevant nodes by title, summary, and page range."
    ),
    operation_id="search_documents",
)
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


@app.post(
    "/expand_node",
    response_model=ExpandNodeResponse,
    tags=["node-search"],
    summary="Expand one node to inspect its immediate children",
    description=(
        "Use this after /search when the returned node is still too broad. It returns the "
        "selected node and its immediate child nodes so the assistant can narrow the search."
    ),
    operation_id="expand_node",
)
def expand(payload: ExpandNodeRequest, db: Session = Depends(get_db)):
    try:
        result = expand_node(db, payload.document_id, payload.node_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExpandNodeResponse(**result)


@app.post(
    "/retrieve-full-content",
    response_model=RetrieveFullContentResponse,
    tags=["full-content"],
    summary="Retrieve the exact PDF text for a node page range",
    description=(
        "Final grounding step. Use this after /search or /expand_node to extract the raw PDF "
        "text for the selected page range. Prefer this endpoint before answering with quotes "
        "or exact evidence."
    ),
    operation_id="retrieve_full_content",
)
def retrieve_node_content(
    payload: RetrieveFullContentRequest, db: Session = Depends(get_db)
):
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


@app.post(
    "/answer_with_pageindex",
    response_model=AnswerWithPageIndexResponse,
    tags=["full-content"],
    summary="Run the full retrieval workflow and return final answer context",
    description=(
        "End-to-end retrieval endpoint for OpenWebUI integration. It selects the most relevant "
        "document, searches and ranks node summaries, asks an internal LLM whether summaries are "
        "sufficient, optionally retrieves full PDF text for the best nodes, and returns a final "
        "context block plus structured sources."
    ),
    operation_id="answer_with_pageindex",
)
def answer_with_pageindex_route(
    payload: AnswerWithPageIndexRequest, db: Session = Depends(get_db)
):
    try:
        result = answer_with_pageindex(
            db=db,
            query=payload.query,
            document_top_k=payload.document_top_k,
            node_top_k=payload.node_top_k,
            selected_node_limit=payload.selected_node_limit,
            status=payload.status,
            model=settings.retrieval_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AnswerWithPageIndexResponse(**result)


@app.get(
    "/api/config",
    tags=["meta"],
    summary="OpenWebUI connector metadata",
    description="Small metadata endpoint for OpenWebUI compatibility checks.",
    operation_id="get_api_config",
)
def get_api_config():
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "openapi_url": "/openapi.json",
        "workflow": [
            "Use find_relevant_documents when document ids are unknown.",
            "Use search after candidate documents are selected.",
            "Use expand_node if a search hit is too broad.",
            "Use retrieve_full_content before giving evidence-based answers or quotes.",
        ],
    }
