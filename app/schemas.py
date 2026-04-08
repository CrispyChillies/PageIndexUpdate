from datetime import datetime

from pydantic import BaseModel, model_validator


class DocumentCreate(BaseModel):
    file_path: str | None = None
    title: str | None = None
    model: str | None = None
    toc_check_pages: int | None = None
    max_pages_per_node: int | None = None
    max_tokens_per_node: int | None = None
    if_add_node_id: str | None = None
    if_add_node_summary: str | None = None
    if_add_doc_description: str | None = "yes"
    if_add_node_text: str | None = None
    if_thinning: str | None = None
    thinning_threshold: int | None = None
    summary_token_threshold: int | None = None

    @model_validator(mode="after")
    def ensure_file_input(self):
        if not self.file_path:
            raise ValueError("file_path is required when no file upload is provided")
        return self


class DocumentCreateResponse(BaseModel):
    document_id: str
    status: str


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    status: str
    source_path: str | None = None
    source_type: str | None = None
    pageindex_doc_id: str | None = None
    doc_name: str | None = None
    doc_description: str | None = None
    raw_tree: dict | None = None
    index_options: dict | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class SearchRequest(BaseModel):
    query: str
    document_ids: list[str]
    top_k: int = 5


class SearchHit(BaseModel):
    document_id: str
    node_id: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class ExpandNodeRequest(BaseModel):
    document_id: str
    node_id: str


class ExpandNodeChild(BaseModel):
    node_id: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None


class ExpandNodeResponse(BaseModel):
    document_id: str
    node_id: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None
    children: list[ExpandNodeChild]


class RetrieveFullContentRequest(BaseModel):
    document_id: str
    node_id: str
    start_page: int
    end_page: int


class RetrieveFullContentResponse(BaseModel):
    document_id: str
    node_id: str
    title: str | None = None
    page_start: int
    page_end: int
    content: str


class FindRelevantDocumentsRequest(BaseModel):
    query: str
    top_k: int = 5
    status: str = "completed"


class RelevantDocument(BaseModel):
    id: str
    name: str | None = None
    status: str
    pageNum: int | None = None
    folderId: str | None = None
    createdAt: str | None = None
    description: str | None = None
    score: float | None = None


class NextSteps(BaseModel):
    options: list[str]
    auto_retry: str
    summary: str


class FindRelevantDocumentsResponse(BaseModel):
    docs: list[RelevantDocument]
    success: bool
    has_more: bool
    next_steps: NextSteps
    search_mode: str
    total_returned: int
