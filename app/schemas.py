from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class DocumentCreate(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="Local file path to a PDF or Markdown document to index.",
        examples=["./docs/sample.pdf"],
    )
    title: str | None = Field(
        default=None,
        description="Optional document title. If omitted, the API derives it from the file name.",
        examples=["Annual Report 2025"],
    )
    model: str | None = Field(default=None, description="LLM used during indexing.")
    toc_check_pages: int | None = Field(default=None, description="PDF TOC detection window.")
    max_pages_per_node: int | None = Field(default=None, description="Maximum page span per node.")
    max_tokens_per_node: int | None = Field(default=None, description="Maximum token span per node.")
    if_add_node_id: str | None = Field(default=None, description='Set to "yes" to include node ids.')
    if_add_node_summary: str | None = Field(default=None, description='Set to "yes" to include node summaries.')
    if_add_doc_description: str | None = Field(default="yes", description='Set to "yes" to include document description.')
    if_add_node_text: str | None = Field(default=None, description='Set to "yes" to store raw node text.')
    if_thinning: str | None = Field(default=None, description='Markdown-only thinning flag.')
    thinning_threshold: int | None = Field(default=None, description="Markdown thinning token threshold.")
    summary_token_threshold: int | None = Field(default=None, description="Markdown summary threshold.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_path": "./docs/sample.pdf",
                "title": "Annual Report 2025",
            }
        }
    }

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
    query: str = Field(
        description="User question to search within already selected documents.",
        examples=["What are the company's major risk factors?"],
    )
    document_ids: list[str] = Field(
        description="Candidate document ids returned by find_relevant_documents.",
        examples=[["7c69bf20-79ff-4dbc-8f8f-8758817c8637"]],
    )
    top_k: int = Field(default=5, description="Maximum number of node hits to return.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the company's major risk factors?",
                "document_ids": ["7c69bf20-79ff-4dbc-8f8f-8758817c8637"],
                "top_k": 5,
            }
        }
    }


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
    document_id: str = Field(description="Document id containing the node.")
    node_id: str = Field(description="Node id to expand.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "7c69bf20-79ff-4dbc-8f8f-8758817c8637",
                "node_id": "0014",
            }
        }
    }


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
    document_id: str = Field(description="Document id containing the PDF node.")
    node_id: str = Field(description="Node id whose pages should be extracted.")
    start_page: int = Field(description="First PDF page to extract, inclusive.")
    end_page: int = Field(description="Last PDF page to extract, inclusive.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "7c69bf20-79ff-4dbc-8f8f-8758817c8637",
                "node_id": "0014",
                "start_page": 22,
                "end_page": 26,
            }
        }
    }


class RetrieveFullContentResponse(BaseModel):
    document_id: str
    node_id: str
    title: str | None = None
    page_start: int
    page_end: int
    content: str


class FindRelevantDocumentsRequest(BaseModel):
    query: str = Field(
        description="User query used to select the most relevant documents before node-level search.",
        examples=["OPNT003 clinical studies NDA submission"],
    )
    top_k: int = Field(default=5, description="Maximum number of documents to return.")
    status: str = Field(default="completed", description="Only search documents in this indexing status.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "OPNT003 clinical studies NDA submission",
                "top_k": 5,
                "status": "completed",
            }
        }
    }


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


class AnswerWithPageIndexRequest(BaseModel):
    query: str = Field(
        description="User question for the end-to-end retrieval workflow.",
        examples=["What are the major risk factors discussed in this document?"],
    )
    document_top_k: int = Field(
        default=1,
        description="Number of documents to consider. Keep this at 1 for now for faster execution.",
    )
    node_top_k: int = Field(
        default=5,
        description="Maximum number of node summaries to retrieve from the selected document.",
    )
    selected_node_limit: int = Field(
        default=2,
        description="Maximum number of nodes to expand to full content when summaries are insufficient.",
    )
    status: str = Field(
        default="completed",
        description="Only search documents in this indexing status.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the major risk factors discussed in this document?",
                "document_top_k": 1,
                "node_top_k": 5,
                "selected_node_limit": 2,
                "status": "completed",
            }
        }
    }


class AnswerWithPageIndexSource(BaseModel):
    document_id: str
    node_id: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    content_type: str
    content: str | None = None


class AnswerWithPageIndexResponse(BaseModel):
    context: str
    sources: list[AnswerWithPageIndexSource]
