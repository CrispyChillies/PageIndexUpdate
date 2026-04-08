import json
import os
import re
from typing import Any

from openai import OpenAI
from sqlalchemy.orm import Session

from app.models import DocumentRecord
from pageindex.utils import get_text_of_pages

SYSTEM_PROMPT = """You are an expert document retrieval assistant.

Your task is to identify which nodes in a document tree are most likely to contain answer-bearing information for the user's query.

Output format (strict JSON only; no markdown, no code blocks, no extra text):
{
    "rationale": "short explanation",
    "node_list": ["node_id_1", "node_id_2"]
}

Rules:
- Return ONLY valid JSON. No extra text before or after.
- Prefer the most specific nodes that are sufficient to answer the query.
- Include a parent node only if the parent summary itself contains answer-bearing information.
- Avoid returning both a parent and all of its children unless both are independently useful.
- Return at most 4 node IDs per pass.
- If no node is relevant, return an empty node_list: [].
- Use the exact node_id strings from the tree.
"""

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


def get_retrieval_client() -> OpenAI:
    api_key = os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("CHATGPT_API_KEY or OPENAI_API_KEY must be set for retrieval")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def build_tree_text(nodes: list[dict[str, Any]], depth: int = 0) -> str:
    lines: list[str] = []
    indent = "  " * depth
    for node in nodes:
        node_id = node.get("node_id", "???")
        title = node.get("title", "(untitled)")
        summary = node.get("summary", "(no summary)")
        children = node.get("nodes", [])

        lines.append(f"{indent}[{node_id}] {title}")
        lines.append(f"{indent}  Summary: {summary}")
        if children:
            lines.append(f"{indent}  Sub-sections:")
            lines.extend(build_tree_text(children, depth + 2).splitlines())
    return "\n".join(lines)


def build_toplevel_text(nodes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for node in nodes:
        node_id = node.get("node_id", "???")
        title = node.get("title", "(untitled)")
        summary = node.get("summary", "(no summary)")
        children = node.get("nodes", [])
        child_note = f"  [has {len(children)} sub-section(s)]" if children else ""
        lines.append(f"[{node_id}] {title}{child_note}")
        lines.append(f"  Summary: {summary}")
    return "\n".join(lines)


def find_node_by_id(nodes: list[dict[str, Any]], target_id: str) -> dict[str, Any] | None:
    for node in nodes:
        if node.get("node_id") == target_id:
            return node
        found = find_node_by_id(node.get("nodes", []), target_id)
        if found:
            return found
    return None


def collect_child_node_ids(nodes: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for node in nodes:
        node_id = node.get("node_id")
        if node_id:
            ids.add(node_id)
        ids |= collect_child_node_ids(node.get("nodes", []))
    return ids


def call_llm(client: OpenAI, model: str, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"rationale": raw, "node_list": []}


def normalize_terms(text: str) -> set[str]:
    terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {term for term in terms if term not in STOPWORDS and len(term) > 1}


def compute_score(query: str, node: dict[str, Any], rank: int) -> float:
    query_terms = normalize_terms(query)
    haystack = " ".join(
        [
            str(node.get("title", "")),
            str(node.get("summary", "")),
            str(node.get("doc_description", "")),
        ]
    )
    node_terms = normalize_terms(haystack)
    if not query_terms:
        return max(0.1, 1.0 - rank * 0.05)

    overlap = len(query_terms & node_terms)
    coverage = overlap / len(query_terms)
    rank_bonus = max(0.0, 0.15 - rank * 0.02)
    title_bonus = 0.1 if normalize_terms(str(node.get("title", ""))) & query_terms else 0.0
    return round(min(0.99, coverage + rank_bonus + title_bonus), 2)


def traverse_tree(
    tree_data: dict[str, Any],
    query: str,
    client: OpenAI,
    model: str,
) -> list[dict[str, Any]]:
    top_nodes = tree_data.get("structure", [])
    top_level_ids = {node.get("node_id") for node in top_nodes if node.get("node_id")}
    max_top_level_candidates = 3
    max_nodes_per_pass = 4

    prompt_pass1 = (
        f"Query: {query}\n\n"
        "Document tree structure (top-level sections only):\n"
        f"{build_toplevel_text(top_nodes)}\n\n"
        f"Identify which top-level sections are relevant to the query. Return up to {max_top_level_candidates} candidate top-level node IDs."
    )
    result1 = call_llm(client, model, prompt_pass1)
    selected_top_raw = result1.get("node_list", [])
    if not isinstance(selected_top_raw, list):
        selected_top_raw = []
    selected_top = [
        node_id
        for node_id in selected_top_raw
        if isinstance(node_id, str) and node_id in top_level_ids
    ][:max_top_level_candidates]

    if not selected_top:
        return []

    final_node_ids: list[str] = []
    for node_id in selected_top:
        node = find_node_by_id(top_nodes, node_id)
        if not node:
            continue
        children = node.get("nodes", [])
        if not children:
            final_node_ids.append(node_id)
            continue

        subtree_text = build_tree_text(children)
        prompt_pass2 = (
            f"Query: {query}\n\n"
            f"You are looking inside the section '{node.get('title')}' (node {node_id}) which has the following sub-sections:\n"
            f"{subtree_text}\n\n"
            f"Which specific sub-section node(s) best answer the query? Also include the parent node {node_id} itself if its own summary is directly relevant."
        )
        result2 = call_llm(client, model, prompt_pass2)
        selected_children_raw = result2.get("node_list", [])
        if not isinstance(selected_children_raw, list):
            selected_children_raw = []
        allowed_ids = collect_child_node_ids(children) | {node_id}
        selected_children = [
            child_id
            for child_id in selected_children_raw
            if isinstance(child_id, str) and child_id in allowed_ids
        ][:max_nodes_per_pass]
        final_node_ids.extend(selected_children)

    deduped_ids: list[str] = []
    seen: set[str] = set()
    for node_id in final_node_ids:
        if node_id not in seen:
            seen.add(node_id)
            deduped_ids.append(node_id)

    results: list[dict[str, Any]] = []
    for node_id in deduped_ids:
        node = find_node_by_id(top_nodes, node_id)
        if node:
            results.append(node)
    return results


def build_hit(document: DocumentRecord, node: dict[str, Any], query: str, rank: int) -> dict[str, Any]:
    return {
        "document_id": document.id,
        "node_id": node.get("node_id"),
        "title": node.get("title"),
        "page_start": node.get("start_index"),
        "page_end": node.get("end_index"),
        "summary": node.get("summary"),
        "score": compute_score(query, node, rank),
    }


def search_documents(
    db: Session,
    query: str,
    document_ids: list[str],
    top_k: int,
    model: str,
) -> list[dict[str, Any]]:
    client = get_retrieval_client()
    hits: list[dict[str, Any]] = []
    for document_id in document_ids:
        record = db.get(DocumentRecord, document_id)
        if record is None or record.status != "completed" or not record.raw_tree:
            continue
        nodes = traverse_tree(record.raw_tree, query, client, model)
        for rank, node in enumerate(nodes, start=1):
            hit = build_hit(record, node, query, rank)
            hits.append(hit)

    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:top_k]


def expand_node(db: Session, document_id: str, node_id: str) -> dict[str, Any]:
    record = db.get(DocumentRecord, document_id)
    if record is None or record.status != "completed" or not record.raw_tree:
        raise ValueError("Document not found or not indexed")

    node = find_node_by_id(record.raw_tree.get("structure", []), node_id)
    if node is None:
        raise ValueError("Node not found")

    children = node.get("nodes", [])
    return {
        "document_id": document_id,
        "node_id": node.get("node_id"),
        "title": node.get("title"),
        "page_start": node.get("start_index"),
        "page_end": node.get("end_index"),
        "summary": node.get("summary"),
        "children": [
            {
                "node_id": child.get("node_id"),
                "title": child.get("title"),
                "page_start": child.get("start_index"),
                "page_end": child.get("end_index"),
                "summary": child.get("summary"),
            }
            for child in children
        ],
    }


def retrieve_full_content(
    db: Session,
    document_id: str,
    node_id: str,
    start_page: int,
    end_page: int,
) -> dict[str, Any]:
    record = db.get(DocumentRecord, document_id)
    if record is None or record.status != "completed" or not record.raw_tree:
        raise ValueError("Document not found or not indexed")
    if record.source_type != "pdf":
        raise ValueError("Full-content retrieval is only supported for PDF documents")
    if not record.source_path or not os.path.exists(record.source_path):
        raise ValueError("Source PDF file not found")
    if start_page <= 0 or end_page <= 0 or start_page > end_page:
        raise ValueError("Invalid page range")

    node = find_node_by_id(record.raw_tree.get("structure", []), node_id)
    if node is None:
        raise ValueError("Node not found")

    content = get_text_of_pages(record.source_path, start_page, end_page, tag=False)
    return {
        "document_id": document_id,
        "node_id": node_id,
        "title": node.get("title"),
        "page_start": start_page,
        "page_end": end_page,
        "content": content,
    }
