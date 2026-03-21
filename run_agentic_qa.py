import argparse
import json
import logging
import os
import sys

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from pageindex.agentic_qa import AgenticPageIndexQA


def build_client(model: str) -> OpenAI:
    load_dotenv()
    if model.lower().startswith("qwen"):
        qwen_api_key = os.getenv("QWEN_API_KEY", "")
        qwen_base_url = os.getenv("QWEN_BASE_URL", "")

        if not qwen_api_key or not qwen_base_url:
            raise ValueError("QWEN_API_KEY and QWEN_BASE_URL must be set in .env for Qwen models")

        http_client = httpx.Client(verify=False)
        return OpenAI(api_key=qwen_api_key, base_url=qwen_base_url, http_client=http_client)

    openai_api_key = os.getenv("CHATGPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    if not openai_api_key:
        raise ValueError("CHATGPT_API_KEY or OPENAI_API_KEY must be set in .env for non-Qwen models")

    if openai_base_url:
        return OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    return OpenAI(api_key=openai_api_key)


def format_citation_line(citation: dict) -> str:
    node_id = citation.get("node_id", "?")
    title = citation.get("title", "(untitled)")
    start = citation.get("start_index", "?")
    end = citation.get("end_index", "?")
    return f"- [{node_id}] {title} | pages {start}-{end}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic PageIndex QA over tree + on-demand full text")
    parser.add_argument("--json_path", type=str, required=True, help="Path to PageIndex structure JSON")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="LLM model")
    parser.add_argument(
        "--adjacent-pages",
        type=int,
        default=0,
        help="When summaries are insufficient, include +/- this many adjacent pages around each selected node",
    )
    parser.add_argument(
        "--max-evidence-nodes",
        type=int,
        default=6,
        help="Maximum number of nodes to include in evidence synthesis",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Optional source document path override (otherwise loaded from JSON metadata)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("agentic_qa_cli")

    if not os.path.isfile(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    with open(args.json_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    try:
        client = build_client(args.model)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    agent = AgenticPageIndexQA(
        tree_data=tree_data,
        client=client,
        model=args.model,
        source_path=args.source_path,
        logger=logger,
    )

    result = agent.answer(
        query=args.query,
        adjacent_pages=max(0, args.adjacent_pages),
        max_evidence_nodes=max(1, args.max_evidence_nodes),
    )

    print("\n" + "=" * 70)
    print("AGENTIC PAGEINDEX QA RESULT")
    print("=" * 70)
    print(f"Query: {args.query}")
    print(f"Evidence sufficient: {result.get('evidence_sufficient', 'no')}")
    print(f"Summary enough: {result.get('summary_enough', 'no')}")
    print(f"Used full text: {result.get('used_full_text', False)}")

    insuff_reason = result.get("insufficient_reason", "")
    if insuff_reason:
        print(f"Insufficient reason: {insuff_reason}")

    print("\nAnswer:")
    print(result.get("answer", ""))

    citations = result.get("citations", [])
    print("\nCitations:")
    if citations:
        for citation in citations:
            print(format_citation_line(citation))
    else:
        print("- (none)")


if __name__ == "__main__":
    main()
