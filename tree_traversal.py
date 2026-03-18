"""
tree_traversal.py — LLM-guided tree traversal for PageIndex JSON documents.

Uses a two-pass strategy:
  1. Top-level scan: identify which top-level branches are relevant.
  2. Drill-down: for each relevant branch that has children, identify the
     specific child nodes that answer the query.

Usage:
  python tree_traversal.py \
      --json_path results/test_structure.json \
      --query "What is CRAR and how does it work?"
"""

import argparse
import json
import os
import re
import sys
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load credentials from .env
# ---------------------------------------------------------------------------
load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")

# ---------------------------------------------------------------------------
# System prompt — instructs the LLM on how to traverse the tree
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert document retrieval assistant.

Your task is to identify which nodes in a document tree structure are most likely to contain information relevant to the user's query.

## How to approach this:
1. Read the user's query carefully to understand what information is being sought.
2. Examine each node's title and summary in the provided tree structure.
3. Reason step-by-step about which nodes are semantically relevant to the query.
4. Consider both direct matches (node explicitly covers the query topic) and indirect matches (node contains context that helps answer the query).
5. If the query is about a specific sub-topic, prefer the most specific (deepest) node in the tree that covers it.

## Output format (strict JSON — no markdown, no code blocks):
{
  "thinking": "<your step-by-step reasoning about which nodes are relevant and why>",
  "node_list": ["node_id_1", "node_id_2", ...]
}

Rules:
- Return ONLY valid JSON. No extra text before or after.
- Include ALL node IDs that are relevant, including parent nodes if the parent's own summary (not just children) is relevant.
- If no node is relevant, return an empty node_list: [].
- Use the exact node_id strings from the tree (e.g., "0001", "0005").
"""

# ---------------------------------------------------------------------------
# Helper: build a compact tree representation for the LLM
# ---------------------------------------------------------------------------

def build_tree_text(nodes: list[dict], depth: int = 0, show_children: bool = True) -> str:
    """Recursively render nodes as indented text for the LLM prompt."""
    lines = []
    indent = "  " * depth
    for node in nodes:
        node_id = node.get("node_id", "???")
        title = node.get("title", "(untitled)")
        summary = node.get("summary", "(no summary)")
        children = node.get("nodes", [])

        lines.append(f"{indent}[{node_id}] {title}")
        lines.append(f"{indent}  Summary: {summary}")
        if children and show_children:
            lines.append(f"{indent}  Sub-sections:")
            lines.extend(build_tree_text(children, depth + 2, show_children=True).splitlines())

    return "\n".join(lines)


def build_toplevel_text(nodes: list[dict]) -> str:
    """Only show top-level nodes (hide their children) for the first pass."""
    lines = []
    for node in nodes:
        node_id = node.get("node_id", "???")
        title = node.get("title", "(untitled)")
        summary = node.get("summary", "(no summary)")
        children = node.get("nodes", [])
        has_children_note = f"  [has {len(children)} sub-section(s)]" if children else ""
        lines.append(f"[{node_id}] {title}{has_children_note}")
        lines.append(f"  Summary: {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: find a node by node_id anywhere in the tree
# ---------------------------------------------------------------------------

def find_node_by_id(nodes: list[dict], target_id: str) -> dict | None:
    for node in nodes:
        if node.get("node_id") == target_id:
            return node
        children = node.get("nodes", [])
        found = find_node_by_id(children, target_id)
        if found:
            return found
    return None


def collect_all_node_ids(nodes: list[dict]) -> set[str]:
    """Collect all node_id values in the tree (for validation)."""
    ids = set()
    for node in nodes:
        nid = node.get("node_id")
        if nid:
            ids.add(nid)
        ids |= collect_all_node_ids(node.get("nodes", []))
    return ids


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model: str, user_prompt: str) -> dict:
    """Call the Qwen LLM and parse the JSON response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n[WARN] Failed to parse LLM response as JSON: {e}")
        print(f"Raw response:\n{raw}\n")
        return {"thinking": raw, "node_list": []}


# ---------------------------------------------------------------------------
# Core traversal logic
# ---------------------------------------------------------------------------

def traverse(
    tree_data: dict,
    query: str,
    client: OpenAI,
    model: str,
    verbose: bool = True,
) -> list[dict]:
    """
    Two-pass traversal:
      Pass 1 — Top-level nodes only (no children shown).
      Pass 2 — For each top-level node selected that has children,
                drill into the subtree to find the most specific nodes.
    Returns a list of retrieved node dicts with their metadata.
    """
    doc_name = tree_data.get("doc_name", "unknown")
    top_nodes: list[dict] = tree_data.get("structure", [])
    all_ids = collect_all_node_ids(top_nodes)

    # ---- Pass 1: top-level scan ----
    print(f"\n{'='*60}")
    print(f"DOCUMENT: {doc_name}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    print("\n[Pass 1] Scanning top-level sections...\n")

    tree_text_toplevel = build_toplevel_text(top_nodes)
    prompt_pass1 = (
        f"Query: {query}\n\n"
        f"Document tree structure (top-level sections only):\n"
        f"{tree_text_toplevel}\n\n"
        f"Identify which top-level sections are relevant to the query."
    )

    result1 = call_llm(client, model, prompt_pass1)
    thinking1 = result1.get("thinking", "")
    selected_top = result1.get("node_list", [])

    # Validate IDs
    selected_top = [nid for nid in selected_top if nid in all_ids]

    if verbose:
        print(f"Thinking (Pass 1):\n{thinking1}\n")
        print(f"Selected top-level nodes: {selected_top}")

    # ---- Pass 2: drill-down into sub-sections ----
    final_node_ids: set[str] = set()

    for nid in selected_top:
        node = find_node_by_id(top_nodes, nid)
        if not node:
            continue
        children = node.get("nodes", [])

        if not children:
            # Leaf node — add directly
            final_node_ids.add(nid)
        else:
            print(f"\n[Pass 2] Drilling into node [{nid}] '{node.get('title')}'...\n")
            subtree_text = build_tree_text(children, depth=0, show_children=True)
            prompt_pass2 = (
                f"Query: {query}\n\n"
                f"You are looking inside the section '{node.get('title')}' "
                f"(node {nid}) which has the following sub-sections:\n"
                f"{subtree_text}\n\n"
                f"Which specific sub-section node(s) best answer the query? "
                f"Also include the parent node {nid} itself if its own summary "
                f"(not just its children) is directly relevant."
            )
            result2 = call_llm(client, model, prompt_pass2)
            thinking2 = result2.get("thinking", "")
            selected_children = result2.get("node_list", [])
            selected_children = [c for c in selected_children if c in all_ids]

            if verbose:
                print(f"Thinking (Pass 2 for {nid}):\n{thinking2}\n")
                print(f"Selected child nodes: {selected_children}")

            final_node_ids.update(selected_children)

    # ---- Collect results ----
    retrieved_nodes = []
    for nid in final_node_ids:
        node = find_node_by_id(top_nodes, nid)
        if node:
            retrieved_nodes.append(node)

    return retrieved_nodes


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(retrieved_nodes: list[dict], query: str) -> None:
    print(f"\n{'='*60}")
    print(f"RETRIEVAL RESULTS for: '{query}'")
    print(f"{'='*60}")
    if not retrieved_nodes:
        print("No relevant nodes found.")
        return
    for i, node in enumerate(retrieved_nodes, 1):
        nid = node.get("node_id", "?")
        title = node.get("title", "(untitled)")
        summary = node.get("summary", "(no summary)")
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        print(f"\n[{i}] Node ID: {nid} | '{title}' | Pages {start}–{end}")
        print(f"     Summary: {summary[:300]}{'...' if len(summary) > 300 else ''}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-guided tree traversal for PageIndex JSON documents"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the PageIndex JSON tree file (e.g. results/test_structure.json)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query to search for in the document",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Model name to use (default: Qwen/Qwen2.5-32B-Instruct)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print LLM reasoning for each pass (default: True)",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    # Load tree
    with open(args.json_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    # Build Qwen client
    if not QWEN_API_KEY or not QWEN_BASE_URL:
        print("Error: QWEN_API_KEY and QWEN_BASE_URL must be set in .env", file=sys.stderr)
        sys.exit(1)

    # Disable SSL verification for the local/self-signed Qwen endpoint
    # (same approach used by pageindex/utils.py)
    http_client = httpx.Client(verify=False)
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, http_client=http_client)

    # Run traversal
    retrieved_nodes = traverse(
        tree_data=tree_data,
        query=args.query,
        client=client,
        model=args.model,
        verbose=args.verbose,
    )

    # Print results
    print_results(retrieved_nodes, args.query)


if __name__ == "__main__":
    main()
