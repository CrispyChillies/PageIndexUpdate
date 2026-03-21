"""
chatbot_ui.py — Streamlit chatbot interface for PageIndex JSON documents.

Run with:
    streamlit run chatbot_ui.py
"""

import json
import os

import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from pageindex.agentic_qa import AgenticPageIndexQA
from pageindex.doc_selector import load_json_documents, select_document_for_query

# ── Import traversal helpers from tree_traversal.py ──────────────────────────
from tree_traversal import (
    collect_all_node_ids,
    find_node_by_id,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PageIndex Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.03);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1.5rem;
}

/* Title area */
.main-header {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.main-header h1 {
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.8rem;
    margin: 0;
}

.main-header p {
    color: rgba(255,255,255,0.5);
    margin: 0.3rem 0 0 0;
    font-size: 0.9rem;
}

/* Chat messages */
.user-msg {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}

.user-msg-bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.93rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
}

.assistant-msg {
    display: flex;
    justify-content: flex-start;
    margin: 0.6rem 0;
}

.assistant-msg-bubble {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255,255,255,0.9);
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%;
    font-size: 0.93rem;
    line-height: 1.6;
    backdrop-filter: blur(10px);
}

/* Node cards in sidebar */
.node-card {
    background: rgba(167, 139, 250, 0.08);
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin: 0.4rem 0;
    font-size: 0.8rem;
}

.node-card .node-title {
    color: #a78bfa;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.node-card .node-pages {
    color: rgba(255,255,255,0.4);
    font-size: 0.72rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.35);
    color: #a78bfa;
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 500;
    margin-left: 0.5rem;
}

/* Chat input area */
.stChatInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Upload styling */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(139, 92, 246, 0.4);
    border-radius: 12px;
    padding: 0.8rem;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.82rem !important;
}

/* Scrollable chat box */
.chat-container {
    height: 60vh;
    overflow-y: auto;
    padding: 1rem;
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    margin-bottom: 1rem;
}

/* Thinking block */
.thinking-block {
    background: rgba(96, 165, 250, 0.07);
    border-left: 3px solid rgba(96, 165, 250, 0.5);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.5rem;
    font-style: italic;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.07) !important;
}

/* Selectbox, text_input */
.stSelectbox > div > div, .stTextInput > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load credentials ──────────────────────────────────────────────────────────
load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")
OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# ── Helper: create LLM client ─────────────────────────────────────────────────
@st.cache_resource
def get_client(model_name: str) -> OpenAI:
    if model_name.lower().startswith("qwen"):
        http_client = httpx.Client(verify=False)
        return OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, http_client=http_client)

    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)


# ── Helper: render tree in sidebar ───────────────────────────────────────────
def load_uploaded_json_documents(uploaded_files: list) -> list[dict]:
    """Load uploaded JSON files into the same metadata format as folder-loaded docs."""
    docs = []
    for uploaded in uploaded_files:
        try:
            uploaded.seek(0)
            raw = json.load(uploaded)
        except Exception:
            continue

        file_stem = os.path.splitext(uploaded.name)[0]
        docs.append(
            {
                "doc_id": raw.get("doc_id") or file_stem,
                "doc_title": raw.get("doc_title") or raw.get("doc_name") or uploaded.name,
                "doc_description": raw.get("doc_description", ""),
                "path": uploaded.name,
                "tree_data": raw,
            }
        )
    return docs


def render_tree_sidebar(nodes: list[dict], depth: int = 0):
    for node in nodes:
        nid = node.get("node_id", "?")
        title = node.get("title", "Untitled")
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        children = node.get("nodes", [])

        indent = "　" * depth  # em-space for visual indent
        label = f"{indent}**{title}**" if depth == 0 else f"{indent}{title}"
        pages = f"p. {start}–{end}"

        st.markdown(
            f"""<div class="node-card">
                <div class="node-title">{indent}[{nid}] {title}</div>
                <div class="node-pages">{pages}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if children:
            render_tree_sidebar(children, depth + 1)


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, nodes, thinking}
if "tree_data" not in st.session_state:
    st.session_state.tree_data = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []


# ══ SIDEBAR ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    source_mode = st.radio(
        "Document source",
        options=["Single JSON", "JSON Folder"],
        index=0,
        help="Use one uploaded file or route across multiple JSON files in a folder.",
    )

    model_name = st.text_input(
        "Model",
        value="gpt-4o-2024-11-20",
        help="Model name (Qwen/* for Qwen endpoint, otherwise OpenAI-compatible endpoint)",
    )

    adjacent_pages = st.slider(
        "Adjacent pages for full-text fallback",
        min_value=0,
        max_value=3,
        value=1,
        help="When summaries are insufficient, include nearby pages around selected nodes",
    )

    max_evidence_nodes = st.slider(
        "Max evidence nodes",
        min_value=1,
        max_value=12,
        value=6,
        help="Maximum number of selected nodes used to build grounded answer evidence",
    )

    st.markdown("### 📄 Document")
    if source_mode == "Single JSON":
        uploaded_files = st.file_uploader(
            "Upload PageIndex JSON (one or many)",
            type=["json"],
            accept_multiple_files=True,
            help="Upload the *_structure.json file generated by run_pageindex.py",
        )

        if uploaded_files:
            try:
                docs = load_uploaded_json_documents(uploaded_files)
                if not docs:
                    raise ValueError("No valid JSON files found in upload")

                if len(docs) == 1:
                    raw = docs[0]["tree_data"]
                    st.session_state.tree_data = raw
                    st.session_state.doc_name = raw.get("doc_name", uploaded_files[0].name)
                    st.session_state.loaded_docs = []
                    st.success(f"✅ Loaded: **{st.session_state.doc_name}**")
                else:
                    st.session_state.tree_data = None
                    st.session_state.doc_name = None
                    st.session_state.loaded_docs = docs
                    st.success(f"✅ Loaded {len(docs)} uploaded document(s)")

                # Clear chat when a new doc is loaded
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

        if st.session_state.tree_data:
            top_nodes = st.session_state.tree_data.get("structure", [])
            all_ids = collect_all_node_ids(top_nodes)

            st.markdown("---")
            st.markdown("### 🌲 Document Tree")

            col1, col2 = st.columns(2)
            col1.metric("Sections", len(top_nodes))
            col2.metric("Total Nodes", len(all_ids))

            with st.expander("View tree structure", expanded=False):
                render_tree_sidebar(top_nodes)
        elif st.session_state.loaded_docs:
            st.markdown("---")
            st.markdown("### 📚 Uploaded Documents")
            for doc in st.session_state.loaded_docs:
                st.markdown(
                    f"- **[{doc.get('doc_id')}] {doc.get('doc_title')}**\n"
                    f"  - source: {doc.get('path')}\n"
                    f"  - description: {doc.get('doc_description', '(no description)')}"
                )
    else:
        json_dir = st.text_input(
            "JSON folder path",
            value="./results",
            help="Folder containing one or more *_structure.json files",
        )
        if st.button("Load JSON folder", use_container_width=True):
            try:
                docs = load_json_documents(json_dir)
                st.session_state.loaded_docs = docs
                st.session_state.tree_data = None
                st.session_state.doc_name = None
                st.session_state.messages = []
                st.success(f"✅ Loaded {len(docs)} document(s) from {json_dir}")
            except Exception as e:
                st.error(f"Failed to load folder: {e}")

        if st.session_state.loaded_docs:
            st.markdown("---")
            st.markdown("### 📚 Available Documents")
            for doc in st.session_state.loaded_docs:
                st.markdown(
                    f"- **[{doc.get('doc_id')}] {doc.get('doc_title')}**\\n"
                    f"  - path: {doc.get('path')}\\n"
                    f"  - description: {doc.get('doc_description', '(no description)')}"
                )

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Credential status
    st.markdown("### 🔐 API Status")
    if model_name.lower().startswith("qwen"):
        if QWEN_API_KEY and QWEN_BASE_URL:
            st.markdown("🟢 Qwen credentials loaded from .env")
        else:
            st.markdown("🔴 Missing QWEN_API_KEY or QWEN_BASE_URL in .env")
    else:
        if OPENAI_API_KEY:
            st.markdown("🟢 OpenAI credentials loaded from .env")
        else:
            st.markdown("🔴 Missing CHATGPT_API_KEY or OPENAI_API_KEY in .env")


# ══ MAIN AREA ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>📄 PageIndex Chat</h1>
    <p>Chat with your documents using LLM-guided tree traversal • No vector DB • No chunking</p>
</div>
""", unsafe_allow_html=True)

# Guard: no doc loaded
if source_mode == "Single JSON" and st.session_state.tree_data is None and not st.session_state.loaded_docs:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: rgba(255,255,255,0.4);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: rgba(255,255,255,0.6); margin-bottom: 0.5rem;">
            No document loaded
        </div>
        <div style="font-size: 0.85rem;">
            Upload a <code>*_structure.json</code> file in the sidebar to start chatting.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if source_mode == "JSON Folder" and not st.session_state.loaded_docs:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: rgba(255,255,255,0.4);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📁</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: rgba(255,255,255,0.6); margin-bottom: 0.5rem;">
            No document folder loaded
        </div>
        <div style="font-size: 0.85rem;">
            Load a folder in the sidebar to enable multi-document routing.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Guard: missing credentials
if model_name.lower().startswith("qwen"):
    if not QWEN_API_KEY or not QWEN_BASE_URL:
        st.error("Qwen credentials missing. Set QWEN_API_KEY and QWEN_BASE_URL in .env.")
        st.stop()
else:
    if not OPENAI_API_KEY:
        st.error("OpenAI credentials missing. Set CHATGPT_API_KEY or OPENAI_API_KEY in .env.")
        st.stop()

# Document label
if source_mode == "Single JSON":
    if st.session_state.tree_data:
        current_tree = st.session_state.tree_data or {}
        doc_node_count = len(collect_all_node_ids(current_tree.get("structure", [])))
        st.markdown(
            f"Chatting with: **{st.session_state.doc_name}** "
            f"<span class='status-badge'>{doc_node_count} nodes</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"Multi-document upload mode "
            f"<span class='status-badge'>{len(st.session_state.loaded_docs)} docs loaded</span>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f"Multi-document routing mode "
        f"<span class='status-badge'>{len(st.session_state.loaded_docs)} docs loaded</span>",
        unsafe_allow_html=True,
    )
st.markdown("---")

# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

            selected_doc = msg.get("selected_doc")
            if selected_doc:
                st.markdown(
                    f"**Selected document:** [{selected_doc.get('doc_id')}] {selected_doc.get('doc_title')}"
                )
                if selected_doc.get("doc_description"):
                    st.markdown(f"**Description:** {selected_doc.get('doc_description')}")

            selection_debug = msg.get("selection_debug")
            if selection_debug:
                with st.expander("🧭 Document selection debug", expanded=False):
                    st.markdown(f"Reason: {selection_debug.get('reason', '')}")
                    st.markdown(f"Uncertain: {selection_debug.get('uncertain', False)}")
                    ranking = selection_debug.get("ranking", [])
                    if ranking:
                        for row in ranking:
                            st.markdown(
                                f"- [{row.get('doc_id')}] {row.get('doc_title')} | score={row.get('score')}"
                            )

            # Show retrieved nodes if present
            nodes = msg.get("nodes", [])
            if nodes:
                with st.expander(f"📌 Retrieved {len(nodes)} section(s) via tree traversal", expanded=False):
                    for node in nodes:
                        nid = node.get("node_id", "?")
                        title = node.get("title", "Untitled")
                        start = node.get("start_index", "?")
                        end = node.get("end_index", "?")
                        summary = node.get("summary", "")
                        st.markdown(
                            f"""<div class="node-card" style="max-width:100%;">
                                <div class="node-title">[{nid}] {title} &nbsp;·&nbsp; Pages {start}–{end}</div>
                                <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;margin-top:0.3rem;">{summary[:350]}{"..." if len(summary)>350 else ""}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # Show traversal thinking if present
            thinking = msg.get("thinking", "")
            if thinking:
                with st.expander("🧠 Traversal reasoning", expanded=False):
                    st.markdown(
                        f'<div class="thinking-block">{thinking}</div>',
                        unsafe_allow_html=True,
                    )


# ── Chat input ─────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the document...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Run traversal + generation
    with st.chat_message("assistant", avatar="🤖"):
        client = get_client(model_name)

        use_multi_doc_routing = source_mode == "JSON Folder" or (source_mode == "Single JSON" and bool(st.session_state.loaded_docs))

        if use_multi_doc_routing:
            docs = st.session_state.loaded_docs
            selection = select_document_for_query(query, docs)
            selected_doc = selection.get("selected_doc")

            if not selected_doc:
                answer = "Insufficient evidence: no document could be selected for this query."
                thinking_all = "Document selection failed."
                retrieved_nodes = []
                selection_debug = selection
                st.markdown(answer)
                with st.expander("🧭 Document selection debug", expanded=True):
                    st.markdown(f"Reason: {selection.get('reason', '')}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "nodes": retrieved_nodes,
                        "thinking": thinking_all,
                        "selected_doc": None,
                        "selection_debug": selection_debug,
                    }
                )
                st.stop()

            selection_debug = {
                "reason": selection.get("reason", ""),
                "uncertain": selection.get("uncertain", False),
                "ranking": selection.get("ranking", []),
            }

            st.markdown(
                f"**Selected document:** [{selected_doc.get('doc_id')}] {selected_doc.get('doc_title')}"
            )
            if selected_doc.get("doc_description"):
                st.markdown(f"**Description:** {selected_doc.get('doc_description')}")

            with st.expander("🧭 Document selection debug", expanded=False):
                st.markdown(f"Reason: {selection_debug.get('reason', '')}")
                st.markdown(f"Uncertain: {selection_debug.get('uncertain', False)}")
                for row in selection_debug.get("ranking", []):
                    st.markdown(f"- [{row.get('doc_id')}] {row.get('doc_title')} | score={row.get('score')}")

            if selection_debug.get("uncertain", False):
                answer = (
                    "Insufficient evidence: document routing is uncertain based on doc descriptions. "
                    "Please refine your query or choose a single document mode."
                )
                thinking_all = "Routing uncertain; traversal skipped."
                retrieved_nodes = []
                st.warning(answer)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "nodes": retrieved_nodes,
                        "thinking": thinking_all,
                        "selected_doc": {
                            "doc_id": selected_doc.get("doc_id"),
                            "doc_title": selected_doc.get("doc_title"),
                            "doc_description": selected_doc.get("doc_description", ""),
                        },
                        "selection_debug": selection_debug,
                    }
                )
                st.stop()

            target_tree_data = selected_doc.get("tree_data", {})
            selected_doc_info = {
                "doc_id": selected_doc.get("doc_id"),
                "doc_title": selected_doc.get("doc_title"),
                "doc_description": selected_doc.get("doc_description", ""),
            }
        else:
            target_tree_data = st.session_state.tree_data or {}
            selected_doc_info = {
                "doc_id": target_tree_data.get("doc_id") or st.session_state.doc_name,
                "doc_title": target_tree_data.get("doc_title") or target_tree_data.get("doc_name") or st.session_state.doc_name,
                "doc_description": target_tree_data.get("doc_description", ""),
            }
            selection_debug = {
                "reason": "Single-document mode.",
                "uncertain": False,
                "ranking": [
                    {
                        "doc_id": selected_doc_info.get("doc_id"),
                        "doc_title": selected_doc_info.get("doc_title"),
                        "score": 1.0,
                    }
                ],
            }

        with st.status("🤖 Running agentic QA (tree -> full text on demand)...", expanded=False) as status:
            agent = AgenticPageIndexQA(
                tree_data=target_tree_data,
                client=client,
                model=model_name,
            )

            status.write("Step 1: Tree traversal on summaries")
            status.write("Step 2: Sufficiency check")
            status.write("Step 3: Optional full-text retrieval")
            status.write("Step 4: Grounded answer with citations")

            result = agent.answer(
                query=query,
                adjacent_pages=adjacent_pages,
                max_evidence_nodes=max_evidence_nodes,
            )

            answer = result.get("answer", "")
            thinking_all = (
                f"Summary enough: {result.get('summary_enough', 'no')}\n"
                f"Used full text: {result.get('used_full_text', False)}\n"
                f"Sufficiency reason: {result.get('summary_sufficiency_reason', '')}\n"
                f"Evidence sufficient: {result.get('evidence_sufficient', 'no')}\n"
                f"Insufficient reason: {result.get('insufficient_reason', '')}"
            )

            top_nodes = target_tree_data.get("structure", [])
            retrieved_nodes = []
            for nid in result.get("retrieved_node_ids", []):
                node = find_node_by_id(top_nodes, nid)
                if node:
                    retrieved_nodes.append(node)

            status.update(label=f"✅ Completed with {len(retrieved_nodes)} retrieved node(s)", state="complete")

        st.markdown(answer)

        citations = result.get("citations", [])
        if citations:
            st.markdown("### Sources")
            for c in citations:
                nid = c.get("node_id", "?")
                title = c.get("title", "Untitled")
                start = c.get("start_index", "?")
                end = c.get("end_index", "?")
                st.markdown(f"- [{nid}] {title} (pages {start}-{end})")

        # Show retrieved nodes
        if retrieved_nodes:
            with st.expander(f"📌 Retrieved {len(retrieved_nodes)} section(s) via tree traversal", expanded=False):
                for node in retrieved_nodes:
                    nid = node.get("node_id", "?")
                    title = node.get("title", "Untitled")
                    start = node.get("start_index", "?")
                    end = node.get("end_index", "?")
                    summary = node.get("summary", "")
                    st.markdown(
                        f"""<div class="node-card" style="max-width:100%;">
                            <div class="node-title">[{nid}] {title} &nbsp;·&nbsp; Pages {start}–{end}</div>
                            <div style="color:rgba(255,255,255,0.6);font-size:0.8rem;margin-top:0.3rem;">{summary[:350]}{"..." if len(summary)>350 else ""}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Show traversal thinking
        if thinking_all:
            with st.expander("🧠 Traversal reasoning", expanded=False):
                st.markdown(
                    f'<div class="thinking-block">{thinking_all}</div>',
                    unsafe_allow_html=True,
                )

    # Persist assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "nodes": retrieved_nodes,
        "thinking": thinking_all,
        "selected_doc": selected_doc_info,
        "selection_debug": selection_debug,
    })
