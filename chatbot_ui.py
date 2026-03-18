"""
chatbot_ui.py — Streamlit chatbot interface for PageIndex JSON documents.

Run with:
    streamlit run chatbot_ui.py
"""

import json
import os
import re

import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── Import traversal helpers from tree_traversal.py ──────────────────────────
from tree_traversal import (
    SYSTEM_PROMPT,
    build_toplevel_text,
    build_tree_text,
    call_llm,
    collect_all_node_ids,
    find_node_by_id,
    traverse,
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

# ── Helper: create LLM client ─────────────────────────────────────────────────
@st.cache_resource
def get_client(api_key: str, base_url: str) -> OpenAI:
    http_client = httpx.Client(verify=False)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# ── Helper: answer generation ─────────────────────────────────────────────────
ANSWER_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on document excerpts.

You are given:
1. The user's question.
2. A set of relevant sections retrieved from a document, each with a title, page range, and summary.

Your task:
- Answer the question clearly and concisely using ONLY the information in the provided sections.
- Cite which section(s) you drew information from (e.g., "According to the Methodology section, ...").
- If the sections do not contain enough information to answer, say so honestly.
- Format your answer in clean markdown with bullet points or headings where helpful.
"""

def generate_answer(
    query: str,
    retrieved_nodes: list[dict],
    client: OpenAI,
    model: str,
    chat_history: list[dict],
) -> str:
    """Generate a final answer from the retrieved nodes + conversation history."""
    if not retrieved_nodes:
        return "I couldn't find any relevant sections in the document for your query."

    # Build context from retrieved nodes
    context_parts = []
    for i, node in enumerate(retrieved_nodes, 1):
        title = node.get("title", "Untitled")
        summary = node.get("summary", "")
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        context_parts.append(
            f"[Section {i}: {title} | Pages {start}–{end}]\n{summary}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_content = (
        f"Question: {query}\n\n"
        f"Relevant document sections:\n\n{context}"
    )

    # Build messages with history (last 6 turns for context)
    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ── Helper: render tree in sidebar ───────────────────────────────────────────
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


# ══ SIDEBAR ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_name = st.text_input(
        "Model",
        value="Qwen/Qwen2.5-32B-Instruct",
        help="Model name exposed by your Qwen endpoint",
    )

    st.markdown("### 📄 Document")

    uploaded_file = st.file_uploader(
        "Upload PageIndex JSON",
        type=["json"],
        help="Upload the *_structure.json file generated by run_pageindex.py",
    )

    if uploaded_file is not None:
        try:
            raw = json.load(uploaded_file)
            st.session_state.tree_data = raw
            st.session_state.doc_name = raw.get("doc_name", uploaded_file.name)
            # Clear chat when a new doc is loaded
            st.session_state.messages = []
            st.success(f"✅ Loaded: **{st.session_state.doc_name}**")
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

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Credential status
    st.markdown("### 🔐 API Status")
    if QWEN_API_KEY and QWEN_BASE_URL:
        st.markdown("🟢 Qwen credentials loaded from `.env`")
    else:
        st.markdown("🔴 Missing `QWEN_API_KEY` or `QWEN_BASE_URL` in `.env`")


# ══ MAIN AREA ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>📄 PageIndex Chat</h1>
    <p>Chat with your documents using LLM-guided tree traversal • No vector DB • No chunking</p>
</div>
""", unsafe_allow_html=True)

# Guard: no doc loaded
if st.session_state.tree_data is None:
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

# Guard: missing credentials
if not QWEN_API_KEY or not QWEN_BASE_URL:
    st.error("Qwen API credentials not found. Make sure `QWEN_API_KEY` and `QWEN_BASE_URL` are set in your `.env` file.")
    st.stop()

# Document label
doc_node_count = len(collect_all_node_ids(st.session_state.tree_data.get("structure", [])))
st.markdown(
    f"Chatting with: **{st.session_state.doc_name}** "
    f"<span class='status-badge'>{doc_node_count} nodes</span>",
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
        client = get_client(QWEN_API_KEY, QWEN_BASE_URL)

        # ---- Step 1: Tree traversal ----
        with st.status("🌲 Traversing document tree...", expanded=False) as status:
            tree_data = st.session_state.tree_data
            top_nodes = tree_data.get("structure", [])
            all_ids = collect_all_node_ids(top_nodes)

            # Pass 1
            status.write("Pass 1: Scanning top-level sections...")
            tree_text_toplevel = build_toplevel_text(top_nodes)
            prompt_pass1 = (
                f"Query: {query}\n\n"
                f"Document tree structure (top-level sections only):\n"
                f"{tree_text_toplevel}\n\n"
                f"Identify which top-level sections are relevant to the query."
            )
            result1 = call_llm(client, model_name, prompt_pass1)
            thinking_all = result1.get("thinking", "")
            selected_top = [
                nid for nid in result1.get("node_list", []) if nid in all_ids
            ]
            status.write(f"→ Selected top-level nodes: {selected_top}")

            # Pass 2
            final_node_ids: set[str] = set()
            for nid in selected_top:
                node = find_node_by_id(top_nodes, nid)
                if not node:
                    continue
                children = node.get("nodes", [])
                if not children:
                    final_node_ids.add(nid)
                else:
                    status.write(f"Pass 2: Drilling into [{nid}] '{node.get('title')}'...")
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
                    result2 = call_llm(client, model_name, prompt_pass2)
                    thinking_all += "\n\n" + result2.get("thinking", "")
                    selected_children = [
                        c for c in result2.get("node_list", []) if c in all_ids
                    ]
                    final_node_ids.update(selected_children)
                    status.write(f"→ Selected: {selected_children}")

            retrieved_nodes = [
                find_node_by_id(top_nodes, nid)
                for nid in final_node_ids
                if find_node_by_id(top_nodes, nid)
            ]

            status.update(label=f"✅ Found {len(retrieved_nodes)} relevant section(s)", state="complete")

        # ---- Step 2: Answer generation ----
        with st.spinner("✍️ Generating answer..."):
            # Build history without node/thinking metadata for the LLM context
            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude the just-added user msg
            ]
            answer = generate_answer(
                query=query,
                retrieved_nodes=retrieved_nodes,
                client=client,
                model=model_name,
                chat_history=history_for_llm,
            )

        st.markdown(answer)

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
    })
