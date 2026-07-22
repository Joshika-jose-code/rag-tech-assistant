# streamlit_app.py
"""
Streamlit UI for the RAG Technical Documentation Assistant.

This is a thin client over the FastAPI backend (app/main.py) - it talks to
it exclusively over HTTP via `requests`, the same way `curl` does in the
README examples. Run the backend separately first:

    uvicorn app.main:app --reload

then in another terminal:

    streamlit run streamlit_app.py
"""
import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120  # seconds; generation + grading + retries can be slow on Groq's free tier

st.set_page_config(page_title="RAG Technical Documentation Assistant", page_icon="📚", layout="wide")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {question, answer, sources, meta, feedback}


def backend_url() -> str:
    return st.session_state.backend_url.rstrip("/")


def send_feedback(question: str, answer: str, rating: str) -> None:
    try:
        requests.post(
            f"{backend_url()}/feedback",
            json={"question": question, "answer": answer, "rating": rating},
            timeout=10,
        )
    except requests.RequestException as e:
        st.toast(f"Couldn't record feedback: {e}", icon="⚠️")


def render_status_badges(meta: dict) -> str:
    badges = []
    if meta.get("is_fallback"):
        badges.append("🟡 Fallback response")
    if meta.get("used_web_search"):
        badges.append("🌐 Used web search")
    if meta.get("grounded") is True:
        badges.append("✅ Grounded")
    elif meta.get("grounded") is False:
        badges.append("⚠️ Possibly ungrounded")
    if meta.get("retries_used"):
        badges.append(f"🔁 {meta['retries_used']} retrieval retr{'y' if meta['retries_used'] == 1 else 'ies'}")
    if meta.get("hallucination_retries_used"):
        badges.append(f"🔁 {meta['hallucination_retries_used']} regeneration"
                       f"{'s' if meta['hallucination_retries_used'] != 1 else ''}")
    return " · ".join(badges)


# ---------------------------------------------------------------------------
# Sidebar: connection, indexed documents, ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)

    try:
        requests.get(f"{backend_url()}/health", timeout=5).raise_for_status()
        st.success("Backend reachable")
    except requests.RequestException:
        st.error("Backend unreachable - is `uvicorn app.main:app` running?")

    st.divider()
    st.header("📄 Indexed Documents")
    if st.button("Refresh"):
        st.rerun()
    try:
        docs = requests.get(f"{backend_url()}/documents", timeout=10).json()
        if docs:
            for d in docs:
                st.write(f"**{d['filename']}** — {d['chunk_count']} chunks")
        else:
            st.caption("Nothing indexed yet.")
    except requests.RequestException as e:
        st.caption(f"Could not load documents: {e}")

    st.divider()
    st.header("➕ Add Documents")
    tab_files, tab_urls = st.tabs(["Upload files", "From URLs"])

    with tab_files:
        uploaded = st.file_uploader("Choose file(s)", accept_multiple_files=True)
        if st.button("Ingest files", disabled=not uploaded, key="ingest_files_btn"):
            files = [("files", (f.name, f.getvalue())) for f in uploaded]
            try:
                with st.spinner("Ingesting..."):
                    resp = requests.post(f"{backend_url()}/ingest/files", files=files, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                st.success(f"Added {resp.json()['chunks_added']} chunks.")
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {e}")

    with tab_urls:
        urls_text = st.text_area("One URL per line")
        if st.button("Ingest URLs", disabled=not urls_text.strip(), key="ingest_urls_btn"):
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            try:
                with st.spinner("Fetching and ingesting..."):
                    resp = requests.post(f"{backend_url()}/ingest/urls", json={"urls": urls}, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                st.success(f"Added {resp.json()['chunks_added']} chunks.")
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {e}")


# ---------------------------------------------------------------------------
# Main area: chat
# ---------------------------------------------------------------------------

st.title("📚 RAG Technical Documentation Assistant")
st.caption("Ask a question about the indexed FastAPI documentation.")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(msg["question"])

    with st.chat_message("assistant"):
        st.markdown(msg["answer"])

        badges = render_status_badges(msg["meta"])
        if badges:
            st.caption(badges)

        if msg["sources"]:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}**" + (f" · score {s['score']:.3f}" if s.get("score") is not None else ""))
                    st.text(s["snippet"])
                    st.divider()

        if msg["feedback"] is None:
            col1, col2, _ = st.columns([1, 1, 10])
            if col1.button("👍", key=f"up_{i}"):
                send_feedback(msg["question"], msg["answer"], "up")
                msg["feedback"] = "up"
                st.rerun()
            if col2.button("👎", key=f"down_{i}"):
                send_feedback(msg["question"], msg["answer"], "down")
                msg["feedback"] = "down"
                st.rerun()
        else:
            st.caption(f"Feedback recorded: {'👍' if msg['feedback'] == 'up' else '👎'}")

question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{backend_url()}/query", json={"question": question}, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.messages.append({
                    "question": question,
                    "answer": data["answer"],
                    "sources": data.get("sources", []),
                    "meta": {
                        "is_fallback": data.get("is_fallback"),
                        "grounded": data.get("grounded"),
                        "used_web_search": data.get("used_web_search"),
                        "retries_used": data.get("retries_used"),
                        "hallucination_retries_used": data.get("hallucination_retries_used"),
                    },
                    "feedback": None,
                })
            except requests.RequestException as e:
                st.session_state.messages.append({
                    "question": question,
                    "answer": f"Could not reach the backend: {e}",
                    "sources": [],
                    "meta": {},
                    "feedback": None,
                })
    st.rerun()
