from __future__ import annotations

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")


def _api_get(path: str, api_url: str, timeout: int = 30):
    return requests.get(f"{api_url}{path}", timeout=timeout)


def _api_post(path: str, api_url: str, json=None, files=None, timeout: int = 60):
    return requests.post(f"{api_url}{path}", json=json, files=files, timeout=timeout)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for source in sources:
        document = source.get("document", "unknown")
        page = source.get("page", "na")
        st.caption(f"- {document} (page {page})")


def _render_history(history: list[dict]) -> None:
    for item in history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            _render_sources(item.get("sources", []))


def _ask_question(api_url: str, question: str):
    response = _api_post("/api/ask", api_url, json={"question": question})
    response.raise_for_status()
    return response.json()


def _upload_pdf(api_url: str, file_name: str, file_bytes: bytes):
    files = {"file": (file_name, file_bytes, "application/pdf")}
    response = _api_post("/api/upload", api_url, files=files)
    response.raise_for_status()
    return response.json()


def _load_documents(api_url: str) -> list[dict]:
    response = _api_get("/api/documents", api_url)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
    st.title("DocuMind - Manufacturing Document Q&A")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.sidebar.header("DocuMind")
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

    try:
        documents = _load_documents(api_url)
    except Exception:
        documents = []

    st.sidebar.subheader("Loaded Documents")
    if documents:
        for doc in documents:
            st.sidebar.caption(f"- {doc.get('name')} ({doc.get('pages', 0)} pages)")
    else:
        st.sidebar.caption("No documents found.")

    page = st.sidebar.radio("Page", ["Ask Questions", "Upload & Ask"])

    if page == "Ask Questions":
        _render_history(st.session_state.history)
        question = st.chat_input("Ask a question")
        if question:
            with st.spinner("Thinking..."):
                try:
                    result = _ask_question(api_url, question)
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    return
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                }
            )
            st.rerun()

    if page == "Upload & Ask":
        upload = st.file_uploader("Upload a PDF", type=["pdf"])
        if upload is not None:
            if st.button("Upload and ingest"):
                with st.spinner("Uploading and ingesting..."):
                    try:
                        _upload_pdf(api_url, upload.name, upload.getvalue())
                        st.success("Upload complete. You can ask questions now.")
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")
                        return
        _render_history(st.session_state.history)
        question = st.chat_input("Ask a question about uploaded PDFs")
        if question:
            with st.spinner("Thinking..."):
                try:
                    result = _ask_question(api_url, question)
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    return
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
