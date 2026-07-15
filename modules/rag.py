# ============================================================
# modules/rag.py — RAG pipeline for document search
# Handles: ChromaDB storage, retriever, QA chain
# Used by: app.py (Document Search and Ask AI pages)
#
# UPDATED: clear_vectorstore() now handles a Windows-specific
# ChromaDB file-locking issue. On Windows, SQLite (which ChromaDB
# uses internally) can keep a file handle open on the persisted
# vector_db folder even after the Python Chroma object goes out of
# scope — causing PermissionError: [WinError 32] when a later
# clear_vectorstore() call tries to delete it. This does NOT happen
# on Linux/Mac, which is why it wasn't caught earlier.
#
# Fix: force garbage collection + retry deletion with short waits,
# and fall back to file-by-file cleanup (skipping locked files)
# rather than crashing the app if Windows still won't release the
# handle after retries.
# ============================================================

import gc
import os
import shutil
import time
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import (
    VECTOR_DB_PATH,
    RETRIEVER_K_SINGLE,
    RETRIEVER_K_MERGED
)
from modules.llm import load_embeddings, load_llm


def format_docs(docs):
    formatted = []
    for doc in docs:
        filename = os.path.basename(
            doc.metadata.get("source", "Unknown")
        )
        page = doc.metadata.get("page", "unknown")
        lang = doc.metadata.get("language", "unknown")
        method = doc.metadata.get("extraction_method", "unknown")
        method_label = (
            "Direct Read" if method == "direct_text"
            else "Groq Vision AI"
        )
        formatted.append(
            f"[Document: {filename} | Page: {page} | "
            f"Language: {lang} | Method: {method_label}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def get_prompts():
    single = ChatPromptTemplate.from_template("""
You are a senior property valuation analyst working for a bank.
Answer the question using ONLY facts explicitly stated in the context.
Do NOT infer or assume anything not directly written.
Mention which document the answer came from.
If not found say: "This information was not found in the uploaded documents."

Context:
{context}

Question: {input}
""")

    merged = ChatPromptTemplate.from_template("""
You are a senior property valuation analyst working for a bank.
These documents belong to DIFFERENT OWNERS whose plots have been
MERGED to construct one building. This is a merged property valuation.

STRICT INSTRUCTIONS:
1. Give details SEPARATELY for each document/owner.
2. After per-document details provide a MERGED SUMMARY:
   - Total combined area
   - Combined outer boundary
   - Note about multiple owners for bank records
3. Use ONLY facts explicitly stated in the context.
4. Format with document name as heading for each section.

Context:
{context}

Question: {input}
""")
    return single, merged


def build_vectorstore(all_chunks):
    # Build ChromaDB from all document chunks
    embeddings = load_embeddings()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    return vectorstore


def load_vectorstore():
    # Load existing ChromaDB from disk
    embeddings = load_embeddings()
    if os.path.exists(VECTOR_DB_PATH):
        return Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    return None


def clear_vectorstore():
    """
    Clears the vector_db folder before building a new vectorstore.

    UPDATED for Windows compatibility — see module docstring above
    for why this is needed. Behavior on success is unchanged: the
    folder is deleted and recreated empty, exactly as before.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)
        return

    # Release any lingering Chroma/SQLite connection objects before
    # attempting deletion. Streamlit can keep a prior run's objects
    # alive slightly longer than expected, so this needs to happen
    # right before the delete attempt, not just once at import time.
    gc.collect()

    max_retries = 5
    deleted = False
    for attempt in range(max_retries):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            deleted = True
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                gc.collect()
            # else: fall through to file-by-file fallback below

    if not deleted:
        # Last resort — delete what we can, file by file, skipping
        # anything still locked, rather than crashing the app.
        # A few leftover files won't break the next build_vectorstore()
        # call since Chroma creates a fresh collection regardless.
        for root, dirs, files in os.walk(VECTOR_DB_PATH, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except PermissionError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    pass

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)


def build_qa_chain(vectorstore, is_merged=False):
    llm = load_llm()
    single_prompt, merged_prompt = get_prompts()
    active_prompt = merged_prompt if is_merged else single_prompt

    k = RETRIEVER_K_MERGED if is_merged else RETRIEVER_K_SINGLE
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    qa_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | active_prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever