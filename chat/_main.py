import os
import glob
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from .llm import embeddings_model, llm_model
from langchain_core.prompts import PromptTemplate

# --------------------
# Config from env
# --------------------

FAISS_DIR = "faiss_store"         # where FAISS index lives
PDF_DIR = "./docs"                # where your PDFs are
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# https://gist.github.com/sinsunsan/436ce36f5e66dca544d48df504d6974d

def load_pdfs(pdf_dir: str):
    """Load and return LangChain Documents from every PDF in a folder."""
    all_docs = []
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True))
    if not pdf_paths:
        print(f"[WARN] No PDFs found under: {pdf_dir}")
    for path in pdf_paths:
        print('files found - {}'.format(path))
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # add a source field for traceability
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = path
        all_docs.extend(docs)
    return all_docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_embeddings():
    """Build FAISS from PDFs and persist to disk."""
    print("[INFO] Loading PDFs…")
    raw_docs = load_pdfs(PDF_DIR)
    if not raw_docs:
        print("[INFO] Nothing to index.")
        return

    print(f"[INFO] Chunking {len(raw_docs)} pdf pages …")
    chunks = chunk_docs(raw_docs)
    print(f"[INFO] Total chunks: {len(chunks)}")

    print("[INFO] Creating FAISS index … (this may take a moment)")
    vs = FAISS.from_documents(chunks, embedding=embeddings_model)

    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(FAISS_DIR)
    print(f"[SUCCESS] FAISS index saved to ./{FAISS_DIR}")

def load_index():
    """Load a persisted FAISS index with the same OCI embedding handle."""
    vs = FAISS.load_local(
        FAISS_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True  # due to pickle for LC metadata
    )
    return vs

def make_qa_chain(vs: FAISS):
    """Create a RetrievalQA chain with an OCI GenAI chat model."""
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # custom_prompt = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=(
    #         "You are an AI assistant who answers questions politely and helpfully.\n"
    #         "Always use professional, friendly, and clear language.\n"
    #         "If you don't know the answer, politely say so.\n\n"
    #         "Context:\n{context}\n\n"
    #         "Question: {question}\n\n"
    #         "Answer in a polite and concise way:"
    #     ),
    # )
    chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff",  # simple & effective for many PDFs
        return_source_documents=True,
        # chain_type_kwargs={"prompt": custom_prompt},
    )
    return chain

def interactive_query():
    """Load FAISS and start an interactive prompt."""
    print("[INFO] Loading FAISS store …")
    vs = load_index()
    chain = make_qa_chain(vs)
    print("[READY] Ask questions about your PDFs. Type 'exit' to quit.\n")
    while True:
        q = input("Q: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        resp = chain({"query": q})
        print("\nA:", resp["result"].strip(), "\n")
        # Show sources
        srcs = resp.get("source_documents") or []
        if srcs:
            print("Sources:")
            for i, d in enumerate(srcs, 1):
                print(f"  [{i}] {d.metadata.get('source')}  (chars={len(d.page_content)})")
        print()

if __name__ == "__main__":
    # build_embeddings()
    interactive_query()
