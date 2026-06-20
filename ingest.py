"""Build and save a FAISS vector index from a source PDF.

Run this script after placing one PDF in the ``data/`` folder:

    python ingest.py

You can also pass a specific PDF path:

    python ingest.py data/my_document.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"


def find_pdf_path() -> Path:
    """Return the PDF to ingest, either from argv or from the data folder."""
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return pdf_path

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            "No PDF found in data/. Add a PDF there or run: python ingest.py path/to/file.pdf"
        )
    if len(pdf_files) > 1:
        names = ", ".join(str(path) for path in pdf_files)
        raise ValueError(
            f"Found multiple PDFs: {names}. Please choose one with: python ingest.py data/file.pdf"
        )

    return pdf_files[0]


def main() -> None:
    """Load a PDF, chunk it, embed it, and save a FAISS index."""
    # Load environment variables from .env, including OPENAI_API_KEY.
    load_dotenv()

    pdf_path = find_pdf_path()
    print(f"Loading PDF: {pdf_path}")

    # PyPDFLoader reads each PDF page into LangChain Document objects.
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # RecursiveCharacterTextSplitter breaks documents into chunks that are
    # small enough for retrieval while preserving nearby context with overlap.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)

    # OpenAIEmbeddings converts each chunk into a numeric vector that FAISS can
    # search by semantic similarity. The API key is read from OPENAI_API_KEY.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # FAISS.from_documents embeds the chunks and builds an in-memory FAISS index.
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the index files locally so future runs can load them without
    # re-embedding the same PDF.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))

    print(f"Created {len(chunks)} chunks.")
    print(f"FAISS index saved to: {INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()
