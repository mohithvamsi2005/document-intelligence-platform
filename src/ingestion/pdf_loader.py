"""
PDF Loader Module
------------------
Extracts text from PDF files and returns LangChain Document objects.
LangChain's entire ecosystem (splitters, vector stores, retrievers)
expects Document objects — so we convert at the source, not downstream.

Document structure:
    Document(
        page_content = "raw text from the page",
        metadata     = {"source": "file.pdf", "page_number": 1}
    )
"""

import os
from typing import List
import fitz  # PyMuPDF
from langchain.schema import Document


def extract_text_from_pdf(file_path: str) -> List[Document]:
    """
    Extracts text from every page of a single PDF file.

    Args:
        file_path: Full path to the PDF file on disk.

    Returns:
        A list of LangChain Document objects, one per page with text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    filename = os.path.basename(file_path)
    documents = []

    doc = fitz.open(file_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text()

        if text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": filename,
                        "page_number": page_index + 1,
                        "total_pages": len(doc),
                        "file_path": file_path
                    }
                )
            )

    doc.close()
    return documents


def extract_text_from_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Runs extraction on multiple PDFs and returns one combined
    list of LangChain Documents across all files.
    """
    all_documents = []

    for path in file_paths:
        try:
            docs = extract_text_from_pdf(path)
            all_documents.extend(docs)
            print(f"✅ Extracted {len(docs)} pages from '{os.path.basename(path)}'")
        except Exception as e:
            print(f"❌ Failed to extract '{path}': {e}")

    return all_documents