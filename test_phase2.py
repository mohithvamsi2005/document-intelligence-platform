from src.ingestion.pdf_loader import extract_text_from_pdf
from langchain.schema import Document

PDF_PATH = "data/sample.pdf"

print("Extracting text from:", PDF_PATH)
print("=" * 60)

documents = extract_text_from_pdf(PDF_PATH)

print(f"Total LangChain Documents: {len(documents)}")
print("=" * 60)

# Show structure of first document
first_doc = documents[0]

print("TYPE CHECK:")
print(f"  Is LangChain Document? {isinstance(first_doc, Document)}")
print()
print("PAGE CONTENT (first 300 chars):")
print(first_doc.page_content[:300])
print()
print("METADATA:")
for key, value in first_doc.metadata.items():
    print(f"  {key}: {value}")