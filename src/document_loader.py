from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from chunking import chunk_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data doc"

all_docs = []

for pdf_path in DATA_ROOT.rglob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    all_docs.extend(docs)

print("Total pages loaded:", len(all_docs))

if all_docs:
    print("\nSample Text:\n")
    print(all_docs[0].page_content[:500])

    chunks = chunk_documents(all_docs)

    print("Total Chunks:", len(chunks))

    print("\nFirst Chunk:\n")
    print(chunks[0].page_content)
else:
    print("No PDF files found under data doc/.")