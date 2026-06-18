from pathlib import Path
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from chunking import chunk_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data doc"

all_docs = []

# =====================
# LOAD PDF DOCUMENTS
# =====================

for pdf_path in DATA_ROOT.rglob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    all_docs.extend(docs)

print("Total PDF pages loaded:", len(all_docs))

# =====================
# LOAD CSV DATASET
# =====================

CSV_PATH = DATA_ROOT / "updated_data.csv"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("CSV_PATH =", CSV_PATH)
print("CSV EXISTS =", CSV_PATH.exists())

if CSV_PATH.exists():

    df = pd.read_csv(CSV_PATH)

    csv_docs = []

    for _, row in df.iterrows():

        text = ""

        for col in df.columns:
            value = row[col]

            if pd.notna(value):
                text += f"{col}: {value}\n"

        csv_docs.append(
            Document(page_content=text)
        )

    all_docs.extend(csv_docs)

    print("CSV records loaded:", len(csv_docs))

# =====================
# DEBUG OUTPUT
# =====================

print("Total documents loaded:", len(all_docs))

if all_docs:

    print("\nSample Text:\n")
    print(all_docs[0].page_content[:500])

    chunks = chunk_documents(all_docs)

    print("Total Chunks:", len(chunks))

    print("\nFirst Chunk:\n")
    print(chunks[0].page_content)

else:
    print("No documents found.")