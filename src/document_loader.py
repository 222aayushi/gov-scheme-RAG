from langchain_community.document_loaders import PyPDFLoader
from chunking import chunk_documents
import os

folder_path = r"data doc\Scheme Guidelines pdf"

all_docs = []

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        all_docs.extend(docs)

print("Total pages loaded:", len(all_docs))

print("\nSample Text:\n")
print(all_docs[0].page_content[:500])
chunks = chunk_documents(all_docs)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0].page_content)