from langchain_core.documents import Document

def chunk_documents(documents):
    chunks = []

    for doc in documents:
        text = doc.page_content

        sections = text.split("\n\n")

        for sec in sections:
            sec = sec.strip()

            if len(sec) > 150:
                chunks.append(Document(page_content=sec))

    return chunks