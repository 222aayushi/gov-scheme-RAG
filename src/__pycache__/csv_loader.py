import pandas as pd
from langchain_core.documents import Document

def load_csv_documents(csv_path):
    df = pd.read_csv(csv_path)

    docs = []

    for _, row in df.iterrows():

        text = ""

        for col in df.columns:
            value = row[col]

            if pd.notna(value):
                text += f"{col}: {value}\n"

        docs.append(
            Document(page_content=text)
        )

    return docs