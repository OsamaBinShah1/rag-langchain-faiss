import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

INDEX_PATH = "faiss_index"

def build_vectorstore(chunks: List[Document], embeddings) -> FAISS:
    print(f"[Retriever] Building FAISS index from {len(chunks)} chunks...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    return vs

def load_vectorstore(embeddings) -> FAISS:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("No FAISS index found. Run build_vectorstore() first.")
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def get_retriever(vectorstore: FAISS, k: int = 4):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
