from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

def load_documents(source: str) -> List[Document]:
    path = Path(source)
    if path.is_file():
        return _load_single_file(path)
    elif path.is_dir():
        return _load_directory(path)
    raise FileNotFoundError(f"Path not found: {source}")

def _load_single_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Unsupported: {ext}")
    return loader.load()

def _load_directory(path: Path) -> List[Document]:
    docs = []
    for ext in SUPPORTED_EXTENSIONS:
        loader = DirectoryLoader(str(path), glob=f"**/*{ext}", show_progress=True)
        try:
            docs.extend(loader.load())
        except Exception:
            pass
    return docs

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[Loader] {len(documents)} docs → {len(chunks)} chunks")
    return chunks
