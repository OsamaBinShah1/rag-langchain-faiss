from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(provider: str = "openai", model: str = None):
    if provider == "openai":
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    raise ValueError(f"Unknown provider: {provider}")
