import os
from dotenv import load_dotenv
from rag.chain import RAGChain

load_dotenv()

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) combines the power of LLMs with external
knowledge retrieval. Instead of relying solely on the model's training data, RAG
fetches relevant documents from a knowledge base and provides them as context to
the LLM, significantly reducing hallucinations and improving accuracy.

Large Language Models (LLMs) like GPT-4 and Claude are transformer-based models
trained on vast amounts of text data. They can generate human-like text, answer
questions, summarize documents, and perform many NLP tasks.
"""

def main():
    os.makedirs("docs", exist_ok=True)
    with open("docs/sample.txt", "w") as f:
        f.write(SAMPLE_TEXT)

    rag = RAGChain(docs_path="./docs", model="gpt-4o-mini")

    questions = [
        "What is Retrieval-Augmented Generation?",
        "How does RAG reduce hallucinations?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {rag.query(q)}\n")

if __name__ == "__main__":
    main()
