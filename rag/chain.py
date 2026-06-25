from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from rag.document_loader import load_documents, chunk_documents
from rag.embeddings import get_embeddings
from rag.retriever import build_vectorstore, load_vectorstore, get_retriever

class RAGChain:
    def __init__(self, docs_path=None, model="gpt-4o-mini", embedding_provider="openai", k=4, load_existing=False):
        self.embeddings = get_embeddings(provider=embedding_provider)
        if load_existing:
            vs = load_vectorstore(self.embeddings)
        else:
            docs = load_documents(docs_path)
            chunks = chunk_documents(docs)
            vs = build_vectorstore(chunks, self.embeddings)
        retriever = get_retriever(vs, k=k)
        llm = ChatOpenAI(model=model, temperature=0)
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        self.chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        print(f"[RAGChain] Ready. Model: {model}")

    def query(self, question: str) -> str:
        return self.chain.invoke({"question": question})["answer"]

    def reset_memory(self):
        self.chain.memory.clear()
