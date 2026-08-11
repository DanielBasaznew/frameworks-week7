import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# 1. Path to Frankenstein text file in Week 5 folder
file_path = os.path.join("..", "private-knowledge-assistant-week5", "data", "raw_docs", "frankenstein.txt")

print(f"Loading document from: {file_path}...")
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Chunking
print("Chunking document...")
docs = [Document(page_content=raw_text)]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

# 3. BM25 Keyword Search
print("Building BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 4. Semantic Vector Search
print("Building Semantic Retriever...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Ensemble Hybrid Search (RRF)
print("Building Ensemble (Hybrid) Retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

# 6. Execute Test Query
if __name__ == "__main__":
    query = "Who created the monster and what was his motivation?"
    print(f"\nExecuting Query: '{query}'\n")

    results = hybrid_retriever.invoke(query)

    print("--- HYBRID SEARCH RESULTS ---")
    for i, doc in enumerate(results):
        print(f"Rank {i+1}:\n{doc.page_content.strip()}\n" + "-"*40)