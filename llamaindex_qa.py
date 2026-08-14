import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # <-- NEW IMPORT

# 1. Load environment variables
load_dotenv()

# 2. Configure LlamaIndex to use Gemini for answering questions (Ignore the deprecation warning in console)
Settings.llm = Gemini(model="gemini-3.1-flash-lite", temperature=0)

# 3. Configure LlamaIndex to use a specific local HuggingFace model for embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 4. Load documents from the folder
print("Loading documents...")
documents = SimpleDirectoryReader("your_docs").load_data()
print(f"Loaded {len(documents)} document page(s)/file(s).")

# 5. Build the VectorStoreIndex
print("Building vector index...")
index = VectorStoreIndex.from_documents(documents)

# 6. Create a query engine and execute a query (UPDATED: fetch top 5 chunks)
query_engine = index.as_query_engine(similarity_top_k=5)

query_text = "On the document what it said about Embeddings and Softmax"
print(f"\nQuerying: '{query_text}'\n")

response = query_engine.query(query_text)


# 7. Print the final answer and source nodes
print("=== ANSWER ===")
print(response)

print("\n=== SOURCE NODES (Retrieved Chunks) ===")
for i, node in enumerate(response.source_nodes, 1):
    score = node.score if node.score is not None else 0.0
    snippet = node.node.get_text().replace("\n", " ")[:150]
    print(f"[{i}] Score: {score:.4f} | Snippet: {snippet}...")