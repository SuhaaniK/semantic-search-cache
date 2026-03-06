from data.load_data import load_dataset
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore

docs, labels, cats = load_dataset()

embedder = Embedder()

embeddings = embedder.embed(docs)

vector_store = VectorStore(embeddings.shape[1])

vector_store.add_documents(embeddings, docs)

print("Index built with", len(docs), "documents")