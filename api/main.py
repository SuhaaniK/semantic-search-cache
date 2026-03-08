from fastapi import FastAPI
from pydantic import BaseModel

from data.load_data import load_dataset
from embeddings.embedder import Embedder
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache

import numpy as np

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str


# Initialize system components
embedder = Embedder()
cache = SemanticCache(threshold=0.85)

print("Loading dataset...")
docs, labels, cats = load_dataset()

print("Generating document embeddings...")
doc_embeddings = embedder.embed(docs)

print("Training fuzzy clustering...")
cluster_model = FuzzyCluster(n_clusters=15)
cluster_model.fit(doc_embeddings)

print("System ready.")


@app.get("/")
def home():
    return {"message": "Semantic Cache API Running"}


@app.post("/query")
def query(request: QueryRequest):

    query_text = request.query

    # 1️⃣ Embed query
    query_embedding = embedder.embed([query_text])[0]

    # 2️⃣ Check semantic cache
    hit, entry, sim = cache.lookup(query_embedding)

    if hit:
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": int(entry["cluster"])
        }

    # 3️⃣ If cache miss → search dataset
    sims = np.dot(doc_embeddings, query_embedding)

    best_idx = int(np.argmax(sims))
    result_doc = docs[best_idx]

    # 4️⃣ Predict cluster membership
    cluster_probs = cluster_model.get_membership(
        query_embedding.reshape(1, -1)
    )

    dominant_cluster = int(np.argmax(cluster_probs))

    # 5️⃣ Store in cache
    cache.add(
        query=query_text,
        embedding=query_embedding,
        result=result_doc,
        cluster=dominant_cluster
    )

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_doc,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}