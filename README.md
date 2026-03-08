# semantic-search-cache
Trademarkia - AI&amp;ML Engineer Task

README.md
.gitignore → Python
License → optional


# Semantic Search with Intelligent Caching

This project implements a **semantic search system with an intelligent caching layer**. The goal is to reduce redundant computation by detecting when new queries are semantically similar to previously processed queries.

Instead of performing a full search every time, the system checks a **semantic cache** to determine whether a similar query has already been answered.

This project demonstrates the use of **transformer embeddings, vector similarity search, fuzzy clustering, and API-based deployment**.

---

# System Architecture

The system is composed of several modular components.

User Query
↓
Embedding Generation
↓
Semantic Cache Lookup
↓
(Cache Hit → Return Cached Result)
(Cache Miss → Perform Vector Search)
↓
Cluster Prediction
↓
Store Result in Cache
↓
Return Result

---

# Technologies Used

* Python
* FastAPI
* NumPy
* Scikit-Learn
* SentenceTransformers
* Uvicorn

Embedding Model:

* all-MiniLM-L6-v2

Dataset:

* 20 Newsgroups Dataset

---

# Project Structure

```
semantic-search-cache/
│
├── api/
│   └── main.py
│
├── cache/
│   └── semantic_cache.py
│
├── clustering/
│   └── fuzzy_cluster.py
│
├── data/
│   └── load_data.py
│
├── embeddings/
│   └── embedder.py
│
├── preprocessing/
│   └── preprocess.py
│
├── test_cluster.py
├── test_cache.py
└── README.md
```

Each module is designed to perform a specific function in the semantic search pipeline.

---

# Core Components

## 1. Dataset Loader

The dataset loader imports documents from the 20 Newsgroups dataset. These documents serve as the searchable knowledge base.

The loader returns:

* documents
* labels
* category names

---

## 2. Text Preprocessing

Text data is normalized before embedding generation. This ensures consistency across documents and queries.

Typical preprocessing steps include:

* lowercasing
* whitespace normalization
* noise removal

---

## 3. Embedding Generation

Text is converted into dense vector representations using a transformer model.

Model used:

all-MiniLM-L6-v2

Each document and query is represented as a **384-dimensional vector** capturing semantic meaning.

Example:

```
"machine learning tools"
→ [0.21, -0.44, 0.91, ...]
```

These embeddings allow the system to perform **semantic similarity search** rather than simple keyword matching.

---

## 4. Vector Search

When a cache miss occurs, the query embedding is compared against all document embeddings using cosine similarity.

The most similar document is returned as the result.

Similarity is calculated using:

Cosine Similarity

Values range from:

1 → identical meaning
0 → unrelated

---

## 5. Fuzzy Clustering

Document embeddings are grouped using a **Gaussian Mixture Model (GMM)**.

Unlike hard clustering algorithms, fuzzy clustering allows documents to belong to multiple clusters with different probabilities.

Example membership distribution:

```
Cluster 2 → 0.74
Cluster 5 → 0.18
Cluster 9 → 0.08
```

This reflects the fact that documents often contain **multiple overlapping topics**.

The system identifies the **dominant cluster** for each query result.

---

# Cluster Analysis

The Gaussian Mixture Model was trained with **15 clusters**.

Clusters roughly correspond to semantic themes present in the dataset, such as:

* computer hardware
* graphics
* religion
* politics
* sports

Because clustering is unsupervised, clusters do not exactly match dataset labels but instead capture **latent semantic structure**.

## Boundary Documents

Some documents show membership across multiple clusters. For example:

```
Graphics cluster → 0.41
Hardware cluster → 0.36
AI cluster → 0.23
```

These **boundary documents** indicate overlapping topics such as GPU acceleration for machine learning.

Fuzzy clustering allows the model to represent this ambiguity effectively.

---

# Semantic Cache

The semantic cache stores previous query results to avoid redundant computation.

Each cache entry contains:

* original query
* query embedding
* search result
* predicted cluster

When a new query arrives:

1. The query is embedded.
2. The cache is searched for embeddings above a similarity threshold.
3. If a match is found, the cached result is returned immediately.

This significantly reduces repeated search operations.

Cache statistics tracked:

* total entries
* cache hits
* cache misses
* hit rate

---

# API Endpoints

The system exposes a REST API built with FastAPI.

Base URL:

```
http://127.0.0.1:8000
```

### Root Endpoint

GET /

Returns a basic message indicating the API is running.

---

### Query Endpoint

POST /query

Request:

```
{
 "query": "machine learning libraries"
}
```

Response:

```
{
 "query": "...",
 "cache_hit": false,
 "matched_query": null,
 "similarity_score": null,
 "result": "...",
 "dominant_cluster": 7
}
```

If a semantically similar query already exists in cache, the system returns a cached response instead.

---

### Cache Statistics

GET /cache/stats

Returns performance metrics for the semantic cache.

Example:

```
{
 "total_entries": 12,
 "hit_count": 5,
 "miss_count": 7,
 "hit_rate": 0.41
}
```

---

### Clear Cache

DELETE /cache

Removes all cached entries.

---

# Running the Project

1. Clone the repository

```
git clone <repo-url>
```

2. Create virtual environment

```
python -m venv venv
```

3. Activate environment

Mac/Linux

```
source venv/bin/activate
```

4. Install dependencies

```
pip install -r requirements.txt
```

5. Start API server

```
uvicorn api.main:app --reload
```

6. Open API documentation

```
http://127.0.0.1:8000/docs
```

---

# Key Features

* Semantic text search using transformer embeddings
* Intelligent semantic caching
* Fuzzy clustering of document embeddings
* REST API interface
* Cache performance monitoring

---

# Possible Future Improvements

* Use FAISS for faster vector search
* Implement cluster-aware caching
* Persist embeddings to disk to speed up server startup
* Add approximate nearest neighbor search

---

# Conclusion

This project demonstrates how modern NLP techniques such as transformer embeddings and semantic similarity can be integrated with caching strategies to build efficient semantic search systems.

The architecture is modular and designed to be extensible for larger datasets and production environments.
