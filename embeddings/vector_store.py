import faiss
import numpy as np


class VectorStore:

    def __init__(self, dim):

        self.index = faiss.IndexFlatIP(dim)
        self.documents = []

    def add_documents(self, embeddings, docs):

        self.index.add(np.array(embeddings))
        self.documents.extend(docs)

    def search(self, embedding, k=5):

        D, I = self.index.search(embedding, k)

        results = [self.documents[i] for i in I[0]]

        return results, D[0]