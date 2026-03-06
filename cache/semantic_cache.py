import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.entries = []
        self.hit_count = 0
        self.miss_count = 0
        self.threshold = threshold

    def lookup(self, embedding):

        for entry in self.entries:

            sim = cosine_similarity(
                [embedding],
                [entry["embedding"]]
            )[0][0]

            if sim > self.threshold:

                self.hit_count += 1
                return True, entry, sim

        self.miss_count += 1
        return False, None, None

    def add(self, query, embedding, result, cluster):

        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = len(self.entries)

        hits = self.hit_count
        misses = self.miss_count

        return {
            "total_entries": total,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": hits/(hits+misses+1e-5)
        }

    def clear(self):

        self.entries = []
        self.hit_count = 0
        self.miss_count = 0