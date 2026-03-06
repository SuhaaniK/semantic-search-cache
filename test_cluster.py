from clustering.fuzzy_cluster import FuzzyCluster
import numpy as np

# fake embeddings
embeddings = np.random.rand(100, 384)

cluster = FuzzyCluster(n_clusters=5)

cluster.fit(embeddings)

sample = embeddings[0].reshape(1, -1)

probs = cluster.get_membership(sample)

print("Cluster probabilities:")
print(probs)