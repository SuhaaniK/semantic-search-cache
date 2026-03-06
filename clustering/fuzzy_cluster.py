from sklearn.mixture import GaussianMixture


class FuzzyCluster:

    def __init__(self, n_clusters=15):

        self.model = GaussianMixture(
            n_components=n_clusters
        )

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def get_membership(self, embedding):

        probs = self.model.predict_proba(embedding)

        return probs