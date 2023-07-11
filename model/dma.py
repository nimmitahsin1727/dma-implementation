import numpy as np

class DMA:
    def __init__(self, K, alpha, beta):
        self.K = K  # number of topics
        self.alpha = alpha  # hyperparameter for the document-topic distribution
        self.beta = beta  # hyperparameter for the topic-word distribution

    def fit(self, X, max_iter=100):
        N, V = X.shape
        z = np.zeros((N, V), dtype=int)  # topic assignments for each word in each document
        n_z = np.zeros(self.K, dtype=int)  # number of words assigned to each topic
        n_zw = np.zeros((self.K, V), dtype=int)  # number of occurrences of each word in each topic
        n_zd = np.zeros((self.K, N), dtype=int)  # number of words assigned to each topic in each document
        n_z_sum = np.zeros(self.K, dtype=int)  # total number of words assigned to each topic

        # Initialize topic assignments randomly
        for i in range(N):
            for j in range(V):
                if X[i, j] > 0:
                    z[i, j] = np.random.choice(self.K)
                    n_z[z[i, j]] += X[i, j]
                    n_zw[z[i, j], j] += X[i, j]
                    n_zd[z[i, j], i] += X[i, j]
                    n_z_sum[z[i, j]] += X[i, j]

        # Iterate over the data and update topic assignments
        for _ in range(max_iter):
            for i in range(N):
                for j in range(V):
                    if X[i, j] > 0:
                        # Remove the current word from the topic assignment
                        n_z[z[i, j]] -= X[i, j]
                        n_zw[z[i, j], j] -= X[i, j]
                        n_zd[z[i, j], i] -= X[i, j]
                        n_z_sum[z[i, j]] -= X[i, j]

                        # Compute the posterior probabilities of the topics
                        p_z = (n_zw[:, j] + self.beta) / (n_z_sum + self.beta * V) * (n_zd[:, i] + self.alpha)
                        p_z /= p_z.sum()

                        # Sample a new topic assignment from the posterior probabilities
                        z[i, j] = np.random.choice(self.K, p=p_z)

                        # Add the current word to the new topic assignment
                        n_z[z[i, j]] += X[i, j]
                        n_zw[z[i, j], j] += X[i, j]
                        n_zd[z[i, j], i] += X[i, j]
                        n_z_sum[z[i, j]] += X[i, j]

        # Compute the topic-word distribution
        self.phi = (n_zw + self.beta) / (n_z_sum[:, np.newaxis] + self.beta * V)

    def predict(self, X):
        N, V = X.shape
        p_z = np.zeros((N, self.K))

        # Compute the posterior probabilities of the topics for each document
        for i in range(N):
            for j in range(V):
                if X[i, j] > 0:
                    p_z[i] += X[i, j] * np.log(self.phi[:, j])

        # Add the prior on the document-topic distribution
        p_z += np.log(self.alpha)

        # Normalize the probabilities for each document
        p_z = np.exp(p_z - p_z.max(axis=1, keepdims=True))
        p_z /= p_z.sum(axis=1, keepdims=True)

        return p_z
