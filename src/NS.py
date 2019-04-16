import zlib
from collections import deque

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class NS:
    def __init__(self, pca_componenets=10, queue_len=5000, k_neighbors=10):
        self.pca = PCA(n_components=pca_componenets)
        self.queue = deque(maxlen=queue_len)
        self.neighbors = NearestNeighbors(n_neighbors=k_neighbors)

    def add_observation(self, ob: np.ndarray):
        # ob_comp = zlib.compress(ob.tostring())
        self.queue.append(ob.flatten())

    def get_novelty_score(self):
        pca_rep = self.pca.fit_transform(self.queue)
        nbrs = self.neighbors.fit(pca_rep)
        distances, indices = nbrs.kneighbors(pca_rep[0])
        return sum(distances)
