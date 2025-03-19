import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.spatial.distance import cdist

class RandomProjectionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X):
        self.tree = self._build_tree(X, depth=0)

    def _build_tree(self, X, depth):
        if depth >= self.max_depth or X.shape[0] <= self.min_samples_split:
            return {'leaf': True, 'data': X}
        
        # Random projection
        random_direction = np.random.randn(X.shape[1])  # Dense direction vector
        projections = X.dot(random_direction)  # Sparse-safe dot product
        
        median = np.median(projections)
        
        left_idx = projections < median
        right_idx = projections >= median
        
        return {
            'leaf': False,
            'direction': random_direction,
            'median': median,
            'left': self._build_tree(X[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], depth + 1)
        }

    def query(self, point, tree=None):
        if tree is None:
            tree = self.tree
        
        if tree['leaf']:
            return tree['data']
        
        if issparse(point):
            projection = point.dot(tree['direction'])  # Sparse-safe dot product
        else:
            projection = np.dot(point, tree['direction'])
        
        if projection < tree['median']:
            return self.query(point, tree['left'])
        else:
            return self.query(point, tree['right'])

class RandomProjectionForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X):
        for _ in range(self.n_trees):
            tree = RandomProjectionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X)
            self.trees.append(tree)

    def query(self, point, k=1):
        all_neighbors = []
        for tree in self.trees:
            neighbors = tree.query(point)
            all_neighbors.extend(neighbors)
        
        # Convert neighbors to a dense array if sparse
        if issparse(all_neighbors[0]):
            all_neighbors = np.array([neighbor.toarray().flatten() for neighbor in all_neighbors])
        else:
            all_neighbors = np.array(all_neighbors)
        
        # Compute distances and find the k nearest neighbors
        if issparse(point):
            point = point.toarray().flatten()
        distances = cdist([point], all_neighbors, metric='euclidean').flatten()
        nearest_indices = np.argsort(distances)[:k]
        nearest_neighbors = [all_neighbors[i] for i in nearest_indices]
        
        return nearest_neighbors

