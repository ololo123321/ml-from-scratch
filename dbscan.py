from collections import defaultdict


class DBSCAN():
    def __init__(self, eps, min_samples, metric):
        self.eps = eps
        self.min_samples = min_samples    
        self.metric = metric
    
    def _region_query(self, p):
        return {q_idx for q_idx, q in enumerate(self.X) if self.metric(p, q) < self.eps}
    
    def _expand_cluster(self, C, neighbours):
        while neighbours:
            q_idx = neighbours.pop()
            q = self.X[q_idx]
            if q_idx not in self.visited_points:
                self.visited_points.add(q_idx)
                q_neighbours = self._region_query(q)
                if len(q_neighbours) >= self.min_samples:
                    neighbours |= q_neighbours
            if q_idx not in self.clustered_points:
                self.clusters[C].add(q_idx)
                self.clustered_points.add(q_idx)
                if q_idx in self.clusters[self.NOISE]:
                    self.clusters[self.NOISE].remove(q_idx)
    
    def _get_labels(self):
        labels = []
        for p_idx, p in enumerate(self.X):
            flag=True
            for cluster, points in self.clusters.items():
                if flag and p_idx in points:
                    labels.append(cluster)
                    flag=False
        return labels
        
    def fit_predict(self, X):
        self.X = X
        self.clusters = defaultdict(set)
        self.visited_points = set()
        self.clustered_points = set()
        self.NOISE = -1
        C = 0
        for p_idx, p in enumerate(self.X):
            if p_idx in self.visited_points:
                continue
            self.visited_points.add(p_idx)
            neighbours = self._region_query(p)
            if len(neighbours) < self.min_samples:
                self.clusters[self.NOISE].add(p_idx)
            else:
                self.clusters[C].add(p_idx)
                self.clustered_points.add(p_idx)
                self._expand_cluster(C, neighbours)
                C += 1
        labels = self._get_labels()
        return labels