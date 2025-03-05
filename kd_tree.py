import numpy as np
import heapq
from numba import jit


class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right
        self.idx = None


class CustomKDTree:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)
        self.n_dims = points.shape[1]
        self._indices = np.arange(len(points))
        self.root = self._build_tree(self._indices, 0)

    def _build_tree(self, point_indices, depth):
        if len(point_indices) == 0:
            return None

        axis = depth % self.n_dims

        sorted_idx = point_indices[np.argsort(self.points[point_indices, axis])]
        median_idx = len(sorted_idx) // 2

        node = KDNode(self.points[sorted_idx[median_idx]])
        node.idx = sorted_idx[median_idx]

        node.left = self._build_tree(sorted_idx[:median_idx], depth + 1)
        node.right = self._build_tree(sorted_idx[median_idx + 1:], depth + 1)

        return node

    @staticmethod
    @jit(nopython=True)
    def _distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def query_ball_point(self, center_point, radius):
        center_point = np.asarray(center_point, dtype=np.float32)
        result_indices = []
        self._query_ball_point_recursive(self.root, center_point, radius, 0, result_indices)
        return result_indices

    def _query_ball_point_recursive(self, node, center_point, radius, depth, result_indices):
        if node is None:
            return

        dist = self._distance(center_point, node.point)

        if dist <= radius:
            result_indices.append(node.idx)

        axis = depth % self.n_dims
        diff = center_point[axis] - node.point[axis]

        if diff <= radius:
            self._query_ball_point_recursive(node.left, center_point, radius, depth + 1, result_indices)

        if diff >= -radius:
            self._query_ball_point_recursive(node.right, center_point, radius, depth + 1, result_indices)

    def query(self, points, k=1):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        n_points = len(points)
        distances = np.zeros((n_points, k), dtype=np.float32)
        indices = np.zeros((n_points, k), dtype=np.int32)

        for i in range(n_points):
            heap = []
            self._query_recursive(self.root, points[i], k, heap, 0)

            heap.sort(reverse=True)

            for j in range(min(k, len(heap))):
                distances[i, j] = -heap[j][0]
                indices[i, j] = heap[j][1]

        return distances, indices

    def _query_recursive(self, node, point, k, heap, depth):
        if node is None:
            return

        dist = self._distance(point, node.point)

        if len(heap) < k:
            heapq.heappush(heap, (-dist, node.idx))
        elif -dist > heap[0][0]:
            heapq.heapreplace(heap, (-dist, node.idx))

        axis = depth % self.n_dims
        diff = point[axis] - node.point[axis]

        radius = -heap[0][0] if heap else float('inf')

        if diff <= 0:
            self._query_recursive(node.left, point, k, heap, depth + 1)
            if abs(diff) < radius:
                self._query_recursive(node.right, point, k, heap, depth + 1)
        else:
            self._query_recursive(node.right, point, k, heap, depth + 1)
            if abs(diff) < radius:
                self._query_recursive(node.left, point, k, heap, depth + 1)