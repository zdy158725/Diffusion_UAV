import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.signal as ss
from collections import defaultdict
import warnings


class TemporalSceneGraph(object):
    def __init__(self,
                 edge_radius,
                 nodes=None,
                 adj_cube=np.zeros((1, 0, 0)),
                 weight_cube=np.zeros((1, 0, 0)),
                 node_type_mat=np.zeros((0, 0)),
                 edge_scaling=None):
        self.edge_radius = edge_radius
        self.nodes = nodes
        if nodes is None:
            self.nodes = np.array([])
        self.adj_cube = adj_cube
        self.weight_cube = weight_cube
        self.node_type_mat = node_type_mat
        self.adj_mat = np.max(self.adj_cube, axis=0).clip(max=1.0)
        self.edge_scaling = edge_scaling
        self.node_index_lookup = None
        self.calculate_node_index_lookup()

    def calculate_node_index_lookup(self):
        node_index_lookup = dict()
        for i, node in enumerate(self.nodes):
            node_index_lookup[node] = i

        self.node_index_lookup = node_index_lookup

    def get_num_edges(self, t=0):
        return np.sum(self.adj_cube[t]) // 2

    def get_index(self, node):
        return self.node_index_lookup[node]

    @staticmethod
    def get_edge_type(n1, n2):
        return '-'.join(sorted([str(n1), str(n2)]))

    @classmethod
    def create_from_temp_scene_dict(cls,
                                    scene_temp_dict,
                                    attention_radius,
                                    duration=1,
                                    graph_mode='radius',
                                    graph_top_k=None,
                                    edge_addition_filter=None,
                                    edge_removal_filter=None):
        """
        Construct a spatiotemporal graph from agent positions in a dataset.

        returns: sg: An aggregate SceneGraph of the dataset.
        """
        nodes = scene_temp_dict.keys()
        N = len(nodes)
        total_timesteps = duration

        position_cube = np.zeros((total_timesteps, N, 3))

        adj_cube = np.zeros((total_timesteps, N, N), dtype=np.int8)
        dist_cube = np.zeros((total_timesteps, N, N), dtype=float)

        node_type_mat = np.zeros((N, N), dtype=np.int8)
        node_attention_mat = np.zeros((N, N), dtype=float)

        for node_idx, node in enumerate(nodes):
            position_cube[:, node_idx] = scene_temp_dict[node]
            node_type_mat[:, node_idx] = node.type.value
            for node_idx_from, node_from in enumerate(nodes):
                node_attention_mat[node_idx_from, node_idx] = attention_radius[(node_from.type, node.type)]

        np.fill_diagonal(node_type_mat, 0)
        agg_adj_matrix = np.zeros((N, N), dtype=np.int8)

        for timestep in range(position_cube.shape[0]):
            dists = squareform(pdist(position_cube[timestep], metric='euclidean'))
            adj_matrix = cls.build_adj_matrix(dists,
                                              node_attention_mat,
                                              node_type_mat,
                                              graph_mode=graph_mode,
                                              graph_top_k=graph_top_k)

            agg_adj_matrix |= adj_matrix

            adj_cube[timestep] = adj_matrix
            dist_cube[timestep] = dists

        dist_cube[np.isnan(dist_cube)] = 0.
        weight_cube = np.divide(1.,
                                dist_cube,
                                out=np.zeros_like(dist_cube),
                                where=(dist_cube > 0.))
        edge_scaling = None
        if edge_addition_filter is not None and edge_removal_filter is not None:
            edge_scaling = cls.calculate_edge_scaling(adj_cube, edge_addition_filter, edge_removal_filter)
        sg = cls(attention_radius, np.array(list(nodes)), adj_cube, weight_cube, node_type_mat, edge_scaling=edge_scaling)
        return sg

    @staticmethod
    def build_radius_adj_matrix(dists, node_attention_mat, node_type_mat):
        # Can produce a warning as dists can be nan if no data for node is available.
        # This is accepted as nan <= x evaluates to False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj_matrix = (dists <= node_attention_mat).astype(np.int8) * node_type_mat

        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    @staticmethod
    def build_knn_adj_matrix(dists, node_type_mat, graph_top_k):
        if graph_top_k is None or graph_top_k <= 0:
            return np.zeros_like(node_type_mat, dtype=np.int8)

        adj_matrix = np.zeros_like(node_type_mat, dtype=np.int8)
        valid_pair_mask = (node_type_mat > 0) & np.isfinite(dists)
        np.fill_diagonal(valid_pair_mask, False)

        for node_idx in range(dists.shape[0]):
            candidate_indices = np.flatnonzero(valid_pair_mask[node_idx])
            if candidate_indices.size == 0:
                continue

            candidate_dists = dists[node_idx, candidate_indices]
            nearest_order = np.argsort(candidate_dists)
            nearest_indices = candidate_indices[nearest_order[:min(graph_top_k, candidate_indices.size)]]
            adj_matrix[node_idx, nearest_indices] = 1

        adj_matrix = np.maximum(adj_matrix, adj_matrix.T).astype(np.int8)
        return adj_matrix * node_type_mat

    @classmethod
    def build_adj_matrix(cls, dists, node_attention_mat, node_type_mat, graph_mode='radius', graph_top_k=None):
        if graph_mode == 'radius':
            return cls.build_radius_adj_matrix(dists, node_attention_mat, node_type_mat)
        if graph_mode == 'knn':
            return cls.build_knn_adj_matrix(dists, node_type_mat, graph_top_k)
        if graph_mode == 'radius_knn':
            radius_adj = cls.build_radius_adj_matrix(dists, node_attention_mat, node_type_mat)
            knn_adj = cls.build_knn_adj_matrix(dists, node_type_mat, graph_top_k)
            return np.maximum(radius_adj, knn_adj).astype(np.int8)
        raise ValueError(f'Unsupported graph_mode: {graph_mode}')

    @staticmethod
    def calculate_edge_scaling(adj_cube, edge_addition_filter, edge_removal_filter):
        new_edges = np.minimum(
            ss.convolve(adj_cube, np.reshape(edge_addition_filter, (-1, 1, 1)), 'same'), 1.
        )

        old_edges = np.minimum(
            ss.convolve(adj_cube, np.reshape(edge_removal_filter, (-1, 1, 1)), 'same'), 1.
        )

        return np.minimum(new_edges + old_edges, 1.)

    def to_scene_graph(self, t, t_hist=0, t_fut=0):
        lower_t = np.clip(t-t_hist, a_min=0, a_max=None)
        higher_t = np.clip(t + t_fut + 1, a_min=None, a_max=self.adj_cube.shape[0] + 1)
        adj_mat = np.max(self.adj_cube[lower_t:higher_t], axis=0)
        weight_mat = np.max(self.weight_cube[lower_t:higher_t], axis=0)
        return SceneGraph(self.edge_radius,
                          self.nodes,
                          adj_mat,
                          weight_mat,
                          self.node_type_mat,
                          self.node_index_lookup,
                          edge_scaling=self.edge_scaling[t])


class SceneGraph(object):
    def __init__(self,
                 edge_radius,
                 nodes=None,
                 adj_mat=np.zeros((0, 0)),
                 weight_mat=np.zeros((0, 0)),
                 node_type_mat=np.zeros((0, 0)),
                 node_index_lookup=None,
                 edge_scaling=None):
        self.edge_radius = edge_radius
        self.nodes = nodes
        if nodes is None:
            self.nodes = np.array([])
        self.node_type_mat = node_type_mat
        self.adj_mat = adj_mat
        self.weight_mat = weight_mat
        self.edge_scaling = edge_scaling
        self.node_index_lookup = node_index_lookup

    def get_index(self, node):
        return self.node_index_lookup[node]

    def get_neighbors(self, node, type):
        node_index = self.get_index(node)
        connection_mask = self.adj_mat[node_index].astype(bool)
        mask = ((self.node_type_mat[node_index] == type.value) * connection_mask)
        return self.nodes[mask]

    def get_edge_scaling(self, node=None):
        if node is None:
            return self.edge_scaling
        else:
            node_index = self.get_index(node)
            return self.edge_scaling[node_index, self.adj_mat[node_index] > 0.]

    def get_edge_weight(self, node=None):
        if node is None:
            return self.weight_mat
        else:
            node_index = self.get_index(node)
            return self.weight_mat[node_index, self.adj_mat[node_index] > 0.]
