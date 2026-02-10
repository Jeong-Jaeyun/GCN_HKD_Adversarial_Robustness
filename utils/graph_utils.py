
import torch
import numpy as np
import networkx as nx
from typing import Tuple, Optional


def convert_networkx_to_tensor(
    graph: nx.DiGraph,
    max_nodes: int = 10000,
    node_feature_dim: int = 768,
    edge_feature_dim: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    nodes = list(graph.nodes())
    if len(nodes) > max_nodes:
        import random
        nodes = random.sample(nodes, max_nodes)
        subgraph = graph.subgraph(nodes)
    else:
        subgraph = graph


    num_nodes = len(subgraph.nodes())
    node_features = np.random.randn(num_nodes, node_feature_dim).astype(np.float32)


    node_mapping = {node: idx for idx, node in enumerate(subgraph.nodes())}

    for idx, (node, node_data) in enumerate(subgraph.nodes(data=True)):
        if 'features' in node_data and node_data['features'] is not None:
            features = node_data['features']
            if isinstance(features, np.ndarray):
                dim = min(node_feature_dim, len(features))
                node_features[idx, :dim] = features[:dim]


    edges = list(subgraph.edges(data=True))
    edge_index = np.zeros((2, len(edges)), dtype=np.int64)

    for edge_idx, (src, tgt, _) in enumerate(edges):
        edge_index[0, edge_idx] = node_mapping[src]
        edge_index[1, edge_idx] = node_mapping[tgt]


    edge_features = np.ones((len(edges), edge_feature_dim), dtype=np.float32) * 0.1

    for edge_idx, (_, _, edge_data) in enumerate(edges):
        if 'weight' in edge_data:
            edge_features[edge_idx, 0] = float(edge_data['weight'])

        if 'type' in edge_data:

            edge_type = edge_data['type']
            edge_type_map = {
                'control_flow': 1,
                'data_flow': 2,
                'sequential': 3,
                'conditional': 4,
                'call_graph': 5
            }
            type_idx = edge_type_map.get(edge_type, 0)
            if type_idx > 0:
                edge_features[edge_idx, min(type_idx, edge_feature_dim - 1)] = 1.0

    return node_features, edge_index, edge_features


def normalize_graph(
    graph: nx.DiGraph,
    norm_type: str = 'degree'
) -> nx.DiGraph:
    normalized_graph = graph.copy()

    if norm_type == 'degree':

        max_degree = max(dict(graph.degree()).values()) if len(graph) > 0 else 1

        for node in normalized_graph.nodes():
            degree = normalized_graph.degree(node)
            normalized_graph.nodes[node]['degree_norm'] = degree / max_degree

    elif norm_type == 'closeness':

        closeness = nx.closeness_centrality(graph)
        if closeness:
            max_closeness = max(closeness.values())
            for node in normalized_graph.nodes():
                normalized_graph.nodes[node]['closeness_norm'] = closeness[node] / max_closeness

    elif norm_type == 'betweenness':

        betweenness = nx.betweenness_centrality(graph)
        if betweenness:
            max_betweenness = max(betweenness.values())
            for node in normalized_graph.nodes():
                normalized_graph.nodes[node]['betweenness_norm'] = betweenness[node] / max_betweenness

    return normalized_graph


def get_node_importance(
    graph: nx.DiGraph,
    importance_type: str = 'pagerank'
) -> dict:
    if importance_type == 'pagerank':
        return nx.pagerank(graph)
    elif importance_type == 'betweenness':
        return nx.betweenness_centrality(graph)
    elif importance_type == 'degree':
        degree_dict = dict(graph.degree())
        max_degree = max(degree_dict.values()) if degree_dict else 1
        return {node: degree / max_degree for node, degree in degree_dict.items()}
    else:
        raise ValueError(f"Unknown importance type: {importance_type}")


def compute_graph_statistics(graph: nx.DiGraph) -> dict:
    return {
        'num_nodes': len(graph.nodes()),
        'num_edges': len(graph.edges()),
        'density': nx.density(graph),
        'is_directed': graph.is_directed(),
        'num_strongly_connected_components': nx.number_strongly_connected_components(graph),
        'average_degree': sum(dict(graph.degree()).values()) / len(graph) if len(graph) > 0 else 0,
    }
