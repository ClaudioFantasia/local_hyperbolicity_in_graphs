import networkx as nx
import numpy as np

def create_graph(type, **kwargs):
    """
    Just a wrapper 
    """
    n = kwargs.get('n', 10)

    if type == 'star':
        G = create_star_graph(n)
    if type == 'tree':
        G = create_tree_graph(n)
    if type == 'cycle':
        G = create_cycle_graph(n)
    if type == 'path':
        G = create_path_graph(n)
    return G 
    
def create_star_graph(n):
    G =  nx.star_graph(n) 
    return G 

def create_tree_graph(n):
    G = nx.random_labeled_tree(n, seed=42)
    return G 

def create_cycle_graph(n):
    G = nx.cycle_graph(n)
    return G 

def create_path_graph(n):
    G = nx.path_graph(n)
    return G 

###

def add_nodes(G, nodes):
    G.add_nodes_from(nodes)
    return G

def add_edges(G, edges):
    """
    edges is a list of tuple 
    """
    for u, v in edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    return G

def add_n_arbitrary_nodes(G, n):
    """
    It add n nodes without any specific labels. It return a graph.
    Just a function for faster experiments. 
    """
    len_graph = len(G)
    nodes = list(range(len_graph,len_graph+n))
    G = add_nodes(G,nodes)
    return G 

def compute_distance_nodes(G):
    """
    Compute the distances between nodes using shortest path as metric
    """
    nodes = list(G.nodes())
    n = len(nodes)
    dist = np.zeros((n, n))

    lengths = dict(nx.all_pairs_shortest_path_length(G))

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            dist[i, j] = lengths[u][v]
    return dist 
    