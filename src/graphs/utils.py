import networkx as nx
import numpy as np
import pickle

seed = 42

def create_graph(type, **kwargs):
    """
    Just a wrapper 
    """
    n = kwargs.get('n', 10)
    p = kwargs.get('p', 0.1)

    if type == 'star':
        G = create_star_graph(n)
    elif type == 'tree':
        G = create_tree_graph(n)
    elif type == 'cycle':
        G = create_cycle_graph(n)
    elif type == 'path':
        G = create_path_graph(n)
    elif type == 'complete':
        G = create_complete_graph(n)
    elif type == 'erdos_renyi':
        G = create_erdos_renyi_graph(n, p)
    elif type == 'lattice':
        G = create_lattice_graph(n)
    else:
        raise ValueError(f"Unknown graph type: {type}")
    return G 

def create_SBM_graph(sizes, p_intra, p_inter=0.01, custom_p=None):
    n_blocks = len(sizes)
    if custom_p is not None:
        p_matrix = custom_p
    else:
        p_matrix = np.full((n_blocks, n_blocks), p_inter)
        for i in range(n_blocks):
                p_matrix[i, i] = p_intra[i]
    return nx.stochastic_block_model(sizes, p_matrix.tolist())

def create_star_graph(n):
    G =  nx.star_graph(n-1) 
    return G 

def create_tree_graph(n):
    G = nx.random_labeled_tree(n, seed=seed)
    return G 

def create_cycle_graph(n):
    G = nx.cycle_graph(n)
    return G 

def create_path_graph(n):
    G = nx.path_graph(n)
    return G 

def create_complete_graph(n):
    G = nx.complete_graph(n)
    return G

def create_erdos_renyi_graph(n, p):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G

def create_lattice_graph(n,m):
    G = nx.cartesian_product(nx.path_graph(n), nx.path_graph(m))
    G = nx.convert_node_labels_to_integers(G)
    return G

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

def remove_edges(G, edges):
    """
    edges is a list of tuple 
    """
    for u, v in edges:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
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
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    dist = np.full((n, n), np.inf)

    for u, lengths in nx.all_pairs_shortest_path_length(G):
        i = index[u]
        for v, d in lengths.items():
            j = index[v]
            dist[i, j] = d

    return dist


def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved graph in {filename}.")

def load_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G