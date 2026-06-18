import networkx as nx
import numpy as np
import pickle
from src.graphs.visualization import draw_layout
def create_graph(type, **kwargs):
    """
    Universal graph creator using keyword arguments (e.g. from yaml)
    """
    n = kwargs.get('n', 10)
    p = kwargs.get('p', 0.1)
    seed = kwargs.get('seed', 42)
    pos = None
    if type == 'star':
        G = create_star_graph(n)
    elif type == 'tree':
        leaves_per_node = kwargs.get('leaves_per_node', 2)
        tree_height = kwargs.get('tree_height', 4)
        G = nx.balanced_tree(leaves_per_node,tree_height)
    elif type == 'cycle':
        G = create_cycle_graph(n)
    elif type == 'path':
        G = create_path_graph(n)
    elif type == 'complete':
        G = create_complete_graph(n)
    elif type == 'erdos_renyi':
        G = create_erdos_renyi_graph(n, p, seed)
    elif type == 'lattice':
        m = kwargs.get('m', n)
        G = create_lattice_graph(n, m)
    elif type == 'sbm':
        sizes = kwargs.get('sizes')
        p_intra = kwargs.get('p_intra')
        p_inter = kwargs.get('p_inter', 0.01)
        G = create_SBM_graph(sizes, p_intra, p_inter)
    elif type == 'geometric':
        radius = kwargs.get('geometric_radius', 0.2)
        G, pos = create_geometric_graph(n, radius)
    elif type == 'tree_of_cycles':
        cycle_size = kwargs.get('cycle_size', 8)
        n_cycles = kwargs.get('n_cycles', 4)
        branching = kwargs.get('leaves_per_node', 3)
        tree_height = kwargs.get('tree_height', 2)
        G, _ = make_tree_of_cycles(cycle_size, n_cycles, branching, tree_height)
    elif type == 'tree_with_grid':
        tree_height = kwargs.get('tree_height', 2)
        leaves_per_node = kwargs.get('leaves_per_node', 3)
        size_grid = kwargs.get('size_grid', 4)
        G = make_tree_with_grid(tree_height, leaves_per_node, size_grid)
    else:
        raise ValueError(f"Unknown graph type: {type}")
    if pos is None:
        pos = draw_layout(G, seed=42)
    return G, pos

def create_SBM_graph(sizes, p_intra, p_inter=0.01, custom_p=None, seed=42):
    n_blocks = len(sizes)
    if custom_p is not None:
        p_matrix = custom_p
    else:
        p_matrix = np.full((n_blocks, n_blocks), p_inter)
        for i in range(n_blocks):
                p_matrix[i, i] = p_intra[i]
    return nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=seed)

def create_star_graph(n):
    G =  nx.star_graph(n-1) 
    return G 

def create_tree_graph(n):
    G = nx.random_labeled_tree(n)
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

def create_erdos_renyi_graph(n, p, seed=42):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G

def create_lattice_graph(n,m):
    G = nx.cartesian_product(nx.path_graph(n), nx.path_graph(m))
    G = nx.convert_node_labels_to_integers(G)
    return G

def create_geometric_graph(n,radius,dim=2, seed=42):
    G = nx.random_geometric_graph(n=n, radius=radius, dim=dim, seed=seed)
    pos = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}
    return G,pos 


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



def make_cycle(n, start=0):
    """C_n with nodes labeled start..start+n-1."""
    G = nx.cycle_graph(n)
    return nx.relabel_nodes(G, {i: i + start for i in range(n)})

def make_tree(branching, height, start=0):
    """Balanced tree with nodes offset by start."""
    G = nx.balanced_tree(branching, height)
    return nx.relabel_nodes(G, {i: i + start for i in G.nodes()})

def make_tree_of_cycles(cycle_size, n_cycles, branching=2, tree_height=2):
    """
    A balanced tree where each leaf has a cycle C_n attached at one node.

    Returns the composed graph and a dict with component membership:
        {'tree': [node, ...], 'cycles': [[node, ...], ...]}
    """
    tree = make_tree(branching, tree_height, start=0)
    offset = tree.number_of_nodes()

    G = tree.copy()
    leaves = [n for n in tree.nodes() if tree.degree(n) == 1]
    metadata = {'tree': list(tree.nodes()), 'cycles': []}

    for i, leaf in enumerate(leaves[:n_cycles]):
        cycle = make_cycle(cycle_size, start=offset)
        attach_node = offset  # first node of the cycle
        G = nx.compose(G, cycle)
        G.add_edge(leaf, attach_node)
        metadata['cycles'].append(list(cycle.nodes()))
        offset += cycle_size

    return G, metadata

def make_tree_with_grid(height_tree, leaves_per_node, size_grid):
    # ── 1. Grid (low hyperbolicity / high delta) ──────────────────────────────────
    grid = nx.grid_2d_graph(size_grid, size_grid)
    grid = nx.relabel_nodes(grid, {node: i for i, node in enumerate(grid.nodes())})
    n_grid = grid.number_of_nodes()  # 16 nodes → IDs 0..15

    # ── 2. Tree (high hyperbolicity / delta = 0) ──────────────────────────────────
    tree = nx.balanced_tree(r=leaves_per_node, h=height_tree)
    # Shift tree IDs to start right after grid IDs 
    tree = nx.relabel_nodes(tree, {n: n + n_grid for n in tree.nodes()})

    # ── 3. Compose and bridge ─────────────────────────────────────────────────────
    hybrid = nx.compose(grid, tree)
    bridge_grid_node = 0          # corner of the grid
    bridge_tree_node = n_grid     # root of the tree (was 0, now n_grid)
    hybrid.add_edge(bridge_grid_node, bridge_tree_node)
    return hybrid