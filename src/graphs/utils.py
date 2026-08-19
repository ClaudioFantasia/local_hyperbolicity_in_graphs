import networkx as nx
import numpy as np
import pickle
from src.graphs.visualization import draw_layout
import random   
def create_graph(type, seed = 42, **kwargs):
    """
    Universal graph creator using keyword arguments (e.g. from yaml)
    It return the networkX graph and the pos to visualise it
    """
    n = kwargs.get('n', 10)
    p = kwargs.get('p', 0.1)
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
        G = create_SBM_graph(sizes=sizes, p_intra=p_intra, p_inter=p_inter, seed=seed)
    elif type == 'geometric':
        radius = kwargs.get('geometric_radius', 0.2)
        G, pos = create_geometric_graph(n=n, radius=radius, seed=seed)
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
    elif type == 'molecule_like':
        G = create_molecule_graph()
    elif type == 'hierarchical':
        tree_height = kwargs.get('tree_height',2)
        leaves_per_node = kwargs.get('leaves_per_node',3)
        random_edges_added = kwargs.get('random_edges_added',0)
        G = create_hierarchical_graph(k=leaves_per_node,depth=tree_height,shortcuts=random_edges_added)
    else:
        raise ValueError(f"Unknown graph type: {type}")
    if pos is None:
        pos = draw_layout(G, seed=42)
    return G, pos

def create_hierarchical_graph(k: int = 3, depth: int = 3, shortcuts: int = 6,
                 seed: int = 42) -> nx.Graph:
    """
    Albero k-ario di profondità d con shortcut casuali tra foglie.

    L'albero è perfettamente iperbolico (δ = 0), gli shortcut aumentano
    δ in modo controllato. Permette di studiare come i cicli degradano
    l'iperbolicità. Alzare 'shortcuts' da 0 gradualmente per calibrare.

    Parametri:
        k        : grado (numero di figli per nodo interno)
        depth    : profondità dell'albero
        shortcuts: archi casuali aggiunti tra foglie
    """
    rng = random.Random(seed)
    G = nx.Graph()
    idx = [0]
    leaves = []

    def build(parent, d):
        if d == 0:
            leaves.append(parent)
            return
        for _ in range(k):
            child = idx[0] + 1
            idx[0] = child
            G.add_node(child)
            G.add_edge(parent, child)
            build(child, d - 1)

    G.add_node(0)
    build(0, depth)

    for _ in range(shortcuts):
        a = rng.choice(leaves)
        b = rng.choice(leaves)
        if a != b:
            G.add_edge(a, b)

    return G


def create_molecule_graph():
    # 1. Create the base components
    tree_part = nx.balanced_tree(r=2, h=3)  # 15 nodes
    main_path = nx.path_graph(16)           # Long backbone path (16 nodes)
    cycle_part = nx.cycle_graph(8)          # 8 nodes
    grid_part = nx.grid_2d_graph(4, 4)      # 16 nodes
    
    # 2. Join all of them disjointly to ensure sequential scalar IDs
    G = nx.disjoint_union_all([tree_part, main_path, cycle_part, grid_part])
    
    # 3. Calculate starting indices for each component in the merged graph
    tree_start = 0
    path_start = len(tree_part)
    cycle_start = path_start + len(main_path)
    grid_start = cycle_start + len(cycle_part)
    
    # 4. Wire the "Molecule" structure together
    
    # A. Connect Tree to the beginning of the Main Path
    G.add_edge(tree_start, path_start)
    
    # B. Connect Grid to the end of the Main Path
    path_end = path_start + len(main_path) - 1
    G.add_edge(path_end, grid_start)
    
    # C. Branch the Cycle off the center of the Main Path
    path_middle = path_start + (len(main_path) // 2)
    G.add_edge(path_middle, cycle_start)
    
    return G




def create_SBM_graph(sizes, p_intra, p_inter=0.01, custom_p=None, seed = 42):
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


def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved graph in {filename}.")

def load_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G
