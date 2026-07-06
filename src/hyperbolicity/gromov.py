import itertools
import numpy as np
import random
from ..graphs.visualization import draw_quadruples
import networkx as nx

random.seed(42)
np.random.seed(42)

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



def compute_gromov_hyperbolicity(G):
    """
    Compute Gromov hyperbolicity from a graph.

    Return:
    max_delta: scalar -> maximum of deltas (i.e. Gromov hyperbolicity)
    deltas: dict -> dictionary of quads and their deltas
    """   
    dist_matrix = compute_distance_nodes(G)
    nodes = list(G.nodes())
    deltas = {}

    for x, y, z, w in itertools.combinations(nodes,4):
        quad = (x,y,z,w)
        deltas[quad] = compute_delta_gromov(dist_matrix, quad)
    
    max_delta = max(deltas.values())
    return max_delta, deltas


def compute_intra_distance(dists,quad):
    return np.mean([dists[quad[i]][quad[j]] for i in range(4) for j in range(i+1, 4)])


#############
#################################
#############

def best_edge_for_gromov_optimization(G, current_gromov,candidate_to_add, candidate_to_remove, target):
    """
    This function tries to find an edge to add or remove that will change the Gromov hyperbolicity in the desired direction (increase or decrease).
    It returns a list of edges that achieve this goal.
    """
    found_edges = []
    # Controllo aggiunte
    for u, v in candidate_to_add:
        G.add_edge(u, v)
        new_val = compute_gromov_on_graph(G)
        if (target == 'increase' and new_val > current_gromov) or \
           (target == 'decrease' and new_val < current_gromov):
            found_edges.append(('add', (u, v), new_val))
        G.remove_edge(u, v)

    # Controllo rimozioni
    for u, v in candidate_to_remove:
        G.remove_edge(u, v)
        new_val = compute_gromov_on_graph(G)
        if (target == 'increase' and new_val > current_gromov) or \
           (target == 'decrease' and new_val < current_gromov):
            found_edges.append(('remove', (u, v), new_val))
        G.add_edge(u, v)
    return found_edges

def evolve_topology_strategy(G, pos, target='increase', strategy='mixed', p=0.5, max_steps = 50):
    """
    The algorithm tries to find an edge to add or remove that will change the Gromov hyperbolicity in the desired direction (increase or decrease).
    If it finds such an edge, it returns it. If not, it applies a fallback strategy, to add or remove a random edge and check 
    again the Gromov hyperbolicity.
    The fallback strategy can be to only add or to only remove random edges. Or a mixture of the two with a probability of adding (p) or removing (1-p).
    """
    result_dict = []
    nodes = list(G.nodes)
    candidate_to_add = [
        (u, v) for u, v in itertools.combinations(nodes, 2) if not G.has_edge(u, v)
    ]
    candidate_to_remove = list(G.edges)
    found_edges = []
    current_gromov = compute_gromov_on_graph(G)
    count = 0
    while (not found_edges):            
        count += 1
        found_edges = best_edge_for_gromov_optimization(G, current_gromov, candidate_to_add, candidate_to_remove, target=target)   
        if found_edges or count > max_steps or not candidate_to_add or not candidate_to_remove:
            break

        ## If we do not find anything we start the fallback strategy
        action = None
        if strategy == 'add':
            action = 'add'
        elif strategy == 'remove':
            action = 'remove'
        elif strategy == 'mixed':
            action = 'add' if random.random() < p else 'remove'

        if action == 'add' and candidate_to_add:
            edge = random.choice(candidate_to_add)
            G.add_edge(*edge)
            print(f"Added random edge: {edge}")
            result_dict.append(('add_random', edge, current_gromov))
            candidate_to_add.remove(edge)
        
        elif action == 'remove' and candidate_to_remove:
            edge = random.choice(candidate_to_remove)
            G.remove_edge(*edge)
            print(f"Removed random edge: {edge}")
            result_dict.append(('remove_random', edge, current_gromov))
            candidate_to_remove.remove(edge)
        fallback_gromov, _, fall_quadruples = compute_gromov_on_graph(G,return_history=True)
        print(f"Current Gromov after fallback: {fallback_gromov}")
        print(f"Number of quadruples after fallback: {len(fall_quadruples)}")
        #draw_quadruples(G, pos ,fall_quadruples)
        
    print(f"We found {len(found_edges)} optimal edges.")
    return found_edges