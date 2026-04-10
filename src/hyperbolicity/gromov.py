import itertools
import numpy as np
from ..graphs.utils import compute_distance_nodes
import random
from ..graphs.visualization import draw_quadruples
import networkx as nx

random.seed(42)
np.random.seed(42)
def compute_gromov_on_graph(G,return_history=False, return_mean=False):
    """
    Compute Gromov hyperbolicity from a graph.
    """
    dist_matrix = compute_distance_nodes(G)
    delta_max, delta_mean, quadruples = compute_gromov_hyperbolicity(dist_matrix, return_history=True)

    if return_history:
        return delta_max, delta_mean, quadruples
    else:
        if return_mean:
            return delta_max, delta_mean
        return delta_max

def compute_gromov_hyperbolicity(dist_matrix, return_history=False):
    n = dist_matrix.shape[0]

    max_delta = -np.inf
    sum_delta = 0.0
    count = 0
    max_quadruples = []

    for i, j, k, l in itertools.combinations(range(n), 4):
        d01, d23 = dist_matrix[i, j], dist_matrix[k, l]
        d02, d13 = dist_matrix[i, k], dist_matrix[j, l]
        d03, d12 = dist_matrix[i, l], dist_matrix[j, k]

        s = [d01 + d23, d02 + d13, d03 + d12]
        s.sort(reverse=True) 
        delta = (s[0] - s[1]) / 2.0
        
        sum_delta += delta
        count += 1

        if delta > max_delta:
            max_delta = delta
            if return_history:
                max_quadruples = [(i, j, k, l)]
        elif delta == max_delta and return_history:
            max_quadruples.append((i, j, k, l))

    mean_delta = sum_delta / count if count > 0 else 0.0

    if return_history:
        return max_delta, mean_delta, max_quadruples
    return max_delta, mean_delta





def compute_gromov_hyperbolicity_not_optimized(dist_matrix):
    """
    Compute Gromov hyperbolicity from a distance matrix.
    """
    n = dist_matrix.shape[0]
    deltas = []
    

    for i, j, k, l in itertools.permutations(range(n), 4):
        d01 = dist_matrix[i, j]
        d23 = dist_matrix[k, l]
        d02 = dist_matrix[i, k]
        d13 = dist_matrix[j, l]
        d03 = dist_matrix[i, l]
        d12 = dist_matrix[j, k]

        s = [d01 + d23, d02 + d13, d03 + d12]
        
        
        delta = 0.5 * (s[0] - max(s[1], s[2]))

        deltas.append(delta)

    return np.max(deltas), np.mean(deltas)

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
        if len(fall_quadruples) < 40:
            draw_quadruples(G, pos ,fall_quadruples)
        else:
            print("too many quadruples to visualize")
        
    print(f"We found {len(found_edges)} optimal edges.")
    return found_edges