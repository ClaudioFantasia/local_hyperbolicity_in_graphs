import os
import sys
import time
import itertools
import math
import argparse

import networkx as nx
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.utils import compute_distance_nodes, create_graph
from src.graphs.visualization import draw_graph_with_values, plot_hist
from src.utils.config import load_optimization_config, load_graph_config
from src.hyperbolicity.local import _get_quads_and_energies, score_KL_divergence, score_entropic, score_softing
from src.optimization.objectives import gromov_energy, normalize
from src.optimization.solver import solve_KL_regularization, solve_entropic_regularization

def compute_local_mu(target, dist_matrix, k, temperature, geometric_temperature, lambda_reg, sigma, method):
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return quads, np.array([])
    
    e_gromov = gromov_energy(quads, dist_matrix)
    if method == "KL_divergence":
        w_local = np.max(dist_matrix[np.array(quads), target], axis=1)
        w_local = softmax(-1 * w_local / geometric_temperature)
        mu = solve_KL_regularization(e_gromov, w_local, temperature)
    elif method == "entropic":
        e_gromov_norm = e_gromov
        w_local = np.max(dist_matrix[np.array(quads), target], axis=1)
        cost = e_gromov_norm - lambda_reg * w_local
        mu = solve_entropic_regularization(cost, temperature)
    else:
        raise ValueError(f"Method {method} does not support spatial sparsity analysis.")

    return quads, mu


def compute_node_contributions(quads, mu, n_nodes):
    contributions = np.zeros(n_nodes)
    for i, quad in enumerate(quads):
        for node in quad:
            contributions[node] += mu[i]
    return contributions


def compute_local_scores(nodes, dist_matrix, quad_cache, k, temperature, geometric_temperature, lambda_reg, sigma, method_name):
    if method_name == "KL_divergence":
        return np.array([
            score_KL_divergence(n, dist_matrix, quad_cache, k, temperature, geometric_temperature)
            for n in tqdm(nodes, desc="Computing local scores (KL_divergence)")
        ])
    elif method_name == "entropic":
        return np.array([
            score_entropic(n, dist_matrix, quad_cache, k, temperature, lambda_reg)
            for n in tqdm(nodes, desc="Computing local scores (entropic)")
        ])
    elif method_name == "softing":
        return np.array([
            score_softing(n, dist_matrix, quad_cache, k, sigma)
            for n in tqdm(nodes, desc="Computing local scores (softing)")
        ])
    return np.zeros(len(nodes))

def visualize_results(G, pos, node_contributions, mu, scores, target, k, temperature, save_dir):
    #draw_graph_with_values(G, pos, node_contributions,
    #                       title=f"Mu Heatmap (target={target})",
    #                       save_path=os.path.join(save_dir, f"{target}_{k}_contributions.png"))

    #plot_hist(mu,
    #          title=f"Mu distribution (mean={mu.mean():.8f}, var={mu.var():.8f})",
    #          bins=20,
    #          save_path=os.path.join(save_dir, f"{target}_{k}_mu.png"))

    draw_graph_with_values(G, pos, scores,
                           title="Local Hyperbolicity Heatmap",
                           save_path=os.path.join(save_dir, f"{k}_hop_T{temperature}.png"))

    #plot_hist(scores,
    #          title=f"Local Hyperbolicity (mean={scores.mean():.8f}, var={scores.var():.8f})",
    #          bins=20)


def main():
    parser = argparse.ArgumentParser(description="Run Local Hyperbolicity Experiments")
    parser.add_argument("--method", type=str, required=True, 
                        choices=["KL_divergence", "entropic","softing"],
                        help="The aggregation method to use.")
    args = parser.parse_args()
    method_name = args.method

    temperature, geometric_temperature, lambda_reg, sigma, k, target = load_optimization_config()

    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
    graph_type = graph_cfg['type']
    G.remove_edge(1,2)
    print(f"Graph diameter: {nx.diameter(G)}")
    
    dist_matrix = compute_distance_nodes(G)
    nodes = sorted(G.nodes())
    quad_cache = {}

    save_dir = os.path.join('data', 'tmp', graph_type, method_name)
    #if method_name == 'KL_divergence':
    #    save_dir = os.path.join('data', 'aaa', graph_type, method_name, f"geometric_temp_{geometric_temperature}")
    #elif method_name == 'entropic':
    #    save_dir = os.path.join('data', 'aaa', graph_type, method_name, f"lambda_{lambda_reg}")
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.perf_counter()

    #quads, mu = compute_local_mu(target, dist_matrix, k, temperature, geometric_temperature, lambda_reg, sigma, method_name)
    #node_contributions = compute_node_contributions(quads, mu, len(nodes))
    #draw_graph_with_values(G, pos, node_contributions,
    #                       title=f"Mu Heatmap (target={target})")
    #plot_hist(mu,
    #          title=f"Mu distribution (mean={mu.mean():.8f}, var={mu.var():.8f})",
    #          bins=20,
    #          save_path=os.path.join(save_dir, f"{target}_{k}_mu.png"))
    for k in range(1,nx.diameter(G) + 1):
        scores = compute_local_scores(nodes, dist_matrix, quad_cache, k, temperature, geometric_temperature, lambda_reg, sigma, method_name)

        elapsed = time.perf_counter() - t0
        print(f"{method_name} {k}-hop done in {elapsed:.2f}s")
        print(f"Cache size: {len(quad_cache)} vs theoretical max {math.comb(len(nodes), 4)}")
        node_contributions = np.zeros(shape=(100,))
        mu = np.zeros(shape=(100,))
        visualize_results(G, pos, node_contributions, mu, scores, target, k, temperature, save_dir)


if __name__ == "__main__":
    main()