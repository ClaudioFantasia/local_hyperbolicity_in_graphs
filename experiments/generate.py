import os
import sys
import argparse
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.utils import compute_distance_nodes, create_graph
from src.hyperbolicity.local import score_KL_divergence
from src.utils.config import load_optimization_config, load_graph_config
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout
import random
from src.hyperbolicity.gromov import compute_gromov_hyperbolicity

def hierarchical(k: int = 3, depth: int = 3, shortcuts: int = 6, seed: int = 42) -> nx.Graph:
    rng = random.Random(seed)
    
    # 1. Genera l'albero perfetto in una riga
    G = nx.balanced_tree(r=k, h=depth)
    
    # 2. Trova le foglie (nodi con grado 1, tranne la radice se depth=0)
    # Oppure, matematicamente, sono gli ultimi nodi dell'albero
    leaves = [n for n, d in G.degree() if d == 1 and n != 0]
    if depth == 0: leaves = [0] # Edge case per profondità zero
    
    # 3. Aggiungi gli shortcut EVITANDO duplicati e self-loop
    existing_shortcuts = set()
    while len(existing_shortcuts) < shortcuts:
        a, b = rng.sample(leaves, 2) # sample garantisce a != b
        edge = tuple(sorted((a, b)))
        
        if edge not in existing_shortcuts and not G.has_edge(*edge):
            G.add_edge(*edge)
            existing_shortcuts.add(edge)
            
    return G

def adapt_data_type(G,scores):
    """
    Genera un singolo oggetto PyG Data calcolando le label di iperbolicità locale.
    """
    num_nodes = G.number_of_nodes()

    nodes = list(G.nodes())
  
    y = np.zeros((num_nodes, 1), dtype=np.float32)

    y[:, 0] = scores
        
    # 3. Create PyG Data object
    # Features: node degree e costante 1
    degrees = np.array([d for n, d in G.degree()]).reshape(-1, 1).astype(np.float32)
    constants = np.ones((num_nodes, 1), dtype=np.float32)

    x = torch.tensor(np.hstack([degrees, constants]), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Edges (Direzione doppia per grafi non orientati in PyG)
    edges = list(G.edges())
    edge_index = torch.tensor([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=torch.long).t().contiguous()
    
    # Build data dict
    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    
    return data

def generate_hierarchical_params(num_graphs: int = 100, seed: int = 42):
    """
    Genera parametri per alberi gerarchici RIGOROSAMENTE sotto i 300 nodi.
    """
    rng = random.Random(seed)
    param_list = []
    
    # Configurazione: (k, depth) -> Numero esatto di nodi totali
    # (2, 4) -> 31 nodi | (2, 5) -> 63 nodi | 
    # (3, 3) -> 40 nodi | 
    # (4, 3) -> 85 nodi 
    valid_configs = [
        (2, 4), (2, 5),
        (3, 3),
        (4, 3),
    ]
    
    for _ in range(num_graphs):
        k, depth = rng.choice(valid_configs)
        
        # Foglie esatte: k^depth
        num_leaves = k ** depth
        
        # Gli shortcut non aumentano il numero di nodi (aggiungono solo archi),
        # ma non possiamo metterne più di quanti le coppie di foglie permettano.
        # Es: max_possible_shortcuts = (num_leaves * (num_leaves - 1)) // 2
        # Restiamo conservativi: da 0 shortcut fino al 25% del numero di foglie.
        max_shortcuts = max(1, int(num_leaves * 0.25))
        shortcuts = rng.randint(1, max_shortcuts)
        
        param_list.append({
            'k': k,
            'depth': depth,
            'shortcuts': shortcuts,
        })  
        
    return param_list

def compute_scores(nodes,dist_matrix,quad_cache,k,temperature):
    scores = np.array([score_KL_divergence(n, dist_matrix, quad_cache, k, temperature) for n in tqdm.tqdm(nodes, desc="Computing local scores")])
    print(f"La lunghezza di quad cache e' {len(quad_cache)}")
    return scores

def generate_hierarchical_data(params,temperature, tau):
    #G = hierarchical(**params)
    while True:
        G,pos = create_graph(type="erdos_renyi", n=50, p=0.05, seed=None)
        if nx.is_connected(G):
            break
    max_delta, _ = compute_gromov_hyperbolicity(G)
    print(max_delta)

    


    # --- Precompute ---
    nodes = sorted(G.nodes())
    quad_cache = {}
    filtration_scores = np.full(fill_value = -1, shape = len(nodes))
    dist_matrix = compute_distance_nodes(G)
    print(f"The diameter of the graph is {nx.diameter(G)}")

    k = 4
    scores = compute_scores(nodes, dist_matrix, quad_cache, k, temperature)

    """
    # --- Main loop ---
    for k in range(nx.diameter(G) + 1):
        scores = compute_scores(nodes, dist_matrix, quad_cache, k, temperature)

        mask = (scores > tau) & (filtration_scores == -1)
        filtration_scores[mask] = k

        if np.all(filtration_scores != -1):
            break
    """
    #draw_graph_with_values(G, pos, scores, title=f"Local hyperbolicity with {k}-hop")
    
    G = nx.convert_node_labels_to_integers(G)
    return G, scores


def main():
    num_graphs = 50

    # Genera la lista dei parametri distribuiti
    dataset_params = generate_hierarchical_params(num_graphs=num_graphs, seed=None)
    temperature, lambda_reg, k, target = load_optimization_config()
    tau = 0.1   
 
    dataset = []
    for params in dataset_params:
        print(params)
        G, scores = generate_hierarchical_data(params,temperature,tau)
        pos = draw_layout(G)
        data_object = adapt_data_type(G,scores)
        #print(data_object)
        dataset.append(data_object)
        #print(data_object.x)
        #print(data_object.y)
        #print(data_object.edge_index)



    # Ora `dataset` contiene i tuoi oggetti PyG pronti per il training!
    print(f"\nDataset creato con successo! Numero di grafi: {len(dataset)}")

    # Save dataset
    out_path = 'data/dataset/synthetic_hyperbolicity5.pt'
    torch.save(dataset, out_path)
    print(f"Dataset generated and saved to {out_path} ({len(dataset)} graphs)")

if __name__ == "__main__":
    main()