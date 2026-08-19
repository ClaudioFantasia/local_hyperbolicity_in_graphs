"""
Shared plumbing for the dataset experiments in this folder.

Everything here is dataset-agnostic: pick a dataset by name (see
DATASETS), get back its largest connected component with nodes relabeled
0..n-1, and load/assemble node features on top of it. The per-task
scripts (generate_features.py, node_classification.py,
link_prediction.py, khop_analysis.py) all sit on this so that adding a
new Planetoid-style dataset is a one-line change to DATASETS.

Node ordering convention
------------------------
Every script here works on the largest connected component (LCC) only,
with LCC nodes taken in ascending order of their *original* dataset node
id. `node_map[i]` is the original id of LCC node i. Any external feature
file (e.g. the csv written by generate_features.py) must have one row per
LCC node, in that same order.
"""

import itertools
import os
import random
import re

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph, to_networkx

# repo root, so the scripts work from any working directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# --dataset choice -> the name Planetoid expects. Add a line here to make
# a new citation-style dataset available to every script in this folder.
DATASETS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def metrics_path(dataset):
    """Where generate_features.py writes / the other scripts read the
    per-node hyperbolicity scores for `dataset`."""
    return os.path.join(os.path.dirname(__file__), f"{dataset}_node_metrics.csv")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_lcc(dataset):
    """Load `dataset` and restrict it to its largest connected component.

    Returns
    -------
    data : torch_geometric.data.Data
        x / edge_index / y for the LCC, node ids relabeled to 0..n-1.
    node_map : np.ndarray, shape [n_lcc]
        LCC node i -> original dataset node id (ascending).
    """
    ds = Planetoid(root=DATA_ROOT, name=DATASETS[dataset])
    full = ds[0]

    G = to_networkx(full, to_undirected=True)
    node_map = np.array(sorted(max(nx.connected_components(G), key=len)))
    subset = torch.tensor(node_map, dtype=torch.long)

    edge_index, _ = subgraph(subset, full.edge_index, relabel_nodes=True,
                             num_nodes=full.num_nodes)

    data = Data(x=full.x[subset], edge_index=edge_index, y=full.y[subset],
                num_nodes=len(node_map))

    print(f"{DATASETS[dataset]}: {full.num_nodes} nodes -> LCC {len(node_map)} "
          f"({len(node_map) / full.num_nodes:.1%}), "
          f"{edge_index.shape[1] // 2} undirected edges")
    return data, node_map


def to_nx(data):
    """The LCC as a plain undirected networkx graph on nodes 0..n-1,
    which is the form src/optimization expects."""
    return to_networkx(data, to_undirected=True)


def citation_patch(dataset, n=100, seed=0, source_node=None):
    """A small connected piece of `dataset`, as a networkx graph on nodes
    0..n-1: BFS from a node of the LCC, keep the first n nodes reached,
    take the induced subgraph (a BFS prefix is connected by construction).

    `source_node` is the LCC node id to start the BFS from (LCC ids, i.e.
    the 0..n_lcc-1 relabeling of load_lcc, *not* the original Cora ids --
    node_map[i] is the original id of LCC node i). With source_node=None
    it is drawn at random using `seed`.

    The point is to have a graph with the degree distribution and the
    local structure of a real citation network, but small enough that
    exact quantities are computable -- on a 100-node patch every k-hop
    ball has at most 100 nodes, so sampling_quads enumerates all the
    quads and G*(v) is exact.

    The patch has an artificial boundary: nodes near it lost part of
    their neighborhood, so their G* is not the G* they have in the full
    graph. It is a realistic *test graph*, not a sub-sample of the real
    scores.
    """
    data, _ = load_lcc(dataset)
    G = to_nx(data)

    start = source_node if source_node is not None else random.Random(seed).choice(list(G.nodes()))
    kept = list(itertools.islice(nx.bfs_tree(G, start), n))
    patch = nx.convert_node_labels_to_integers(G.subgraph(kept), ordering="sorted")

    # il relabeling e' per id crescente, quindi la sorgente nel patch e' il
    # suo rango fra i nodi tenuti: serve per puntarci target_nodes
    start_in_patch = sorted(kept).index(start)

    print(f"patch di {dataset}: {patch.number_of_nodes()} nodi, "
          f"{patch.number_of_edges()} archi, da BFS su {start} "
          f"(= nodo {start_in_patch} nel patch)")
    return patch


# ----------------------------------------------------------------------
# External per-node features
# ----------------------------------------------------------------------

def _parse_bracket_array(s):
    """Parse a numpy-print-style string like '[0.1 0.2\n 0.3]' into a
    1-D float array. KL_score returns one value per geometric_temperature,
    and generate_features.py writes that vector straight to csv, so this
    is the format of the metric_result column."""
    return np.array(re.split(r"\s+", s.strip().strip("[]").strip()), dtype=float)


def load_node_features(path, num_nodes):
    """Load a per-node feature matrix, one row per LCC node (see the node
    ordering note at the top of this module). Returns a float tensor of
    shape [num_nodes, F].

    Accepted formats, detected from the extension:
      .pt / .npy  -- an array of shape [N, F] or [N].
      .csv        -- either a plain numeric matrix (an optional leading
                     0..N-1 index column is detected and dropped), or the
                     long format written by generate_features.py: a
                     node_id column plus a metric_result column holding a
                     stringified array per row.
    """
    if path.endswith(".pt"):
        arr = torch.load(path)
        arr = arr.numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    elif path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".csv"):
        import pandas as pd

        with open(path) as f:
            first_line = f.readline()
        try:
            [float(t) for t in first_line.strip().split(",")]
            has_header = False
        except ValueError:
            has_header = True

        df = pd.read_csv(path, header=0 if has_header else None)

        if df.shape[1] == 2 and pd.api.types.is_string_dtype(df.iloc[:, 1]):
            # long format: node_id + stringified vector
            id_col, val_col = df.columns[0], df.columns[1]
            df = df.sort_values(id_col).reset_index(drop=True)
            vectors = df[val_col].apply(_parse_bracket_array)
            lengths = vectors.apply(len)
            assert lengths.nunique() == 1, \
                f"{path}: rows have different vector lengths {sorted(lengths.unique())}"
            arr = np.stack(vectors.values)
        else:
            arr = df.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported file extension for {path}")

    if arr.ndim == 1:
        arr = arr[:, None]

    # pandas' df.to_csv(index=True) leaves a 0..N-1 column in front
    if arr.shape[1] > 1 and np.allclose(arr[:, 0], np.arange(arr.shape[0])):
        print(f"[info] {path}: dropped the leading index column")
        arr = arr[:, 1:]

    assert arr.shape[0] == num_nodes, (
        f"{path} has {arr.shape[0]} rows but the LCC has {num_nodes} nodes -- "
        f"the file must cover the LCC only, in ascending original-node-id order"
    )
    return torch.tensor(arr, dtype=torch.float)


def node_degrees(edge_index, num_nodes):
    """Plain node degree as a float tensor of shape [num_nodes]."""
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    return deg


def build_features(mode, bow_x, edge_index, num_nodes, custom_path=None,
                   num_custom=None, add_degree=False):
    """Assemble the node feature matrix for one run.

    mode          : "bow" (the dataset's own bag-of-words), "custom" (the
                    external file only), or "concat" (bow ++ custom).
    num_custom    : keep only the first m columns of the custom file --
                    handy since the hyperbolicity profile's later
                    T_geom columns carry little extra signal.
    add_degree    : append node degree as one extra column at the end.
    """
    if mode == "bow":
        x = bow_x
    else:
        assert custom_path is not None, f"--features {mode} needs --custom-features-path"
        custom_x = load_node_features(custom_path, num_nodes)
        if num_custom is not None:
            assert num_custom <= custom_x.shape[1], \
                f"--custom-num-features {num_custom} > {custom_x.shape[1]} columns available"
            custom_x = custom_x[:, :num_custom]

        if mode == "custom":
            x = custom_x
        elif mode == "concat":
            x = torch.cat([bow_x, custom_x], dim=1)
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

    if add_degree:
        x = torch.cat([x, node_degrees(edge_index, num_nodes)[:, None]], dim=1)

    return x
