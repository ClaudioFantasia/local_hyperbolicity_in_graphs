"""
Shared plumbing for the dataset experiments in this folder.

Everything here is dataset-agnostic: pick a dataset by name, get its
graph(s) back with nodes relabeled 0..n-1, and load/assemble node
features on top. Adding a dataset is a one-line change to DATASETS or
TU_DATASETS.

Two families of dataset live here, because the tasks need different
shapes:

  DATASETS     one big graph, node-level tasks (cora / citeseer /
               pubmed). `load_lcc` returns its largest connected
               component.
  TU_DATASETS  a collection of small graphs, graph-level tasks (mutag /
               proteins / ...). `load_tu` returns the list of graphs.

Node ordering convention
------------------------
For the single-graph datasets every script works on the largest
connected component (LCC) only, with LCC nodes taken in ascending order
of their *original* dataset node id. `node_map[i]` is the original id of
LCC node i. Any external feature file (e.g. the csv written by
generate_features.py) must have one row per LCC node, in that same
order.

For the TU datasets each graph already comes numbered 0..n_g-1, and an
external feature file must cover every node of every graph -- see
`load_graph_node_features`.
"""

import csv
import itertools
import json
import os
import random
import re

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import subgraph, to_networkx

# repo root, so the scripts work from any working directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
# tutto cio' che gli script generano sta sotto data/, mai accanto al codice:
# i profili di iperbolicita' per nodo e le curve di training per epoca
FEATURES_ROOT = os.path.join(DATA_ROOT, "hyperbolic_features")
CURVES_ROOT = os.path.join(DATA_ROOT, "curves")

# --dataset choice -> the name Planetoid expects. Add a line here to make
# a new citation-style dataset available to every script in this folder.
DATASETS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}

# --dataset choice -> the name TUDataset expects. These are collections of
# small graphs with one label per graph, for graph_classification.py.
# Adding a line here makes a new benchmark available everywhere.
TU_DATASETS = {
    "mutag": "MUTAG",
    "proteins": "PROTEINS",
    "enzymes": "ENZYMES",
    "nci1": "NCI1",
    "imdb-binary": "IMDB-BINARY",
}


def metrics_path(dataset):
    """Where generate_features.py writes / the other scripts read the
    per-node hyperbolicity scores for `dataset`."""
    os.makedirs(FEATURES_ROOT, exist_ok=True)
    return os.path.join(FEATURES_ROOT, f"{dataset}_node_metrics.csv")


def feature_tag(args):
    """Etichetta compatta della configurazione di feature di un run, per i
    nomi automatici dei file di output: bow, concat-std, custom8-m4-deg..."""
    tag = args.features
    if args.custom_num_features:
        tag += str(args.custom_num_features)
    if args.custom_features_path and args.features in ("custom", "concat"):
        # cora_node_metrics_m4.csv -> m4, cosi' due run sullo stesso dataset
        # ma su file di feature diversi non si sovrascrivono a vicenda
        stem = os.path.basename(args.custom_features_path)[:-len(".csv")]
        suffix = stem.replace(f"{args.dataset}_node_metrics", "").strip("_")
        if suffix:
            tag += "-" + suffix
    if args.add_degree:
        tag += "-deg"
    if args.standardize_custom:
        tag += "-std"
    return tag


def check_meta(path, num_nodes, num_edges):
    """Verifica che il csv di feature `path` sia stato calcolato sul grafo
    che stiamo caricando adesso, confrontandolo con l'impronta che
    generate_features.py scrive accanto ad esso (<file>.meta.json).

    Il numero di righe lo controlla gia' chi legge; qui il controllo che
    conta e' il numero di archi. Due grafi con lo stesso numero di nodi ma
    archi diversi -- una LCC estratta da una versione precedente del
    loader, un dataset diverso di taglia simile -- passerebbero l'assert
    sulle righe e produrrebbero risultati plausibili e falsi.

    Stampa anche con che parametri il profilo e' stato generato, che
    altrimenti non sono scritti da nessuna parte."""
    meta_file = path.replace(".csv", ".meta.json")
    if not os.path.exists(meta_file):
        print(f"[warning] {os.path.basename(path)}: manca il .meta.json accanto, "
              f"non posso verificare su che grafo e' stato calcolato ne' con "
              f"che parametri -- rigeneralo con generate_features.py per averlo")
        return

    with open(meta_file) as f:
        meta = json.load(f)

    assert meta["num_nodes"] == num_nodes and meta["num_edges"] == num_edges, (
        f"{path} e' stato calcolato su {meta['dataset']} con {meta['num_nodes']} "
        f"nodi e {meta['num_edges']} archi, ma il grafo caricato ora ne ha "
        f"{num_nodes} e {num_edges}: non e' lo stesso grafo, le righe del csv "
        f"non corrispondono ai suoi nodi"
    )

    t_geom = meta["geometric_temperature"]
    print(f"{os.path.basename(path)}: profilo di {meta['dataset']}, k={meta['k']}, "
          f"temperature={meta['temperature']}, {len(t_geom)} scale T_geom in "
          f"[{min(t_geom):.2f}, {max(t_geom):.2f}], strategia {meta['strategy']}"
          + (f" (m={meta['m']}, seed={meta['seed']})"
             if meta["strategy"] == "increasing_neighborhood" else ""))


def curves_path(*parts):
    """data/curves/<parts unite da _>.csv -- il nome che --curves si sceglie
    da solo quando lo passi senza argomento."""
    os.makedirs(CURVES_ROOT, exist_ok=True)
    return os.path.join(CURVES_ROOT, "_".join(str(p) for p in parts) + ".csv")


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


def load_tu(dataset):
    """Load a TU graph-classification benchmark (see TU_DATASETS) as a
    plain list of torch_geometric Data objects.

    Each graph already comes with its nodes numbered 0..n_g-1, in the
    same order as its .x rows -- which is also the order
    `compute_distance_nodes` / `KL_score` index by, so no relabeling is
    needed. Unlike the citation datasets nothing is thrown away: a
    disconnected graph is kept whole, since every k-hop ball lives
    inside a single component anyway and that is all the scoring touches.

    Benchmarks with no node attributes at all (imdb-binary) get node
    degree as their single feature column, so every graph has a usable .x.
    """
    ds = TUDataset(root=DATA_ROOT, name=TU_DATASETS[dataset])
    graphs = list(ds)

    if graphs[0].x is None:
        print(f"[info] {TU_DATASETS[dataset]} has no node attributes: using degree")
        for d in graphs:
            d.x = node_degrees(d.edge_index, d.num_nodes)[:, None]

    sizes = np.array([d.num_nodes for d in graphs])
    print(f"{TU_DATASETS[dataset]}: {len(graphs)} graphs, "
          f"{sizes.min()}-{sizes.max()} nodes (mean {sizes.mean():.1f}), "
          f"{graphs[0].x.shape[1]} node features, "
          f"{int(max(d.y for d in graphs)) + 1} classes")
    return graphs


def tu_graph(dataset, index):
    """One graph of a TU benchmark as a plain networkx graph.

    This is what `citation_patch` is for cora/citeseer -- a real graph
    small enough to score exactly -- except no BFS patch is needed: a TU
    graph is already tiny (a MUTAG molecule is 10-28 nodes, so
    sampling_quads enumerates every quad and G*(v) is exact). Used by
    experiments/synthetic/run_experiment.py.
    """
    graphs = load_tu(dataset)
    G = to_nx(graphs[index])
    print(f"{dataset}[{index}]: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, label y={int(graphs[index].y)}")
    return G


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


def load_graph_node_features(path, sizes):
    """Per-node features for a multi-graph (TU) dataset.

    Reads the long csv generate_features.py writes for those datasets --
    a `graph_id`, `node_id`, `metric_result` triple per node, where
    metric_result is the stringified profile -- and splits it back into
    one [n_g, F] float tensor per graph, in graph order.

    `sizes[g]` is the number of nodes of graph g; the file must cover
    every node of every graph.
    """
    import pandas as pd

    df = pd.read_csv(path).sort_values(["graph_id", "node_id"])

    blocks = []
    for graph_id, size in enumerate(sizes):
        rows = df[df["graph_id"] == graph_id]
        assert len(rows) == size, (
            f"{path}: graph {graph_id} has {len(rows)} rows but {size} nodes -- "
            f"rerun generate_features.py, it stopped early"
        )
        profiles = np.stack(rows["metric_result"].apply(_parse_bracket_array).values)
        blocks.append(torch.tensor(profiles, dtype=torch.float))

    lengths = {b.shape[1] for b in blocks}
    assert len(lengths) == 1, f"{path}: graphs have different profile lengths {lengths}"
    print(f"{path}: {len(blocks)} graphs, {sum(sizes)} nodes, "
          f"{blocks[0].shape[1]} columns per node")
    return blocks


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def write_curves(path, columns, rows):
    """Dump the per-epoch training history of a run, long format: one row
    per (run, epoch), where a "run" is a seed (or a seed/fold pair).

    Every training script writes its own columns -- the only fixed ones
    are `epoch` and `loss` -- and plot_curves.py plots whatever it finds.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Curve di training scritte in {path} ({len(rows)} righe)")


def node_degrees(edge_index, num_nodes):
    """Plain node degree as a float tensor of shape [num_nodes]."""
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    return deg


def build_features(mode, bow_x, edge_index, num_nodes, custom_path=None,
                   custom_x=None, num_custom=None, add_degree=False,
                   standardize_custom=False):
    """Assemble the node feature matrix for one run.

    mode          : "bow" (the dataset's own bag-of-words), "custom" (the
                    external file only), "concat" (bow ++ custom), or
                    "degree" (node degree as the only column -- the pure
                    structural baseline: whatever a model reaches with
                    it, it reached without knowing anything about the
                    dataset's own node attributes).
    custom_x      : the custom block already in memory, which skips
                    reading custom_path. graph_classification.py uses
                    this: it loads every graph's block in one go and then
                    calls this once per graph.
    num_custom    : keep only the first m columns of the custom file --
                    handy since the hyperbolicity profile's later
                    T_geom columns carry little extra signal.
    add_degree    : append node degree as one extra column at the end.
    standardize_custom : zero-mean / unit-variance each custom column.
                    The hyperbolicity columns carry a large constant
                    offset (mean/std ~ 4 on cora, ~3.6 on citeseer) that
                    holds no per-node information, so it only shifts the
                    block off the scale of the bow columns. The bow block
                    is deliberately left alone: it is binary, so its
                    columns are already commensurate, and standardizing
                    it would multiply the rarest words by 10-20x.
    """
    if mode == "bow":
        x = bow_x
    elif mode == "degree":
        # baseline strutturale puro: nessuna feature del dataset, solo il grado
        x = node_degrees(edge_index, num_nodes)[:, None]
    else:
        if custom_x is None:
            assert custom_path is not None, f"--features {mode} needs --custom-features-path"
            check_meta(custom_path, num_nodes, edge_index.shape[1] // 2)
            custom_x = load_node_features(custom_path, num_nodes)
        if num_custom is not None:
            assert num_custom <= custom_x.shape[1], \
                f"--custom-num-features {num_custom} > {custom_x.shape[1]} columns available"
            custom_x = custom_x[:, :num_custom]

        if standardize_custom:
            custom_x = (custom_x - custom_x.mean(0)) / (custom_x.std(0) + 1e-8)

        if mode == "custom":
            x = custom_x
        elif mode == "concat":
            x = torch.cat([bow_x, custom_x], dim=1)
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

    if add_degree:
        x = torch.cat([x, node_degrees(edge_index, num_nodes)[:, None]], dim=1)

    return x
