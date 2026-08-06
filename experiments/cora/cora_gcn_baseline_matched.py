"""
Baseline node classification (and optional node regression) experiments on
the Cora citation network, restricted to its largest connected component
(LCC).

Baseline model: PyTorch Geometric's built-in GCNConv layer
(Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
Networks", 2017).

Feature modes (selectable via --features):
    bow     -> the standard Cora bag-of-words node features (1433-dim, binary)
    custom  -> your own per-node feature vectors, loaded from a file
    concat  -> bow features concatenated with your custom features

Usage examples
---------------
# Classification baseline, standard bag-of-words features
python cora_gcn_baseline.py --task classification --features bow

# Classification with your own features only
python cora_gcn_baseline.py --task classification --features custom \
    --custom-features-path my_features.pt

# Classification with bow + custom concatenated
python cora_gcn_baseline.py --task classification --features concat \
    --custom-features-path my_features.pt

# Node regression (needs a continuous target per node)
python cora_gcn_baseline.py --task regression --features concat \
    --custom-features-path my_features.pt \
    --custom-targets-path my_targets.pt

Truncating custom features / adding node degree
-------------------------------------------------
If your custom (e.g. hyperbolic) features barely change after the first
few dimensions, use --custom-num-features m to keep only the first m
columns. --add-degree appends plain node degree as one extra feature
column, on top of whichever --features mode you picked. Together these
cover a 5-way comparison such as:

    # 1) original (bag-of-words) features only
    python cora_gcn_baseline.py --features bow

    # 2) custom features, first m dims only
    python cora_gcn_baseline.py --features custom \
        --custom-features-path my_features.pt --custom-num-features 6

    # 3) custom (first m dims) + node degree
    python cora_gcn_baseline.py --features custom \
        --custom-features-path my_features.pt --custom-num-features 6 --add-degree

    # 4) original + custom (first m dims)
    python cora_gcn_baseline.py --features concat \
        --custom-features-path my_features.pt --custom-num-features 6

    # 5) original + custom (first m dims) + node degree
    python cora_gcn_baseline.py --features concat \
        --custom-features-path my_features.pt --custom-num-features 6 --add-degree

Averaging over multiple runs
------------------------------
--seeds controls how many runs to average (default "0,1,2,3,4" -> 5 runs).
Each seed determines BOTH that run's train/val/test split and the model's
initialization/dropout, and the script reports mean +/- std across runs.
Use the exact same --seeds list when comparing different --features
configurations, so run i always trains/evaluates on the same split for
every method (a fair, paired comparison, not just independently-random
splits for each). Pass a single seed (e.g. --seeds 0) to disable averaging.

Custom feature / target file format
------------------------------------
Any of the following are accepted (detected from the file extension):
    .pt   -> a torch tensor saved with torch.save, shape [N, F] (features)
             or [N] / [N, 1] (targets).
    .npy  -> a numpy array with the same shape convention.
    .csv  -> a csv file with N rows (no header, no index column), one row
             per node.

IMPORTANT: N must equal the number of nodes in the LCC (printed at runtime,
2485 for the standard Cora graph). Rows must be ordered the same way this
script orders LCC nodes: ascending order of the *original* Planetoid node
id (i.e. take the original 2708 Cora nodes in order, drop the ones that
aren't in the largest connected component, keep the rest in that same
relative order -- that's node_map / the row order this script expects).
Use --dump-node-map to write out that exact node ordering as a .npy file
so you can build your CSV/features in matching order.

If you don't have your own features/targets yet, run with --features bow
(classification) first to sanity-check the pipeline. For regression, if you
don't pass --custom-targets-path, a synthetic demo target (log of node
degree) is used just so the script runs end-to-end -- replace it with your
real target as soon as you have it.
"""

import argparse
from math import inf

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph, to_networkx

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_cora_lcc(root="./data/Cora"):
    """Load Cora and restrict it to its largest connected component (LCC).

    Returns
    -------
    data : torch_geometric.data.Data
        Graph restricted to the LCC (edge_index re-indexed to 0..n-1).
    bow_x : torch.Tensor [n_lcc, 1433]
        Standard bag-of-words features for the LCC nodes.
    y : torch.Tensor [n_lcc]
        Class labels for the LCC nodes.
    node_map : np.ndarray [n_lcc]
        Mapping from new (LCC) node index -> original Cora node index.
        Use this to subset any external feature/target file you load.
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]

    g = to_networkx(data, to_undirected=True)
    largest_cc = max(nx.connected_components(g), key=len)
    node_map = np.array(sorted(largest_cc))
    print(f"Full graph: {data.num_nodes} nodes -> LCC: {len(node_map)} nodes "
          f"({len(node_map) / data.num_nodes:.1%})")

    subset = torch.tensor(node_map, dtype=torch.long)
    edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True,
                              num_nodes=data.num_nodes)

    bow_x = data.x[subset]
    y = data.y[subset]

    lcc_data = Data(x=bow_x, edge_index=edge_index, y=y, num_nodes=len(node_map))
    return lcc_data, bow_x, y, node_map


def _parse_bracket_array(s):
    """Parse a numpy-print-style array string, e.g. '[0.1 0.2\n 0.3]',
    into a 1-D numpy float array. Handles arbitrary whitespace/newlines
    between numbers (numpy's default str(array) wraps long rows)."""
    return np.array(s.strip().strip("[]").split(), dtype=float)


def load_external_array(path, node_map):
    """Load a feature/target file with one row per LCC node (len(node_map)
    rows), ordered by ascending original Cora node id (the same order as
    node_map / --dump-node-map).

    Supported formats:
      .pt / .npy        -> a plain array/tensor, shape [N, F] or [N].
      .csv (wide)        -> a plain numeric csv, one column per feature,
                             one row per node. A leading index column
                             (0..N-1), e.g. from pandas df.to_csv() with
                             index=True, is detected and dropped.
      .csv (long format)  -> a csv with a header and exactly two columns:
                             an id column (e.g. "node_id") and a value
                             column (e.g. "metric_result") containing a
                             numpy-print-style stringified array per row,
                             such as "[0.1 0.2 0.3\n 0.4 0.5]". Rows are
                             sorted by id before use.
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
            # Long format: id column + stringified-array value column.
            id_col, val_col = df.columns[0], df.columns[1]
            df = df.sort_values(id_col).reset_index(drop=True)
            vectors = df[val_col].apply(_parse_bracket_array)
            lengths = vectors.apply(len)
            if lengths.nunique() > 1:
                raise ValueError(
                    f"{path}: rows in '{val_col}' have inconsistent vector "
                    f"lengths ({sorted(lengths.unique())}); every node's "
                    f"vector must be the same length."
                )
            arr = np.stack(vectors.values)
        else:
            # Wide format: plain numeric matrix, possibly with a leading
            # index column and/or a header row (pandas handles both).
            arr = df.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported file extension for {path}")

    if arr.ndim == 2 and arr.shape[1] > 1:
        first_col = arr[:, 0]
        if np.allclose(first_col, np.arange(arr.shape[0])):
            print(f"[info] {path}: detected a leading index column "
                  f"(0..{arr.shape[0] - 1}) and dropped it before use.")
            arr = arr[:, 1:]

    if arr.shape[0] != len(node_map):
        raise ValueError(
            f"Expected {len(node_map)} rows (one per LCC node), got "
            f"{arr.shape[0]}. Your file must contain a row for every LCC "
            f"node only (not the full 2708-node Cora graph), ordered by "
            f"ascending original node id -- use --dump-node-map to check "
            f"the exact ordering expected."
        )
    return torch.tensor(arr, dtype=torch.float)


def compute_degree(edge_index, num_nodes):
    """Plain node degree (float tensor, shape [num_nodes])."""
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    return deg


def build_features(feat_mode, bow_x, node_map, custom_features_path,
                    custom_num_features=None, add_degree=False,
                    edge_index=None, num_nodes=None):
    """Build the node feature matrix for one experiment.

    feat_mode: "bow", "custom", or "concat" (bow ++ custom).
    custom_num_features: if set, only the first m columns of the loaded
        custom features are used (handy when later dimensions of e.g. a
        hyperbolic embedding carry little extra signal beyond a few
        significant digits).
    add_degree: if True, appends plain node degree as one extra feature
        column, after the bow/custom/concat features are assembled.

    This lets you cover combinations like:
        bow                                   -> features=bow
        custom (first m dims)                 -> features=custom, custom_num_features=m
        custom (first m dims) + degree         -> features=custom, custom_num_features=m, add_degree=True
        bow + custom (first m dims)           -> features=concat, custom_num_features=m
        bow + custom (first m dims) + degree   -> features=concat, custom_num_features=m, add_degree=True
    """
    if feat_mode == "bow":
        x = bow_x
    else:
        if custom_features_path is None:
            raise ValueError(f"--features {feat_mode} requires --custom-features-path")
        custom_x = load_external_array(custom_features_path, node_map)
        if custom_x.dim() == 1:
            custom_x = custom_x.unsqueeze(1)

        if custom_num_features is not None:
            if custom_num_features > custom_x.shape[1]:
                raise ValueError(
                    f"--custom-num-features {custom_num_features} exceeds the "
                    f"{custom_x.shape[1]} columns available in the custom "
                    f"feature file."
                )
            custom_x = custom_x[:, :custom_num_features]

        if feat_mode == "custom":
            x = custom_x
        elif feat_mode == "concat":
            x = torch.cat([bow_x, custom_x], dim=1)
        else:
            raise ValueError(f"Unknown feature mode: {feat_mode}")

    if add_degree:
        deg = compute_degree(edge_index, num_nodes).unsqueeze(1)
        x = torch.cat([x, deg], dim=1)

    return x


def build_targets(task, y_cls, node_map, custom_targets_path, edge_index, num_nodes):
    if task == "classification":
        return y_cls, int(y_cls.max().item()) + 1

    # regression
    if custom_targets_path is not None:
        t = load_external_array(custom_targets_path, node_map).squeeze()
        return t, 1

    print("[warning] no --custom-targets-path given for regression: using "
          "log(1 + node degree) as a synthetic demo target. Replace this "
          "with your real per-node target as soon as you have one.")
    deg = compute_degree(edge_index, num_nodes)
    return torch.log1p(deg), 1


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class GCN(torch.nn.Module):
    """GCN baseline matching Local_Curvature_Profile's models/node_model.py
    architecture: an arbitrary number of GCNConv layers (default 3), with
    ReLU + dropout after every layer except the last."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = torch.nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def make_splits(num_nodes, seed, train_frac=0.5, val_frac=0.25):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(num_nodes)
    n_train = int(train_frac * num_nodes)
    n_val = int(val_frac * num_nodes)
    train_idx, val_idx, test_idx = (
        idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    )

    def mask(ids):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[torch.tensor(ids)] = True
        return m

    return mask(train_idx), mask(val_idx), mask(test_idx)


def train(model, edge_index, x, y, train_mask, val_mask, test_mask, task,
          max_epochs, lr, weight_decay, device, patience=100,
          stopping_threshold=1.01, verbose=True):
    """Matches Local_Curvature_Profile's experiments/node_classification.py
    Experiment.run(): Adam (NOTE: their code sets weight_decay in
    default_args but never actually passes it into torch.optim.Adam, so
    their real weight_decay is 0 -- pass weight_decay=0 here to reproduce
    that, or keep a nonzero value if you'd rather fix that apparent bug),
    ReduceLROnPlateau(patience=25), and early stopping on validation
    accuracy with the given patience/threshold, up to max_epochs."""
    model = model.to(device)
    x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
    train_mask, val_mask, test_mask = (
        train_mask.to(device), val_mask.to(device), test_mask.to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25)
    metric_name = "acc" if task == "classification" else "mse"

    best_val, best_test = (0.0, 0.0) if task == "classification" else (inf, inf)
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)

        if task == "classification":
            loss = F.cross_entropy(out[train_mask], y[train_mask])
        else:
            loss = F.mse_loss(out.squeeze(-1)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            if task == "classification":
                pred = out.argmax(dim=1)
                train_m = (pred[train_mask] == y[train_mask]).float().mean().item()
                val_m = (pred[val_mask] == y[val_mask]).float().mean().item()
                test_m = (pred[test_mask] == y[test_mask]).float().mean().item()
            else:
                pred = out.squeeze(-1)
                train_m = F.mse_loss(pred[train_mask], y[train_mask]).item()
                val_m = F.mse_loss(pred[val_mask], y[val_mask]).item()
                test_m = F.mse_loss(pred[test_mask], y[test_mask]).item()

        scheduler.step(val_m if task == "regression" else -val_m)

        if task == "classification":
            improved = val_m > best_val * stopping_threshold if best_val > 0 else True
        else:
            improved = val_m < best_val / stopping_threshold if best_val < inf else True

        if improved:
            best_val, best_test = val_m, test_m
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"epoch {epoch:03d} | loss {loss.item():.4f} | "
                  f"train {metric_name} {train_m:.4f} | "
                  f"val {metric_name} {val_m:.4f} | "
                  f"test {metric_name} {test_m:.4f}")

        if epochs_no_improve > patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
            break

    if verbose:
        print(f"Best val {metric_name}: {best_val:.4f} | "
              f"corresponding test {metric_name}: {best_test:.4f}")
    return best_val, best_test


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--features", choices=["bow", "custom", "concat"], default="bow")
    parser.add_argument("--custom-features-path", default=None)
    parser.add_argument("--custom-num-features", type=int, default=None,
                         help="If set, only use the first m columns of the "
                              "custom feature file (e.g. useful when later "
                              "dimensions of an embedding are nearly "
                              "constant / carry little extra signal).")
    parser.add_argument("--add-degree", action="store_true",
                         help="Append plain node degree as one extra "
                              "feature column, on top of whichever "
                              "--features mode is selected.")
    parser.add_argument("--custom-targets-path", default=None)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000000,
                         help="Max epochs; early stopping (--patience) will "
                              "almost always stop well before this, as in "
                              "their Experiment.run().")
    parser.add_argument("--patience", type=int, default=100,
                         help="Early-stopping patience on validation acc, "
                              "matching their default_args['patience'].")
    parser.add_argument("--stopping-threshold", type=float, default=1.01,
                         help="Val acc must exceed best_val * this factor "
                              "to count as an improvement, matching their "
                              "default_args['stopping_threshold'].")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                         help="Their default_args sets weight_decay=1e-5 but "
                              "their code never actually passes it into "
                              "torch.optim.Adam(...), so their real "
                              "effective weight_decay is 0. Default here "
                              "matches that real behavior; pass "
                              "--weight-decay 1e-5 if you'd rather match "
                              "their config value instead of their code's "
                              "actual (buggy) behavior.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                         help="Comma-separated list of seeds, e.g. '0,1,2,3,4'. "
                              "Each seed controls BOTH the train/val/test split "
                              "and the model's initialization/dropout for that "
                              "run, so the script trains once per seed and "
                              "reports mean +/- std across runs. Use the SAME "
                              "--seeds list across the different feature "
                              "configurations you want to compare, so run i "
                              "always uses the same split for every method "
                              "(a fair, paired comparison). Pass a single "
                              "value (e.g. '0') for one run only.")
    parser.add_argument("--data-root", default="./data/Cora")
    parser.add_argument("--dump-node-map", default=None,
                         help="Optional path (.npy) to save the LCC node_map "
                              "(original Cora node ids, in the order this "
                              "script expects your custom feature/target "
                              "rows to be in), then exit.")
    args = parser.parse_args()

    #seeds = [int(s) for s in args.seeds.split(",")]
    seeds = np.arange(0,101,1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, bow_x, y_cls, node_map = load_cora_lcc(root=args.data_root)
    num_nodes = data.num_nodes

    if args.dump_node_map is not None:
        np.save(args.dump_node_map, node_map)
        print(f"Saved LCC node_map ({len(node_map)} original Cora node ids, "
              f"ascending order) to {args.dump_node_map}")
        return

    # Feature/target construction is deterministic given the data and CLI
    # args (no randomness involved), so it only needs to happen once and is
    # then reused, unchanged, across every seed.
    x = build_features(
        args.features, bow_x, node_map, args.custom_features_path,
        custom_num_features=args.custom_num_features,
        add_degree=args.add_degree,
        edge_index=data.edge_index, num_nodes=num_nodes,
    )
    y, out_dim = build_targets(
        args.task, y_cls, node_map, args.custom_targets_path, data.edge_index, num_nodes
    )

    degree_note = " + degree" if args.add_degree else ""
    trunc_note = f" (first {args.custom_num_features} custom dims)" if args.custom_num_features else ""
    print(f"\nTask: {args.task} | Features: {args.features}{trunc_note}{degree_note} "
          f"(dim={x.shape[1]}) | Nodes: {num_nodes} | "
          f"Edges: {data.edge_index.shape[1] // 2} | Seeds: {seeds}")

    metric_name = "acc" if args.task == "classification" else "mse"
    val_scores, test_scores = [], []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Same seed -> same split. Running the same list of seeds across
        # different --features configurations means run i is always
        # evaluated on the same train/val/test partition for every config.
        train_mask, val_mask, test_mask = make_splits(num_nodes, seed=seed)

        model = GCN(in_dim=x.shape[1], hidden_dim=args.hidden_dim,
                    out_dim=out_dim, num_layers=args.num_layers,
                    dropout=args.dropout)

        best_val, best_test = train(
            model, data.edge_index, x, y, train_mask, val_mask, test_mask,
            args.task, args.epochs, args.lr, args.weight_decay, device,
            patience=args.patience, stopping_threshold=args.stopping_threshold,
            verbose=(len(seeds) == 1),
        )
        val_scores.append(best_val)
        test_scores.append(best_test)
        print(f"seed {seed:>3d} | val {metric_name} {best_val:.4f} | "
              f"test {metric_name} {best_test:.4f}")

    val_scores, test_scores = np.array(val_scores), np.array(test_scores)
    print(f"\nAcross {len(seeds)} seed(s):")
    print(f"  val  {metric_name}: {val_scores.mean():.4f} +/- {val_scores.std():.4f}")
    print(f"  test {metric_name}: {test_scores.mean():.4f} +/- {test_scores.std():.4f}")


if __name__ == "__main__":
    main()