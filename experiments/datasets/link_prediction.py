"""
Link prediction on a citation network's largest connected component,
same feature modes as node_classification.py so the hyperbolicity
profile can be tested on a second downstream task.

Baselines reported side by side:
    1. Common neighbors      (heuristic, feature-free)
    2. Jaccard coefficient   (heuristic, feature-free)
    3. Adamic-Adar index     (heuristic, feature-free)
    4. GCN encoder + dot-product decoder  (uses the node features)

Only the GCN sees --features, so the three heuristics act as a fixed
reference line across every feature configuration.

Usage
-----
    python experiments/datasets/link_prediction.py --dataset cora --features bow
    python experiments/datasets/link_prediction.py --dataset cora --features custom
    python experiments/datasets/link_prediction.py --dataset citeseer --features concat

--custom-features-path defaults to
data/hyperbolic_features/<dataset>_node_metrics.csv. No feature block is
rescaled: bow and custom go in as they are, so
--features bow and --features concat differ by the hyperbolicity columns
and nothing else. --add-degree appends a log-scaled, standardized degree
column on top (off by default, as in node_classification.py).

--seeds picks how many runs to average (default 10). A seed fixes BOTH the
edge split and the model init, so running the same --seeds list across
feature modes gives a paired comparison on identical splits.

Metrics: ROC-AUC and average precision on the held-out edges.
"""

import argparse
import os
import random
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common import (DATASETS, build_features, curves_path, feature_tag,
                    load_lcc, metrics_path, node_degrees, write_curves)


# ----------------------------------------------------------------------
# Heuristic baselines
# ----------------------------------------------------------------------

def score_heuristic(G, edge_label_index, heuristic):
    """Score each candidate pair with a networkx link-prediction index."""
    pairs = list(zip(edge_label_index[0].tolist(), edge_label_index[1].tolist()))

    if heuristic == "common_neighbors":
        raw = nx.common_neighbor_centrality(G, pairs, alpha=1.0)
    elif heuristic == "jaccard":
        raw = nx.jaccard_coefficient(G, pairs)
    elif heuristic == "adamic_adar":
        raw = nx.adamic_adar_index(G, pairs)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    return np.array([s for _, _, s in raw])


def evaluate_heuristics(train_data, val_data, test_data, verbose):
    # heuristics only get to see the message-passing (training) edges
    G = nx.Graph()
    G.add_nodes_from(range(train_data.num_nodes))
    G.add_edges_from(train_data.edge_index.t().tolist())

    if verbose:
        print("=== Heuristic baselines (feature-free) ===")
        print(f"{'Method':<20} {'Val AUC':>8} {'Val AP':>8} {'Test AUC':>9} {'Test AP':>8}")

    results = {}
    for h in ["common_neighbors", "jaccard", "adamic_adar"]:
        val_s = score_heuristic(G, val_data.edge_label_index, h)
        test_s = score_heuristic(G, test_data.edge_label_index, h)
        val_y, test_y = val_data.edge_label.numpy(), test_data.edge_label.numpy()

        results[h] = {
            "val_auc": roc_auc_score(val_y, val_s),
            "val_ap": average_precision_score(val_y, val_s),
            "test_auc": roc_auc_score(test_y, test_s),
            "test_ap": average_precision_score(test_y, test_s),
        }
        r = results[h]
        if verbose:
            print(f"{h:<20} {r['val_auc']:>8.4f} {r['val_ap']:>8.4f} "
                  f"{r['test_auc']:>9.4f} {r['test_ap']:>8.4f}")
    return results


# ----------------------------------------------------------------------
# GCN encoder + dot-product decoder
# ----------------------------------------------------------------------

class LinkPredGCN(nn.Module):
    """Same encoder as node_classification.py's GCN (num_layers GCNConv,
    ReLU + dropout after all but the last, so the last layer's output is
    the embedding), plus a dot-product decoder on node pairs.

    GCNConv already does D^-1/2 (A + I) D^-1/2 H W by default
    (add_self_loops=True, normalize=True), so there is nothing to write
    out by hand here."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=32,
                 num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = dropout

    def encode(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


@torch.no_grad()
def evaluate_gcn(model, data, device):
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device))
    scores = torch.sigmoid(model.decode(z, data.edge_label_index.to(device))).cpu().numpy()
    labels = data.edge_label.numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_gcn(train_data, val_data, test_data, args, device, verbose):
    model = LinkPredGCN(train_data.x.shape[1], args.hidden_dim, args.out_dim,
                        args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_auc, best_epoch, best_state, no_improve = 0.0, 0, None, 0
    # loss e metriche di validazione sono gia' calcolate a ogni epoca:
    # registrarle non costa niente. Il test NON viene toccato nel ciclo,
    # quindi le curve non contengono una colonna di test
    history = []

    if verbose:
        print("\n=== GCN baseline (training) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
        logits = model.decode(z, train_data.edge_label_index.to(device))
        loss = F.binary_cross_entropy_with_logits(
            logits, train_data.edge_label.float().to(device))
        loss.backward()
        optimizer.step()

        val_auc, val_ap = evaluate_gcn(model, val_data, device)
        history.append((epoch, loss.item(), val_auc, val_ap))

        if val_auc > best_val_auc:
            best_val_auc, best_epoch = val_auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3} | loss {loss.item():.4f} | "
                  f"val AUC {val_auc:.4f} | val AP {val_ap:.4f}")

        if no_improve >= args.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best: {best_epoch})")
            break

    model.load_state_dict(best_state)
    val_auc, val_ap = evaluate_gcn(model, val_data, device)
    test_auc, test_ap = evaluate_gcn(model, test_data, device)
    if verbose:
        print(f"\n  best epoch {best_epoch}: val AUC {val_auc:.4f}  val AP {val_ap:.4f}")
        print(f"  test: AUC {test_auc:.4f}  AP {test_ap:.4f}")

    return {"val_auc": val_auc, "val_ap": val_ap,
            "test_auc": test_auc, "test_ap": test_ap,
            "best_epoch": best_epoch, "history": history}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(DATASETS), default="cora")
parser.add_argument("--features", choices=["bow", "custom", "concat", "degree"],
                    default="bow")
parser.add_argument("--custom-features-path", default=None,
                    help="Defaults to data/hyperbolic_features/"
                         "<dataset>_node_metrics.csv.")
parser.add_argument("--custom-num-features", type=int, default=None,
                    help="Keep only the first m columns of the custom feature file.")
parser.add_argument("--add-degree", action="store_true",
                    help="Append a log-scaled, standardized degree column.")
parser.add_argument("--standardize-custom", action="store_true",
                    help="Zero-mean / unit-variance each hyperbolicity column "
                         "(applied in every --features mode, so the comparison "
                         "stays symmetric).")
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--out-dim", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--patience", type=int, default=100)
# patience 100 come node_classification.py: la val AUC resta in un plateau
# piatto per qualche decina di epoche prima di decollare, e con patience 30
# l'early stopping la uccideva dentro il plateau (7 seed su 10 con --features
# bow si fermavano all'epoca 31 senza aver imparato niente)
parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                    help="Comma-separated; one run per seed, results averaged.")
parser.add_argument("--curves", nargs="?", const="auto", default=None,
                    help="Salva la storia per epoca (seed, epoch, loss, val "
                         "AUC, val AP) di ogni seed. Da solo sceglie il nome "
                         "in base al run: "
                         "data/curves/linkpred_<dataset>_<features>.csv. "
                         "Con un path scrive li'. Si plotta con plot_curves.py.")
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

custom_path = args.custom_features_path
if custom_path is None and args.features in ("custom", "concat"):
    custom_path = metrics_path(args.dataset)

data, node_map = load_lcc(args.dataset)

# le feature non dipendono dal seed, quindi si costruiscono una volta sola.
# nessun blocco viene riscalato: bow e custom entrano come sono, cosi'
# l'unica differenza fra --features bow e --features concat sono le
# colonne di iperbolicita' in piu'
data.x = build_features(args.features, data.x, data.edge_index, data.num_nodes,
                        custom_path=custom_path,
                        num_custom=args.custom_num_features,
                        standardize_custom=args.standardize_custom)

if args.add_degree:
    # log1p + standardize: Cora/CiteSeer degrees are heavily skewed, and a
    # raw degree column dominates the rest of the feature vector
    deg = torch.log1p(node_degrees(data.edge_index, data.num_nodes))
    deg = (deg - deg.mean()) / (deg.std() + 1e-8)
    data.x = torch.cat([data.x, deg[:, None]], dim=1)

trunc = f" (first {args.custom_num_features} custom dims)" if args.custom_num_features else ""
degree_note = " + degree" if args.add_degree else ""
std_note = " [custom standardizzate]" if args.standardize_custom else ""
print(f"{args.dataset} | features: {args.features}{trunc}{degree_note}{std_note} "
      f"(dim={data.x.shape[1]}) | seeds {seeds}\n")

verbose = len(seeds) == 1
heuristic_names = ["common_neighbors", "jaccard", "adamic_adar"]
runs = {name: [] for name in heuristic_names + ["gcn"]}   # nome -> [(auc, ap), ...]
curve_rows = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # il seed fissa sia lo split degli archi sia l'init del modello
    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.05, num_test=0.10, is_undirected=True,
        add_negative_train_samples=True, neg_sampling_ratio=1.0,
    )(data)

    h = evaluate_heuristics(train_data, val_data, test_data, verbose)
    g = train_gcn(train_data, val_data, test_data, args, device, verbose)

    for name in heuristic_names:
        runs[name].append((h[name]["test_auc"], h[name]["test_ap"]))
    runs["gcn"].append((g["test_auc"], g["test_ap"]))
    curve_rows += [(seed,) + row for row in g["history"]]

    print(f"seed {seed:>3d} | GCN test AUC {g['test_auc']:.4f} | "
          f"test AP {g['test_ap']:.4f} | best epoch {g['best_epoch']:>4d} | "
          f"adamic-adar AUC {h['adamic_adar']['test_auc']:.4f}")

print("\n" + "=" * 58)
print(f"LINK PREDICTION -- {args.dataset}, features={args.features}, "
      f"{len(seeds)} seed(s)")
print("=" * 58)
print(f"{'Method':<22} {'Test AUC':>16} {'Test AP':>16}")
for name in heuristic_names + ["gcn"]:
    auc = np.array([r[0] for r in runs[name]])
    ap = np.array([r[1] for r in runs[name]])
    label = "GCN (dot-product)" if name == "gcn" else name
    print(f"{label:<22} {auc.mean():>9.4f} +/- {auc.std():.4f} "
          f"{ap.mean():>9.4f} +/- {ap.std():.4f}")

if args.curves:
    if args.curves == "auto":
        args.curves = curves_path("linkpred", args.dataset, feature_tag(args))
    write_curves(args.curves,
                 ["seed", "epoch", "loss", "val_auc", "val_ap"], curve_rows)
