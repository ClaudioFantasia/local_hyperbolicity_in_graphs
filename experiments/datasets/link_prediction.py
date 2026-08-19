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

--custom-features-path defaults to <dataset>_node_metrics.csv next to this
script. Log-scaled, standardized node degree is appended to the GCN's
features unless --no-degree is passed (Cora's degree distribution is very
skewed, and the raw column swamps everything else without it).

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
from torch_geometric.transforms import RandomLinkSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common import DATASETS, build_features, load_lcc, metrics_path


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


def evaluate_heuristics(train_data, val_data, test_data):
    # heuristics only get to see the message-passing (training) edges
    G = nx.Graph()
    G.add_nodes_from(range(train_data.num_nodes))
    G.add_edges_from(train_data.edge_index.t().tolist())

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
        print(f"{h:<20} {r['val_auc']:>8.4f} {r['val_ap']:>8.4f} "
              f"{r['test_auc']:>9.4f} {r['test_ap']:>8.4f}")
    print()
    return results


# ----------------------------------------------------------------------
# GCN encoder + dot-product decoder
# ----------------------------------------------------------------------

class GCNLayer(nn.Module):
    """H' = D^-1/2 (A + I) D^-1/2 H W, written out rather than imported so
    the whole model is readable in one file."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, num_nodes):
        loops = torch.arange(num_nodes, device=x.device)
        edge_index = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)

        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)),
                         x[row] * norm.unsqueeze(1))
        return self.lin(agg)


class LinkPredGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, x, edge_index, num_nodes):
        x = F.relu(self.conv1(x, edge_index, num_nodes))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, num_nodes)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


@torch.no_grad()
def evaluate_gcn(model, data, device):
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device), data.num_nodes)
    scores = torch.sigmoid(model.decode(z, data.edge_label_index.to(device))).cpu().numpy()
    labels = data.edge_label.numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_gcn(train_data, val_data, test_data, args, device):
    model = LinkPredGCN(train_data.x.shape[1], args.hidden_dim, args.out_dim,
                        args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_auc, best_epoch, best_state, no_improve = 0.0, 0, None, 0

    print("=== GCN baseline (training) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x.to(device), train_data.edge_index.to(device),
                         train_data.num_nodes)
        logits = model.decode(z, train_data.edge_label_index.to(device))
        loss = F.binary_cross_entropy_with_logits(
            logits, train_data.edge_label.float().to(device))
        loss.backward()
        optimizer.step()

        val_auc, val_ap = evaluate_gcn(model, val_data, device)
        if val_auc > best_val_auc:
            best_val_auc, best_epoch = val_auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3} | loss {loss.item():.4f} | "
                  f"val AUC {val_auc:.4f} | val AP {val_ap:.4f}")

        if no_improve >= args.patience:
            print(f"  early stopping at epoch {epoch} (best: {best_epoch})")
            break

    model.load_state_dict(best_state)
    val_auc, val_ap = evaluate_gcn(model, val_data, device)
    test_auc, test_ap = evaluate_gcn(model, test_data, device)
    print(f"\n  best epoch {best_epoch}: val AUC {val_auc:.4f}  val AP {val_ap:.4f}")
    print(f"  test: AUC {test_auc:.4f}  AP {test_ap:.4f}\n")

    return {"val_auc": val_auc, "val_ap": val_ap,
            "test_auc": test_auc, "test_ap": test_ap}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(DATASETS), default="cora")
parser.add_argument("--features", choices=["bow", "custom", "concat"], default="bow")
parser.add_argument("--custom-features-path", default=None,
                    help="Defaults to <dataset>_node_metrics.csv next to this script.")
parser.add_argument("--custom-num-features", type=int, default=None)
parser.add_argument("--no-degree", action="store_true",
                    help="Skip the log-scaled, standardized degree column.")
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--out-dim", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

custom_path = args.custom_features_path
if custom_path is None and args.features != "bow":
    custom_path = metrics_path(args.dataset)

data, node_map = load_lcc(args.dataset)

# standardize the bag-of-words block before concatenating, so the two
# blocks live on comparable scales
bow_x = data.x
if args.features == "concat":
    bow_x = (bow_x - bow_x.mean(0, keepdim=True)) / (bow_x.std(0, keepdim=True) + 1e-8)

data.x = build_features(args.features, bow_x, data.edge_index, data.num_nodes,
                        custom_path=custom_path,
                        num_custom=args.custom_num_features)

if not args.no_degree:
    # log1p + standardize: Cora/CiteSeer degrees are heavily skewed, and a
    # raw degree column dominates the rest of the feature vector
    deg = torch.zeros(data.num_nodes)
    deg.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
    deg = torch.log1p(deg)
    deg = (deg - deg.mean()) / (deg.std() + 1e-8)
    data.x = torch.cat([data.x, deg[:, None]], dim=1)

train_data, val_data, test_data = RandomLinkSplit(
    num_val=0.05, num_test=0.10, is_undirected=True,
    add_negative_train_samples=True, neg_sampling_ratio=1.0,
)(data)

print(f"features: {args.features} (dim={data.x.shape[1]}) | "
      f"train/val/test edge labels: {train_data.edge_label.numel()} / "
      f"{val_data.edge_label.numel()} / {test_data.edge_label.numel()}\n")

heuristic_results = evaluate_heuristics(train_data, val_data, test_data)
gcn_results = train_gcn(train_data, val_data, test_data, args, device)

print("=" * 52)
print(f"LINK PREDICTION -- {args.dataset}, features={args.features}")
print("=" * 52)
print(f"{'Method':<22} {'Test AUC':>10} {'Test AP':>10}")
for name, r in heuristic_results.items():
    print(f"{name:<22} {r['test_auc']:>10.4f} {r['test_ap']:>10.4f}")
print(f"{'GCN (dot-product)':<22} {gcn_results['test_auc']:>10.4f} "
      f"{gcn_results['test_ap']:>10.4f}")
