"""
Link Prediction Benchmark Baseline — Cora Dataset
==================================================
Baselines implemented:
  1. Common Neighbors (heuristic)
  2. Jaccard Coefficient (heuristic)
  3. Adamic-Adar Index (heuristic)
  4. GCN-based VAE / dot-product decoder (GNN baseline)

Metrics: ROC-AUC, Average Precision (AP)

Requirements:
  pip install torch torch-geometric scikit-learn networkx

Expects cora_node_metrics.csv (produced by generate_hyperbolic_features.py
--dataset cora) to sit next to this script.
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import pandas as pd
import re
import torch

import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx, to_undirected, degree

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & SPLITTING
# ════════════════════════════════════════════════════════════════════════════



def add_degree_feature(data, num_nodes, log_scale=True, normalize=True):
    """Compute node degree from data.edge_index and append as a feature column."""
    deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)

    if log_scale:
        deg = torch.log1p(deg)  # helps with Cora's skewed degree distribution

    if normalize:
        deg = (deg - deg.mean()) / (deg.std() + 1e-8)

    data.x = torch.cat([data.x, deg.view(-1, 1)], dim=1)
    return data

def load_custom_features_from_csv(csv_path, num_nodes=None):
    df = pd.read_csv(csv_path)

    # Make sure nodes are sorted by node_id (must match Cora's node ordering!)
    df = df.sort_values("node_id").reset_index(drop=True)

    def parse_array_str(s):
        # Strip brackets, then split on any whitespace (handles multi-line strings too)
        s = s.strip().lstrip("[").rstrip("]")
        nums = re.split(r"\s+", s.strip())
        return [float(x) for x in nums]

    parsed = df["metric_result"].apply(parse_array_str)

    # Sanity check: all rows should have the same length
    lengths = parsed.apply(len)
    assert lengths.nunique() == 1, f"Inconsistent feature lengths found: {lengths.unique()}"

    features = np.stack(parsed.values)  # shape (num_nodes, k)

    if num_nodes is not None:
        assert features.shape[0] == num_nodes, \
            f"Got {features.shape[0]} rows, expected {num_nodes} nodes"

    return torch.tensor(features, dtype=torch.float32)

def load_cora(feature_mode="original", custom_features_path=None):
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    orig_num_nodes = data.num_nodes  # <-- capture BEFORE modifying data.x

    # --- Extract largest connected component ---
    G = build_nx_graph(data.edge_index, orig_num_nodes)
    largest_cc = max(nx.connected_components(G), key=len)
    keep_nodes = sorted(largest_cc)
    keep_idx = torch.tensor(keep_nodes, dtype=torch.long)

    # Subset node-level tensors (original Cora features/labels)
    data.x = data.x[keep_idx]
    data.y = data.y[keep_idx]

    # Subset + relabel edges
    src, dst = data.edge_index
    mask = torch.isin(src, keep_idx) & torch.isin(dst, keep_idx)
    src, dst = src[mask], dst[mask]

    relabel = torch.full((orig_num_nodes,), -1, dtype=torch.long)  # <-- use orig_num_nodes
    relabel[keep_idx] = torch.arange(len(keep_idx))
    data.edge_index = torch.stack([relabel[src], relabel[dst]], dim=0)

    data.num_nodes = len(keep_nodes)  # now safe to set explicitly

    # --- Load custom features if needed (already in post-LCC order) ---
    custom_features = None
    if feature_mode in ("custom", "concat"):
        assert custom_features_path is not None, \
            f"custom_features_path required for feature_mode='{feature_mode}'"
        custom_features = load_custom_features_from_csv(
            custom_features_path, num_nodes=data.num_nodes
        )

    if feature_mode == "original":
        pass

    elif feature_mode == "custom":
        data.x = custom_features

    elif feature_mode == "concat":
        # Normalize BOTH blocks to comparable scale
        orig_x = data.x
        orig_x = (orig_x - orig_x.mean(dim=0, keepdim=True)) / (orig_x.std(dim=0, keepdim=True) + 1e-8)
        data.x = torch.cat([orig_x, custom_features], dim=1)

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    transform = RandomLinkSplit(
        num_val=0.05, num_test=0.10, is_undirected=True,
        add_negative_train_samples=True, neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)

    train_data = add_degree_feature(train_data, data.num_nodes)
    val_data   = add_degree_feature(val_data, data.num_nodes)
    test_data  = add_degree_feature(test_data, data.num_nodes)

    def pos_neg_counts(split):
        labels = split.edge_label
        return int((labels == 1).sum()), int((labels == 0).sum())

    train_pos, train_neg = pos_neg_counts(train_data)
    val_pos, val_neg     = pos_neg_counts(val_data)
    test_pos, test_neg   = pos_neg_counts(test_data)

    print(f"=== Dataset: Cora (feature_mode='{feature_mode}', LCC only) ===")
    print(f"  Nodes (LCC)        : {data.num_nodes}")
    print(f"  Features/node      : {data.num_node_features}")
    print(f"  Classes            : {dataset.num_classes}")
    print(f"  Unique undirected edges (total) : {data.edge_index.shape[1] // 2}")
    print(f"  Train edges (msg-passing graph)  : {train_data.edge_index.shape[1] // 2}")
    print(f"  Train labels  : {train_pos} pos + {train_neg} neg = {train_pos + train_neg}")
    print(f"  Val   labels  : {val_pos} pos + {val_neg} neg = {val_pos + val_neg}")
    print(f"  Test  labels  : {test_pos} pos + {test_neg} neg = {test_pos + test_neg}\n")

    return dataset, train_data, val_data, test_data


# ════════════════════════════════════════════════════════════════════════════
# 2. HEURISTIC BASELINES
# ════════════════════════════════════════════════════════════════════════════

def build_nx_graph(edge_index, num_nodes):
    """Build an undirected NetworkX graph from a PyG edge_index."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


def score_heuristic(G, edge_label_index, heuristic: str):
    """
    Score node pairs using a graph heuristic.
    Returns (scores, labels) as numpy arrays.
    """
    src = edge_label_index[0].tolist()
    dst = edge_label_index[1].tolist()
    pairs = list(zip(src, dst))

    if heuristic == "common_neighbors":
        raw = nx.common_neighbor_centrality(G, pairs, alpha=1.0)
        scores = np.array([s for _, _, s in raw])

    elif heuristic == "jaccard":
        raw = nx.jaccard_coefficient(G, pairs)
        scores = np.array([s for _, _, s in raw])

    elif heuristic == "adamic_adar":
        raw = nx.adamic_adar_index(G, pairs)
        scores = np.array([s for _, _, s in raw])

    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    return scores


def evaluate_heuristic(train_data, val_data, test_data):
    """Evaluate all three heuristics on val and test splits."""
    G = build_nx_graph(train_data.edge_index, num_nodes=train_data.num_nodes)

    heuristics = ["common_neighbors", "jaccard", "adamic_adar"]
    print("=== Heuristic Baselines ===")
    print(f"{'Method':<22} {'Val AUC':>8} {'Val AP':>8} {'Test AUC':>9} {'Test AP':>8}")
    print("-" * 60)

    results = {}
    for h in heuristics:
        val_scores  = score_heuristic(G, val_data.edge_label_index,  h)
        test_scores = score_heuristic(G, test_data.edge_label_index, h)

        val_labels  = val_data.edge_label.numpy()
        test_labels = test_data.edge_label.numpy()

        val_auc  = roc_auc_score(val_labels,  val_scores)
        val_ap   = average_precision_score(val_labels,  val_scores)
        test_auc = roc_auc_score(test_labels, test_scores)
        test_ap  = average_precision_score(test_labels, test_scores)

        results[h] = {"val_auc": val_auc, "val_ap": val_ap,
                      "test_auc": test_auc, "test_ap": test_ap}

        print(f"{h:<22} {val_auc:>8.4f} {val_ap:>8.4f} {test_auc:>9.4f} {test_ap:>8.4f}")

    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
# 3. GCN-BASED BASELINE
# ════════════════════════════════════════════════════════════════════════════

class GCNConv(nn.Module):
    """Minimal GCN layer (without PyG dependency for clarity).

    Implements: H' = D̂^{-1/2} Â D̂^{-1/2} H W
    where Â = A + I (self-loops added) and D̂ is the degree of Â.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, num_nodes):
        device = x.device

        # Add self-loops: Â = A + I
        self_loops = torch.arange(num_nodes, device=device)
        self_loop_index = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)

        # Symmetric normalisation: D̂^{-1/2} Â D̂^{-1/2}
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        agg  = torch.zeros(num_nodes, x.size(1), device=device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)),
                         x[row] * norm.unsqueeze(1))
        return self.lin(agg)


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, num_nodes):
        x = self.conv1(x, edge_index, num_nodes)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, num_nodes)
        return x


class DotProductDecoder(nn.Module):
    """Score an edge (u,v) as σ(z_u · z_v)."""
    def forward(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


class LinkPredGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, dropout=0.5):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.decoder = DotProductDecoder()

    def encode(self, x, edge_index, num_nodes):
        return self.encoder(x, edge_index, num_nodes)

    def decode(self, z, edge_label_index):
        return self.decoder(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index, num_nodes):
        z = self.encode(x, edge_index, num_nodes)
        return self.decode(z, edge_label_index)


# ── Training helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_gcn(model, data, split="val"):
    model.eval()
    z = model.encode(
        data.x.to(DEVICE),
        data.edge_index.to(DEVICE),
        data.num_nodes,
    )
    logits = model.decode(z, data.edge_label_index.to(DEVICE))
    scores = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.numpy()
    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    return auc, ap


def train_gcn(train_data, val_data, test_data,
              hidden=64, out_dim=32, dropout=0.5,
              lr=0.01, weight_decay=5e-4, epochs=200, patience=20):

    model = LinkPredGCN(
        in_channels=train_data.num_node_features,
        hidden_channels=hidden,
        out_channels=out_dim,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc, best_epoch = 0.0, 0
    patience_counter = 0
    best_state = None

    print("=== GCN Baseline (training) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(
            train_data.x.to(DEVICE),
            train_data.edge_index.to(DEVICE),
            train_data.edge_label_index.to(DEVICE),
            train_data.num_nodes,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, train_data.edge_label.float().to(DEVICE)
        )
        loss.backward()
        optimizer.step()

        val_auc, val_ap = evaluate_gcn(model, val_data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f} "
                  f"| Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    # Load best checkpoint
    model.load_state_dict(best_state)
    val_auc,  val_ap  = evaluate_gcn(model, val_data)
    test_auc, test_ap = evaluate_gcn(model, test_data)

    print(f"\n  Best epoch {best_epoch}: Val AUC={val_auc:.4f}  Val AP={val_ap:.4f}")
    print(f"  Test                : Test AUC={test_auc:.4f}  Test AP={test_ap:.4f}\n")

    return {"val_auc": val_auc, "val_ap": val_ap,
            "test_auc": test_auc, "test_ap": test_ap}


# ════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_summary(heuristic_results, gcn_results):
    print("=" * 60)
    print("BENCHMARK SUMMARY — Link Prediction on Cora")
    print("=" * 60)
    print(f"{'Method':<22} {'Test AUC':>10} {'Test AP':>10}")
    print("-" * 44)
    for name, r in heuristic_results.items():
        print(f"{name:<22} {r['test_auc']:>10.4f} {r['test_ap']:>10.4f}")
    print(f"{'GCN (dot-product)':<22} {gcn_results['test_auc']:>10.4f} {gcn_results['test_ap']:>10.4f}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load data
    cora_features_path = os.path.join(os.path.dirname(__file__), "cora_node_metrics.csv")
    dataset, train_data, val_data, test_data = load_cora(
        feature_mode='custom',
        custom_features_path=cora_features_path)

    # Heuristic baselines
    heuristic_results = evaluate_heuristic(train_data, val_data, test_data)

    # GCN baseline
    gcn_results = train_gcn(
        train_data, val_data, test_data,
        hidden=64,
        out_dim=32,
        dropout=0.5,
        lr=0.0001,
        weight_decay=5e-4,
        epochs=300,
        patience=30,
    )

    # Final summary
    print_summary(heuristic_results, gcn_results)
