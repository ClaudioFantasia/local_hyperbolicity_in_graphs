"""
Train a GCN to regress the per-node local hyperbolicity produced by
generate_dataset.py, and look at how well it does.

The interesting run is the out-of-distribution one: train on the
hierarchical family, test on the random one. If the model only works
in-distribution, it is picking up cheap local correlates (degree, short
cycles) rather than delta itself -- which is what
local_hyperbolicity_valutazione.md Sec. 6 predicts, since delta is a
global 4-point quantity that message passing provably cannot compute.

    # in-distribution: split one dataset into train/test
    python experiments/learning/train_gnn.py --dataset data/dataset/hierarchical.pt

    # out-of-distribution: train on one family, test on another
    python experiments/learning/train_gnn.py \\
        --dataset data/dataset/hierarchical.pt \\
        --test-dataset data/dataset/random.pt

The mean-predictor MAE printed alongside the model's is the number to
beat: matching it means the model has learned nothing beyond the average
target.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.config import REPO_ROOT


class HyperbolicityGNN(nn.Module):
    """Three GCNConv layers then a linear head, one scalar per node."""

    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        return self.lin(x)


def predict_all(model, dataset):
    """Model predictions and ground truth over every node of every graph."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in dataset:
            preds.append(model(data.x, data.edge_index).flatten())
            targets.append(data.y.flatten())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", default="data/dataset/hierarchical.pt",
                    help="Training dataset written by generate_dataset.py.")
parser.add_argument("--test-dataset", default=None,
                    help="Evaluate on this dataset instead of holding out part "
                         "of --dataset (the out-of-distribution check).")
parser.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of --dataset used for training when "
                         "--test-dataset is not given.")
parser.add_argument("--hidden-dim", type=int, default=32)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save-plot", default=None,
                    help="Defaults to data/figures/gnn_predictions.png")
args = parser.parse_args()

torch.manual_seed(args.seed)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


dataset = torch.load(resolve(args.dataset), weights_only=False)

if args.test_dataset is not None:
    train_set = dataset
    test_set = torch.load(resolve(args.test_dataset), weights_only=False)
    setting = f"OOD: {os.path.basename(args.dataset)} -> {os.path.basename(args.test_dataset)}"
else:
    split = int(args.train_frac * len(dataset))
    train_set, test_set = dataset[:split], dataset[split:]
    setting = f"in-distribution split of {os.path.basename(args.dataset)}"

print(f"{setting}: {len(train_set)} train graphs, {len(test_set)} test graphs")

model = HyperbolicityGNN(train_set[0].num_node_features, args.hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

# full-batch: accumulate gradients over every training graph, then step once
for epoch in range(1, args.epochs + 1):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for data in train_set:
        loss = criterion(model(data.x, data.edge_index), data.y)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch:04d} | train MSE {total_loss / len(train_set):.5f}")

preds, targets = predict_all(model, test_set)
train_targets = torch.cat([d.y for d in train_set]).flatten().numpy()

mae = np.abs(preds - targets).mean()
mse = ((preds - targets) ** 2).mean()
# the baseline to beat: always predict the training set's mean target
mean_mae = np.abs(train_targets.mean() - targets).mean()
r = np.corrcoef(preds, targets)[0, 1]

print(f"\nTest over {targets.size} nodes:")
print(f"  model MSE {mse:.5f} | model MAE {mae:.5f}")
print(f"  mean-predictor MAE {mean_mae:.5f}  <- beat this or nothing was learned")
print(f"  Pearson r(pred, true) {r:.4f}")

save_path = args.save_plot or os.path.join(REPO_ROOT, "data", "figures",
                                           "gnn_predictions.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.figure(figsize=(7, 6))
plt.scatter(targets, preds, alpha=0.4, s=12, label="predictions")
lims = [targets.min(), targets.max()]
plt.plot(lims, lims, "r--", label="ideal")
plt.axhline(train_targets.mean(), color="gray", ls=":", label="train mean")
plt.xlabel("true local hyperbolicity $G^*(v)$")
plt.ylabel("predicted $G^*(v)$")
plt.title(f"{setting}\nMAE {mae:.4f} vs mean-predictor {mean_mae:.4f}, r={r:.3f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(save_path, dpi=150)
print(f"Saved plot to {save_path}")
