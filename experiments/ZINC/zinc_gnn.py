"""
Complete GNN pipeline for the ZINC molecular property regression benchmark.

Task: predict the constrained solubility (penalized logP) of small drug-like
molecules from their 2D molecular graph (atoms = nodes, bonds = edges).

Pipeline:
    1. Data loading   -> torch_geometric.datasets.ZINC (12k subset)
    2. Model          -> GINE (Graph Isomorphism Network with Edge features)
    3. Training loop  -> Adam + ReduceLROnPlateau, L1 (MAE) loss
    4. Evaluation     -> MAE on val/test splits

Usage:
    python zinc_gnn.py --epochs 100 --hidden_dim 128 --num_layers 4

Requirements:
    pip install torch torch_geometric

Note: torch_geometric downloads ZINC from Dropbox on first run. Make sure
your environment has open internet access (this is NOT reachable from
network-restricted sandboxes).
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def get_dataloaders(root: str, subset: bool, batch_size: int, num_workers: int = 2):
    """Load the ZINC train/val/test splits and wrap them in DataLoaders.

    subset=True  -> ZINC-12k  (12,000 molecules, standard benchmark size)
    subset=False -> ZINC-full (~250,000 molecules)
    """
    train_ds = ZINC(root=root, subset=subset, split="train")
    val_ds = ZINC(root=root, subset=subset, split="val")
    test_ds = ZINC(root=root, subset=subset, split="test")

    print(f"Train / Val / Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(f"Example graph: {train_ds[0]}")
    print(f"Num atom types (node feature vocab): {train_ds[0].x.max().item() + 1} (approx, see full scan below)")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds


# --------------------------------------------------------------------------
# 2. Model: GINE (GIN with edge features), standard choice for ZINC
# --------------------------------------------------------------------------
class GINEBlock(nn.Module):
    """One GINE convolution + BatchNorm + ReLU + residual connection."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        # GINEConv projects edge_attr to hidden_dim internally via edge_dim
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return x + h  # residual connection


class ZincGNN(nn.Module):
    """
    Atom / bond embeddings -> stack of GINE blocks -> sum pooling -> MLP head.

    ZINC node features (x) and edge features (edge_attr) are integer category
    indices (atom type / bond type), so we embed them rather than treating
    them as continuous inputs.
    """

    def __init__(
        self,
        num_atom_types: int,
        num_bond_types: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.bond_embedding = nn.Embedding(num_bond_types, hidden_dim)

        self.blocks = nn.ModuleList(
            [GINEBlock(hidden_dim, edge_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_embedding(x.squeeze(-1) if x.dim() > 1 else x)
        edge_attr = self.bond_embedding(edge_attr)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        graph_repr = global_add_pool(x, batch)  # sum pooling over nodes in each graph
        out = self.head(graph_repr)
        return out.squeeze(-1)


# --------------------------------------------------------------------------
# 3. Training / evaluation loops
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.l1_loss(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_examples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        mae = F.l1_loss(pred, data.y, reduction="sum")
        total_mae += mae.item()
        total_examples += data.num_graphs
    return total_mae / total_examples


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        root=args.data_root, subset=args.subset, batch_size=args.batch_size
    )
    
    # ZINC-12k vocab sizes are fixed by the benchmark: 28 atom types, 4 bond types.
    # We scan the actual data to be safe/robust to future dataset versions.
    num_atom_types = int(train_ds.x.max().item()) + 1
    num_bond_types = int(train_ds.edge_attr.max().item()) + 1
    print(f"num_atom_types={num_atom_types}, num_bond_types={num_bond_types}")

    model = ZincGNN(
        num_atom_types=num_atom_types,
        num_bond_types=num_bond_types,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience, min_lr=1e-5
    )

    best_val_mae = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train MAE {train_loss:.4f} | val MAE {val_mae:.4f} "
            f"| lr {lr:.2e} | {time.time() - t0:.1f}s"
            + (" *" if improved else "")
        )

        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            print(f"No val improvement for {epochs_since_improve} epochs, stopping early.")
            break

    # Load best checkpoint and report final test performance
    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader, device)
    print(f"\nBest val MAE: {best_val_mae:.4f}")
    print(f"Test MAE (best-val checkpoint): {test_mae:.4f}")

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(best_state, args.save_path)
        print(f"Saved best model weights to {args.save_path}")

    return model, best_val_mae, test_mae


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GNN pipeline for ZINC molecular regression")
    p.add_argument("--data_root", type=str, default="./data/ZINC")
    p.add_argument("--subset", action="store_true", default=True, help="Use ZINC-12k subset (default). ")
    p.add_argument("--full", dest="subset", action="store_false", help="Use ZINC-full (~250k molecules) instead.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5, help="LR scheduler patience (epochs).")
    p.add_argument("--early_stop_patience", type=int, default=30, help="0 disables early stopping.")
    p.add_argument("--save_path", type=str, default="./checkpoints/zinc_gnn_best.pt")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    run_training(args)