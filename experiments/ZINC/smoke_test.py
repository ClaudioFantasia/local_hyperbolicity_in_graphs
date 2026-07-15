"""
Smoke test: validates ZincGNN + the training loop using synthetic graphs
that mimic ZINC's exact schema (x: atom-type indices, edge_attr: bond-type
indices, y: scalar regression target). This does not require downloading
the real ZINC dataset, so it also works in network-restricted environments.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from zinc_gnn import ZincGNN, train_one_epoch, evaluate

torch.manual_seed(0)

NUM_ATOM_TYPES = 28  # matches ZINC-12k
NUM_BOND_TYPES = 4


def random_molecule_graph():
    num_nodes = torch.randint(8, 30, (1,)).item()
    x = torch.randint(0, NUM_ATOM_TYPES, (num_nodes,))

    # random spanning-tree-ish + a few extra edges, undirected
    edges = []
    for i in range(1, num_nodes):
        j = torch.randint(0, i, (1,)).item()
        edges.append((i, j))
        edges.append((j, i))
    num_extra = torch.randint(0, 5, (1,)).item()
    for _ in range(num_extra):
        a, b = torch.randint(0, num_nodes, (2,)).tolist()
        if a != b:
            edges.append((a, b))
            edges.append((b, a))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.randint(0, NUM_BOND_TYPES, (edge_index.size(1),))
    y = torch.randn(1) * 2  # fake penalized-logP-like target

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def make_dataset(n):
    return [random_molecule_graph() for _ in range(n)]


def main():
    device = torch.device("cpu")

    train_ds = make_dataset(64)
    val_ds = make_dataset(16)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = ZincGNN(
        num_atom_types=NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        hidden_dim=32,
        num_layers=3,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Running 5 epochs on synthetic data to validate shapes/gradients...")
    losses = []
    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device)
        losses.append(train_loss)
        print(f"epoch {epoch}: train MAE {train_loss:.4f} | val MAE {val_mae:.4f}")

    assert losses[-1] < losses[0], "Loss did not decrease -- something is wrong with the training loop."
    print("\nSMOKE TEST PASSED: forward/backward pass, batching, pooling, and training loop all work correctly.")


if __name__ == "__main__":
    main()