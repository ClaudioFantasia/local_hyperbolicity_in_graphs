"""
GIN graph classification on a TU benchmark (MUTAG by default), used to
test whether the local hyperbolicity profile adds anything on top of the
dataset's own node features.

Feature modes (--features):
    bow            the dataset's own node features (one-hot atom type on
                   MUTAG)
    custom         the per-node hyperbolicity profile from
                   data/hyperbolic_features/<dataset>_node_metrics.csv (or
                   any file passed with --custom-features-path)
    concat         bow ++ custom
    degree         node degree, as the only column
    degree-onehot  node degree one-hot over the dataset's degree range
The profile is concatenated to the node features, so it goes through
message passing like any other feature and reaches the classifier through
the final global pooling. --add-degree appends node degree on top.

The two degree modes are the control this experiment needs. MUTAG is
largely solvable from structure alone, so `custom` landing near `bow`
proves nothing by itself: the profile may simply be acting as a generic
structural summary. The degree runs measure how far plain structure
already gets, and only the margin above them is evidence that what the
profile carries is specifically about hyperbolicity.

They are two modes because the encoding matters and not only the
information. `degree` hands the GIN a single scalar column;
`degree-onehot` widens it to one column per distinct degree (5 on MUTAG,
whose degrees run 1-4), which the first GINConv can separate linearly.
The published "structure only" TU baselines use the one-hot form, so that
is the fair comparison; if the scalar lands well below it, the gap is
about encoding rather than about information.

On the TU datasets generate_features.py scores the whole graph (k is set
past the diameter) and sweeps T_geom over 50 values, so the profile is a
50-column multi-scale encoding of the node's position: column j is G*(v)
read at scale T_geom = 0.05*(j+1), from nearly-local to nearly-global.
All 50 columns are used by default. They are highly redundant -- on MUTAG
the first two principal components already carry 99.1% of the variance --
so --custom-num-features m, which keeps the first m (the most local
scales), is worth trying as an ablation rather than as the default.

Protocol
--------
The TU benchmarks ship no standard split, so results are reported the way
the literature does: stratified 10-fold cross validation, mean +/- std of
the test accuracy across the folds, repeated over --seeds. A seed fixes
the folds, the inner validation split and the model init, so running the
same --seeds across feature modes is a paired comparison on identical
folds.

Inside each fold 10% of the training portion is held out for validation
and the reported test accuracy is the one at the best validation epoch
(--model-selection val). That is the protocol of Errica et al. 2020, "A
Fair Comparison of Graph Neural Networks for Graph Classification".
--model-selection test reports the best *test* accuracy over epochs
instead. It is not a protocol, it is a function of the epoch budget --
measured on MUTAG it reaches 0.968 at 500 epochs -- so use it only to
reproduce a published number, never to decide whether the hyperbolicity
features help.

Training runs until the validation accuracy stops improving for
--patience epochs, with the lr halved every --lr-decay-step epochs. Both
are there for the same reason: with the earlier fixed 200-epoch budget
and no decay, `bow` was converged well before the deadline (median best
epoch 20) while `concat` was not (median 165, still improving at 400), so
the comparison measured convergence speed rather than feature quality --
the sign of concat - bow flipped from -4.3 to +3.2 points between a
50- and a 400-epoch budget. A fixed budget is not neutral between feature
modes of different width.

MUTAG is 188 graphs, so a fold is ~19 graphs and the std across folds is
large (6-11 accuracy points). That means the honest error bar on a
difference between two rows is the *paired* per-fold difference, not the
std across seed means, which only measures seed stability and is 5-10x
smaller. Note also that MUTAG is essentially solved by graph size and
degree histogram alone: a plain logistic regression on 6 such numbers
reaches 0.880 here (graph_classification_linear.py), above every GNN row.
Prefer NCI1 for any conclusion that has to hold.

Usage
-----
    python experiments/datasets/graph_classification.py --features bow
    python experiments/datasets/graph_classification.py --features concat
    python experiments/datasets/graph_classification.py --features degree-onehot
    python experiments/datasets/graph_classification.py --dataset mutag
        --features concat --standardize-custom

--custom-features-path defaults to
data/hyperbolic_features/<dataset>_node_metrics.csv (what
`generate_features.py --dataset mutag` writes).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GINConv, global_add_pool, global_max_pool,
                                global_mean_pool)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common import (TU_DATASETS, build_features, check_meta, curves_path,
                    feature_tag, load_graph_node_features, load_tu,
                    metrics_path, node_degrees, write_curves)

POOLINGS = {"add": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


class GIN(torch.nn.Module):
    """num_layers GINConv layers, each wrapping a 2-layer MLP with batch
    norm, then global pooling and a linear classifier."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5,
                 pooling="add"):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = torch.nn.ModuleList([
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(dims[i], hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim)))
            for i in range(num_layers)
        ])
        self.pool = POOLINGS[pooling]
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
    return correct / len(loader.dataset)


def run_fold(model, train_loader, val_loader, test_loader, args, device):
    """Train until the selection metric stops improving (or the epoch
    budget runs out), tracking that metric.
    Returns (val acc, test acc, epoch) at the selected epoch, plus the
    per-epoch history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # halving the lr every 50 epochs is what the GIN paper does on the TU
    # benchmarks. Without it the constant 1e-2 never settles: measured on
    # MUTAG the train loss still bounces between 0.26 and 0.39 at epoch 300.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    best_selected, best_val, best_test, best_epoch = -1.0, 0.0, 0.0, 0
    epochs_no_improve = 0
    # loss (media sui batch) e le due accuratezze sono gia' calcolate a ogni
    # epoca qui sotto: registrarle non costa niente in piu'
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        scheduler.step()

        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        history.append((epoch, epoch_loss / len(train_loader.dataset),
                        val_acc, test_acc))

        selected = val_acc if args.model_selection == "val" else test_acc
        if selected > best_selected:
            best_selected, best_val, best_test = selected, val_acc, test_acc
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # a fixed epoch budget is not neutral between feature modes: the
        # wider input of `concat` converges much more slowly than `bow`, so
        # stopping both at 200 compares a converged model against a
        # truncated one (see graph_classification_mutag.md Sec. 6)
        if args.patience and epochs_no_improve > args.patience:
            break

    return best_val, best_test, best_epoch, history


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(TU_DATASETS), default="mutag")
parser.add_argument("--features",
                    choices=["bow", "custom", "concat", "degree", "degree-onehot"],
                    default="bow")
parser.add_argument("--custom-features-path", default=None,
                    help="Defaults to data/hyperbolic_features/"
                         "<dataset>_node_metrics.csv.")
parser.add_argument("--custom-num-features", type=int, default=None,
                    help="Keep only the first m columns of the custom feature file.")
parser.add_argument("--add-degree", action="store_true",
                    help="Append node degree as one extra feature column.")
parser.add_argument("--standardize-custom", action="store_true",
                    help="Zero-mean / unit-variance each custom feature column, "
                         "over every node of every graph. The hyperbolicity "
                         "columns carry a large constant offset that holds no "
                         "per-node information and only puts the block off the "
                         "scale of the one-hot node features.")
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--pooling", choices=sorted(POOLINGS), default="add",
                    help="Readout over the nodes of a graph. Sum pooling is what "
                         "the GIN paper uses on the TU benchmarks.")
parser.add_argument("--epochs", type=int, default=500,
                    help="Max epochs; early stopping normally fires first.")
parser.add_argument("--patience", type=int, default=100,
                    help="Stop after this many epochs with no improvement of "
                         "the selection metric. 0 disables early stopping and "
                         "trains for the full --epochs budget.")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--lr-decay", type=float, default=0.5,
                    help="Multiplies the lr every --lr-decay-step epochs.")
parser.add_argument("--lr-decay-step", type=int, default=50)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--folds", type=int, default=10)
parser.add_argument("--model-selection", choices=["val", "test"], default="val",
                    help="Which metric picks the reported epoch; see the module "
                         "docstring. 'test' is the optimistic protocol behind the "
                         "published MUTAG numbers.")
parser.add_argument("--seeds", default="0,1,2",
                    help="Comma-separated; one full 10-fold CV per seed.")
parser.add_argument("--curves", nargs="?", const="auto", default=None,
                    help="Salva la storia per epoca (seed, fold, epoch, loss, "
                         "val/test acc) di ogni fold. Da solo sceglie il nome "
                         "in base al run: "
                         "data/curves/graphclass_<dataset>_<pooling>_<features>.csv. "
                         "Con un path scrive li'. Si plotta con plot_curves.py.")
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graphs = load_tu(args.dataset)
sizes = [d.num_nodes for d in graphs]
labels = np.array([int(d.y) for d in graphs])
out_dim = int(labels.max()) + 1

uses_custom = args.features in ("custom", "concat")

custom_path = args.custom_features_path
if custom_path is None and uses_custom:
    custom_path = metrics_path(args.dataset)

# The custom block is truncated and standardized over the *whole* dataset
# and only then split back per graph: normalising inside a single molecule
# would erase exactly the differences between molecules that the graph-level
# readout is supposed to see.
custom_blocks = None
if uses_custom:
    check_meta(custom_path, sum(sizes),
               sum(d.edge_index.shape[1] // 2 for d in graphs))
    custom_all = torch.cat(load_graph_node_features(custom_path, sizes))
    if args.custom_num_features is not None:
        assert args.custom_num_features <= custom_all.shape[1], \
            f"--custom-num-features {args.custom_num_features} > {custom_all.shape[1]} columns"
        custom_all = custom_all[:, :args.custom_num_features]
    if args.standardize_custom:
        custom_all = (custom_all - custom_all.mean(0)) / (custom_all.std(0) + 1e-8)
    custom_blocks = list(torch.split(custom_all, sizes))

# feature construction has no randomness in it, so it happens once and is
# reused unchanged across every fold and every seed
# degree-onehot is the one mode build_features cannot assemble by itself:
# the width of the one-hot is the largest degree in the *dataset*, while
# build_features only ever sees one graph at a time
max_degree = int(max(node_degrees(d.edge_index, d.num_nodes).max() for d in graphs))

dataset = []
for i, d in enumerate(graphs):
    if args.features == "degree-onehot":
        degrees = node_degrees(d.edge_index, d.num_nodes).long()
        x = F.one_hot(degrees, num_classes=max_degree + 1).float()
        if args.add_degree:
            x = torch.cat([x, degrees[:, None].float()], dim=1)
    else:
        x = build_features(args.features, d.x, d.edge_index, d.num_nodes,
                           custom_x=None if custom_blocks is None else custom_blocks[i],
                           add_degree=args.add_degree)
    dataset.append(Data(x=x, edge_index=d.edge_index, y=d.y))

trunc = f" (first {args.custom_num_features} custom dims)" if args.custom_num_features else ""
degree_note = " + degree" if args.add_degree else ""
std_note = " [custom standardizzate]" if args.standardize_custom else ""
print(f"\n{args.dataset} | features: {args.features}{trunc}{degree_note}{std_note} "
      f"(dim={dataset[0].x.shape[1]}) | {len(dataset)} graphs | "
      f"{args.folds}-fold CV, model selection on {args.model_selection} | seeds {seeds}")

seed_means = []
curve_rows = []
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    fold_scores, fold_epochs = [], []

    for fold, (train_idx, test_idx) in enumerate(folds.split(labels, labels)):
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1,
                                              stratify=labels[train_idx],
                                              random_state=seed)

        def loader(idx, shuffle=False):
            return DataLoader([dataset[i] for i in idx],
                              batch_size=args.batch_size, shuffle=shuffle)

        model = GIN(dataset[0].x.shape[1], args.hidden_dim, out_dim,
                    num_layers=args.num_layers, dropout=args.dropout,
                    pooling=args.pooling)

        val_acc, test_acc, best_epoch, history = run_fold(
            model, loader(train_idx, shuffle=True), loader(val_idx),
            loader(test_idx), args, device)
        curve_rows += [(seed, fold) + row for row in history]
        fold_scores.append(test_acc)
        fold_epochs.append(best_epoch)
        print(f"seed {seed} fold {fold:>2d} | val acc {val_acc:.4f} | "
              f"test acc {test_acc:.4f} | selected epoch {best_epoch}")

    fold_scores = np.array(fold_scores)
    seed_means.append(fold_scores.mean())
    print(f"seed {seed}: test acc {fold_scores.mean():.4f} +/- "
          f"{fold_scores.std():.4f} across {args.folds} folds "
          f"(selected epoch: median {int(np.median(fold_epochs))}, "
          f"max {max(fold_epochs)})")

seed_means = np.array(seed_means)
print(f"\nAcross {len(seeds)} seed(s) of {args.folds}-fold CV:")
print(f"  test acc: {seed_means.mean():.4f} +/- {seed_means.std():.4f}")

if args.curves:
    if args.curves == "auto":
        args.curves = curves_path("graphclass", args.dataset, args.pooling,
                                  feature_tag(args))
    write_curves(args.curves,
                 ["seed", "fold", "epoch", "loss", "val_acc", "test_acc"],
                 curve_rows)
