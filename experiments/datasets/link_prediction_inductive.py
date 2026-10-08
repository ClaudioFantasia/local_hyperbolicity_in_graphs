"""
Inductive link prediction on a multi-graph (TU) benchmark: the split is
over *graphs*, not over edges, so the model is tested on molecules it has
never seen.

Same pieces as link_prediction.py -- the GCN encoder + dot-product
decoder, the three feature-free heuristics, the same feature modes -- but
a different evaluation loop:

    per ogni seed:
      1. split dei grafi: 80% train, 10% val, 10% test
      2. dentro ogni grafo, RandomLinkSplit nasconde il 20% degli archi
      3. un solo modello condiviso, addestrato in minibatch su tutti i
         grafi di train insieme; early stopping sulla val AUC dei grafi
         di validazione
      4. i grafi di test non sono mai stati visti: l'encoder li propaga
         sui soli archi visibili e predice quelli nascosti
      5. tutte le predizioni di test finiscono in un unico pool -> una
         AUC e una AP per seed

Si riportano media e deviazione standard fra i seed. Qui un seed cambia
lo split dei grafi, non solo l'inizializzazione, quindi la std e' molto
piu' larga di quella di link_prediction.py: misura anche quali molecole
sono capitate nel test.

Le molecole restano componenti separate dentro ogni batch, quindi il
message passing non le attraversa mai: l'encoder che gira su una molecola
e' esattamente quello di link_prediction.py su un grafo solo, ma i pesi
sono stimati su tutti i grafi di training insieme.

Le euristiche
-------------
Su MUTAG le tre euristiche danno la stessa identica AUC, sotto 0.5. Non
e' un bug: sono tutte funzioni monotone dell'insieme dei vicini comuni, e
in una molecola quel conteggio vale solo 0 o 1, quindi l'ordinamento che
producono e' lo stesso. Sotto 0.5 perche' due atomi legati condividono un
vicino solo dentro un triangolo, e le molecole non ne hanno (gli anelli
sono a 5 e 6 atomi): misurato sui 18 grafi di test del seed 0, zero
triangoli, e i vicini comuni valgono in media 0.00 sulle coppie legate
contro 0.27 su quelle non legate. Sono cioe' un predittore
anti-correlato. Restano nella tabella come riferimento feature-free -- la
riga che dice quanto poco la sola struttura locale basti qui -- non come
una baseline da battere.

Usage
-----
    python experiments/datasets/link_prediction_inductive.py --features bow
    python experiments/datasets/link_prediction_inductive.py --features concat
    python experiments/datasets/link_prediction_inductive.py --features custom
        --standardize-custom

--custom-features-path defaults to
data/hyperbolic_features/<dataset>_node_metrics.csv (what
`generate_features.py --dataset mutag` writes).
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common import (TU_DATASETS, build_features, check_meta, curves_path,
                    feature_tag, load_graph_node_features, load_tu,
                    metrics_path, write_curves)

HEURISTICS = ["common_neighbors", "jaccard", "adamic_adar"]


# ----------------------------------------------------------------------
# Heuristic baselines
# ----------------------------------------------------------------------

def score_heuristic(G, edge_label_index, heuristic):
    """Score each candidate pair with a networkx link-prediction index.
    Copia di link_prediction.py: alpha=1.0 riduce CCPA al conteggio dei
    vicini comuni, l'unica forma definita anche su coppie di nodi che
    stanno in componenti diverse."""
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


def evaluate_heuristics(splits):
    """Le tre euristiche sul pool degli archi nascosti di `splits`.
    Ogni grafo diventa un nx.Graph costruito sui soli archi visibili -- lo
    stesso che vede l'encoder -- e le coppie da classificare sono i suoi
    edge_label_index."""
    results = {}
    for h in HEURISTICS:
        scores, labels = [], []
        for d in splits:
            G = nx.Graph()
            G.add_nodes_from(range(d.num_nodes))
            G.add_edges_from(d.edge_index.t().tolist())
            scores.append(score_heuristic(G, d.edge_label_index, h))
            labels.append(d.edge_label.numpy())
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        results[h] = (roc_auc_score(labels, scores),
                      average_precision_score(labels, scores))
    return results


# ----------------------------------------------------------------------
# GCN encoder + dot-product decoder
# ----------------------------------------------------------------------

class LinkPredGCN(nn.Module):
    """Copia dell'encoder di link_prediction.py: num_layers GCNConv, ReLU
    + dropout dopo tutti tranne l'ultimo (quindi l'uscita dell'ultimo e'
    l'embedding), piu' un decoder a prodotto scalare sulle coppie."""

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
def evaluate_gcn(model, loader, device):
    """AUC e AP sul pool di tutti gli archi da predire del loader.
    L'encoder propaga sui soli edge_index (gli archi visibili): quelli in
    edge_label_index non entrano mai nel message passing."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        scores.append(torch.sigmoid(model.decode(z, batch.edge_label_index)).cpu())
        labels.append(batch.edge_label.cpu())
    scores, labels = torch.cat(scores).numpy(), torch.cat(labels).numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_gcn(train_loader, val_loader, in_dim, args, device, verbose):
    model = LinkPredGCN(in_dim, args.hidden_dim, args.out_dim,
                        args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_auc, best_epoch, best_state, no_improve = 0.0, 0, None, 0
    # loss e metriche di validazione sono gia' calcolate a ogni epoca:
    # registrarle non costa niente. Il test non viene toccato nel ciclo,
    # quindi le curve non contengono una colonna di test
    history = []

    if verbose:
        print("\n=== GCN (training) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_edges = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model.encode(batch.x, batch.edge_index)
            logits = model.decode(z, batch.edge_label_index)
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      batch.edge_label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.edge_label.shape[0]
            total_edges += batch.edge_label.shape[0]

        val_auc, val_ap = evaluate_gcn(model, val_loader, device)
        history.append((epoch, total_loss / total_edges, val_auc, val_ap))

        if val_auc > best_val_auc:
            best_val_auc, best_epoch = val_auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4} | loss {total_loss / total_edges:.4f} | "
                  f"val AUC {val_auc:.4f} | val AP {val_ap:.4f}")

        if no_improve >= args.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best: {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, best_epoch, history


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(TU_DATASETS), default="mutag")
parser.add_argument("--features", choices=["bow", "custom", "concat", "degree"],
                    default="bow")
parser.add_argument("--custom-features-path", default=None,
                    help="Un nome di file nudo lo cerca in "
                         "data/hyperbolic_features/. Default: "
                         "<dataset>_node_metrics.csv.")
parser.add_argument("--custom-num-features", type=int, default=None,
                    help="Keep only the first m columns of the custom feature file.")
parser.add_argument("--add-degree", action="store_true",
                    help="Append node degree as one extra feature column.")
parser.add_argument("--standardize-custom", action="store_true",
                    help="Zero-mean / unit-variance di ogni colonna custom, su "
                         "tutti i nodi di tutti i grafi.")
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--out-dim", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--val-graphs", type=float, default=0.1,
                    help="Frazione dei grafi tenuta per l'early stopping.")
parser.add_argument("--test-graphs", type=float, default=0.1,
                    help="Frazione dei grafi tenuta per il test.")
parser.add_argument("--hidden-edges", type=float, default=0.2,
                    help="Frazione degli archi nascosta dentro ogni grafo. Con "
                         "il 5%% di link_prediction.py meta' delle molecole di "
                         "MUTAG resterebbe senza archi da predire.")
parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                    help="Comma-separated; un seed fissa lo split dei grafi, "
                         "quello degli archi e l'init del modello.")
parser.add_argument("--curves", nargs="?", const="auto", default=None,
                    help="Salva la storia per epoca (seed, epoch, loss, val "
                         "AUC, val AP) di ogni seed. Da solo sceglie il nome "
                         "in base al run: "
                         "data/curves/linkpred-ind_<dataset>_<features>.csv. "
                         "Con un path scrive li'. Si plotta con plot_curves.py.")
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

graphs = load_tu(args.dataset)
sizes = [d.num_nodes for d in graphs]

# le feature non dipendono da nessuno split, quindi si costruiscono una
# volta sola. Il blocco custom viene troncato e standardizzato sull'intero
# dataset e solo dopo risplittato per grafo: normalizzare dentro una
# singola molecola cancellerebbe proprio le differenze fra molecole
custom_blocks = None
if args.features in ("custom", "concat"):
    custom_path = metrics_path(args.dataset, args.custom_features_path)
    check_meta(custom_path, sum(sizes),
               sum(d.edge_index.shape[1] // 2 for d in graphs))
    custom_all = torch.cat(load_graph_node_features(custom_path, sizes))
    if args.custom_num_features is not None:
        custom_all = custom_all[:, :args.custom_num_features]
    if args.standardize_custom:
        custom_all = (custom_all - custom_all.mean(0)) / (custom_all.std(0) + 1e-8)
    custom_blocks = list(torch.split(custom_all, sizes))

dataset = []
for i, d in enumerate(graphs):
    x = build_features(args.features, d.x, d.edge_index, d.num_nodes,
                       custom_x=None if custom_blocks is None else custom_blocks[i],
                       add_degree=args.add_degree)
    dataset.append(Data(x=x, edge_index=d.edge_index))

trunc = f" (first {args.custom_num_features} custom dims)" if args.custom_num_features else ""
degree_note = " + degree" if args.add_degree else ""
std_note = " [custom standardizzate]" if args.standardize_custom else ""
print(f"{args.dataset} | features: {args.features}{trunc}{degree_note}{std_note} "
      f"(dim={dataset[0].x.shape[1]}) | {len(dataset)} grafi | "
      f"{int(args.hidden_edges * 100)}% archi nascosti per grafo | seeds {seeds}\n")

verbose = len(seeds) == 1
runs = {name: [] for name in HEURISTICS + ["gcn"]}   # nome -> [(auc, ap), ...]
curve_rows = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. split dei grafi
    perm = np.random.permutation(len(dataset))
    n_test = int(args.test_graphs * len(dataset))
    n_val = int(args.val_graphs * len(dataset))
    test_ids, val_ids = perm[:n_test], perm[n_test:n_test + n_val]
    train_ids = perm[n_test + n_val:]

    # 2. split degli archi dentro ogni grafo. I negativi li campiona dentro
    #    la stessa molecola, quindi restano coppie plausibili invece che
    #    coppie fra molecole diverse, che sarebbero banali da scartare
    splitter = RandomLinkSplit(num_val=0.0, num_test=args.hidden_edges,
                               is_undirected=True, add_negative_train_samples=True,
                               neg_sampling_ratio=1.0)
    # di un grafo di training tengo il pezzo `train`: message passing e
    # supervisione sugli stessi archi visibili piu' altrettanti negativi,
    # come in link_prediction.py. Di un grafo di val/test tengo il pezzo
    # `test`: message passing sugli archi visibili e, da predire, quelli
    # nascosti piu' altrettanti negativi
    train_splits = [splitter(dataset[i])[0] for i in train_ids]
    val_splits = [splitter(dataset[i])[2] for i in val_ids]
    test_splits = [splitter(dataset[i])[2] for i in test_ids]

    train_loader = DataLoader(train_splits, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_splits, batch_size=args.batch_size)
    test_loader = DataLoader(test_splits, batch_size=args.batch_size)

    # 3. un solo modello condiviso su tutti i grafi di training
    model, best_epoch, history = train_gcn(train_loader, val_loader,
                                           dataset[0].x.shape[1], args, device,
                                           verbose)

    # 4-5. i grafi di test non sono mai stati visti; tutte le loro
    #      predizioni finiscono in un pool solo
    test_auc, test_ap = evaluate_gcn(model, test_loader, device)
    h = evaluate_heuristics(test_splits)

    runs["gcn"].append((test_auc, test_ap))
    for name in HEURISTICS:
        runs[name].append(h[name])
    curve_rows += [(seed,) + row for row in history]

    n_edges = sum(d.edge_label.shape[0] for d in test_splits)
    print(f"seed {seed:>3d} | grafi {len(train_ids)}/{len(val_ids)}/{len(test_ids)} | "
          f"{n_edges} coppie di test | GCN AUC {test_auc:.4f} | AP {test_ap:.4f} | "
          f"adamic-adar AUC {h['adamic_adar'][0]:.4f} | best epoch {best_epoch}")

print("\n" + "=" * 58)
print(f"INDUCTIVE LINK PREDICTION -- {args.dataset}, features={args.features}, "
      f"{len(seeds)} seed(s)")
print("=" * 58)
print(f"{'Method':<22} {'Test AUC':>16} {'Test AP':>16}")
for name in HEURISTICS + ["gcn"]:
    auc = np.array([r[0] for r in runs[name]])
    ap = np.array([r[1] for r in runs[name]])
    label = "GCN (dot-product)" if name == "gcn" else name
    print(f"{label:<22} {auc.mean():>9.4f} +/- {auc.std():.4f} "
          f"{ap.mean():>9.4f} +/- {ap.std():.4f}")

if args.curves:
    if args.curves == "auto":
        args.curves = curves_path("linkpred-ind", args.dataset, feature_tag(args))
    write_curves(args.curves,
                 ["seed", "epoch", "loss", "val_auc", "val_ap"], curve_rows)
