"""
Logistic regression su descrittori graph-level, senza nessuna GNN.

Serve a separare due domande che graph_classification.py confonde:

  1. il profilo di iperbolicita' locale distingue le classi del dataset?
  2. una GIN riesce a usarlo, passandolo per message passing + pooling?

Lo script risponde alla (1). Ogni grafo diventa un vettore di statistiche
sui suoi nodi -- media/std/max/min di ciascuna delle colonne T_geom del
profilo -- e sopra ci va una logistic regression. Niente message passing,
niente pooling appreso: se l'iperbolicita' porta segnale utilizzabile,
qui si vede in forma pura.

Set di feature (--features):
    hyp      le statistiche del profilo di iperbolicita' (200 colonne)
    hyp6     media e std di tre sole scale T_geom (0.05 / 0.50 / 1.25):
             sei numeri, gli stessi che ha `degree`, cosi' il confronto
             non e' falsato dalla dimensionalita' -- con ~170 grafi di
             training 200 colonne partono svantaggiate comunque
    atoms    conteggio dei tipi atomici del grafo (somma delle colonne
             one-hot di data.x): il baseline "solo chimica"
    degree   istogramma dei gradi + numero di nodi: il baseline "solo
             struttura"
    size     il solo numero di nodi
    hyp+atoms / hyp+degree   le combinazioni

Stesso protocollo di graph_classification.py -- 10-fold stratificata,
stessi seed, quindi stessi fold -- cosi' i numeri sono confrontabili con
quelli della GIN. Due differenze deliberate:

  - lo StandardScaler e' fittato SOLO sul training fold (dentro una
    Pipeline), quindi qui non c'e' il leakage di standardizzazione che
    graph_classification.py ha;
  - la forza della regolarizzazione e' scelta da una CV interna al
    training fold, perche' con 200 feature e ~170 grafi di training il
    risultato dipende quasi solo da quella.

Alla fine stampa anche la differenza appaiata fold per fold contro il
baseline --baseline (default atoms), che e' la statistica giusta per
confrontare due righe su fold identici.

    python experiments/datasets/graph_classification_linear.py
    python experiments/datasets/graph_classification_linear.py --features hyp+atoms
    python experiments/datasets/graph_classification_linear.py --dataset nci1
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common import (TU_DATASETS, load_graph_node_features, load_tu,
                    metrics_path, node_degrees)

FEATURE_SETS = ["hyp", "hyp6", "atoms", "degree", "size", "hyp+atoms", "hyp+degree"]

# LogisticRegressionCV ne emette due per ogni fit, cioe' ~180 KB di rumore
# per run, che seppelliscono la tabella finale
warnings.simplefilter("ignore", FutureWarning)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(TU_DATASETS), default="mutag")
parser.add_argument("--features", choices=FEATURE_SETS + ["all"], default="all")
parser.add_argument("--baseline", choices=FEATURE_SETS, default="atoms",
                    help="Riga contro cui calcolare la differenza appaiata.")
parser.add_argument("--custom-features-path", default=None)
parser.add_argument("--folds", type=int, default=10)
parser.add_argument("--seeds", default="0,1,2")
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]

graphs = load_tu(args.dataset)
sizes = [d.num_nodes for d in graphs]
labels = np.array([int(d.y) for d in graphs])

profiles = load_graph_node_features(
    args.custom_features_path or metrics_path(args.dataset), sizes)

# un grafo -> un vettore. Le statistiche sono quelle che il sum pooling
# della GIN non puo' vedere: la somma sui nodi conserva (a meno della
# taglia) solo la media, mentre std/max/min descrivono la *forma* della
# distribuzione dei G*(v) dentro la molecola.
hyp_rows, hyp6_rows, atom_rows, deg_rows, size_rows = [], [], [], [], []
for d, prof in zip(graphs, profiles):
    p = prof.numpy()
    hyp_rows.append(np.concatenate([p.mean(0), p.std(0), p.max(0), p.min(0)]))
    # tre scale sole (T_geom = 0.05, 0.50, 1.25), media e std: sei numeri
    hyp6_rows.append(np.concatenate([p[:, [0, 9, 24]].mean(0),
                                     p[:, [0, 9, 24]].std(0)]))

    atom_rows.append(d.x.sum(0).numpy())

    degrees = node_degrees(d.edge_index, d.num_nodes).long()
    deg_rows.append(np.bincount(degrees.numpy(), minlength=5)[:5])

    size_rows.append([d.num_nodes])

BLOCKS = {
    "hyp": np.array(hyp_rows),
    "hyp6": np.array(hyp6_rows),
    "atoms": np.array(atom_rows),
    "degree": np.column_stack([np.array(deg_rows), np.array(size_rows)]),
    "size": np.array(size_rows, dtype=float),
}
BLOCKS["hyp+atoms"] = np.column_stack([BLOCKS["hyp"], BLOCKS["atoms"]])
BLOCKS["hyp+degree"] = np.column_stack([BLOCKS["hyp"], BLOCKS["degree"]])

to_run = FEATURE_SETS if args.features == "all" else [args.features]

print(f"\n{args.dataset} | logistic regression su descrittori graph-level | "
      f"{len(graphs)} grafi | {args.folds}-fold CV | seeds {seeds}")
print(f"dimensioni: " + ", ".join(f"{n} {BLOCKS[n].shape[1]}" for n in to_run))

# fold_scores[name] = accuratezza per ogni (seed, fold), nello stesso ordine
# per ogni riga, cosi' le differenze si possono appaiare
fold_scores = {name: [] for name in to_run}

for seed in seeds:
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(labels, labels):
        for name in to_run:
            X = BLOCKS[name]
            model = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=5, max_iter=5000,
                                     scoring="accuracy", random_state=seed))
            model.fit(X[train_idx], labels[train_idx])
            fold_scores[name].append(model.score(X[test_idx], labels[test_idx]))

print(f"\n{'features':<12} {'dim':>4} | {'test acc':>16} | vs {args.baseline} (appaiata)")
base = np.array(fold_scores[args.baseline]) if args.baseline in fold_scores else None
for name in to_run:
    s = np.array(fold_scores[name])
    line = f"{name:<12} {BLOCKS[name].shape[1]:>4} | {s.mean():.4f} +/- {s.std():.4f} |"
    if base is not None and name != args.baseline:
        d = s - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        line += (f" {d.mean():+.4f} +/- {se:.4f}  "
                 f"({(d > 0).sum()}-{(d < 0).sum()}-{(d == 0).sum()})")
    print(line)

print(f"\nLa +/- della colonna centrale e' la std fra i {len(seeds) * args.folds} fold, "
      f"non l'errore sulla media.\nL'ultima colonna e' media +/- errore standard della "
      f"differenza appaiata, con (vinti-persi-pari).")
