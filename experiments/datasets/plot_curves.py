"""
Plot the training curves written by --curves.

I tre script di training (node_classification.py, link_prediction.py,
graph_classification.py) accettano --curves e scrivono la storia per
epoca di ogni run in formato long: una riga per (run, epoca), dove un run
e' un seed (o una coppia seed/fold su graph_classification). Passato da
solo, --curves sceglie il nome in base alla configurazione del run e
salva in data/curves/ (es. nodeclass_cora_gcn_concat-std.csv); con un
path scrive invece li'.

Questo script legge uno o piu' di quei csv e produce un pannello per
colonna: le loss (train/val/test dove ci sono, altrimenti la sola loss di
training) e poi ogni metrica presente nel file (train/val/test acc, val
AUC, val AP, mse...). Ogni run e' una linea sottile; sopra ci va la
mediana fra i run, in grassetto.

    # una sola configurazione, tutti i run
    python experiments/datasets/plot_curves.py data/curves/nodeclass_cora_gcn_bow.csv

    # due configurazioni a confronto (solo le mediane, leggibile)
    python experiments/datasets/plot_curves.py \\
        data/curves/nodeclass_cora_gcn_bow.csv \\
        data/curves/nodeclass_cora_gcn_concat.csv --labels bow concat --median-only

--max-epoch taglia l'asse x, utile quando un run va molto piu' lungo degli
altri e schiaccia tutti gli altri a sinistra.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("curves", nargs="+", help="Uno o piu' csv scritti da --curves.")
parser.add_argument("--labels", nargs="*", default=None,
                    help="Nome di ogni csv in legenda; default il nome del file.")
parser.add_argument("--median-only", action="store_true",
                    help="Disegna solo la mediana fra i run, senza le singole "
                         "linee. Da usare quando confronti piu' csv.")
parser.add_argument("--max-epoch", type=int, default=None)
parser.add_argument("--save", default=None, help="Salva qui invece di mostrare.")
args = parser.parse_args()

labels = args.labels or [os.path.basename(p).replace(".csv", "") for p in args.curves]
assert len(labels) == len(args.curves), "--labels deve avere un nome per csv"

frames = [pd.read_csv(p) for p in args.curves]

# le colonne da plottare: loss piu' tutte le metriche. seed/fold/epoch sono
# identificatori, non curve
id_cols = {"seed", "fold", "epoch"}
value_cols = [c for c in frames[0].columns if c not in id_cols]

# con train/val/test loss piu' train/val/test metrica i pannelli sono tanti:
# li dispongo su una griglia di al massimo 3 colonne invece che su una riga
ncols = min(len(value_cols), 3)
nrows = int(np.ceil(len(value_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                         squeeze=False)
axes = axes.ravel()

for f, (df, label) in enumerate(zip(frames, labels)):
    if args.max_epoch:
        df = df[df["epoch"] <= args.max_epoch]

    # un run = un seed, o una coppia (seed, fold) su graph_classification
    run_cols = [c for c in ("seed", "fold") if c in df.columns]
    runs = list(df.groupby(run_cols))
    color = f"C{f}"

    for ax, col in zip(axes, value_cols):
        if col not in df.columns:
            continue

        if not args.median_only:
            for _, run in runs:
                ax.plot(run["epoch"], run[col], color=color, alpha=0.25, linewidth=0.8)

        # mediana per epoca: i run finiscono a epoche diverse (early stopping),
        # quindi ogni epoca e' mediata sui soli run ancora vivi -- la coda a
        # destra e' meno affidabile, e' fatta di pochi run lunghi
        median = df.groupby("epoch")[col].median()
        alive = df.groupby("epoch")[col].count()
        ax.plot(median.index, median.values, color=color, linewidth=2,
                label=f"{label} (n={len(runs)})")

        # dove restano meno della meta' dei run, la mediana e' rumorosa
        cutoff = alive[alive < len(runs) / 2]
        if len(cutoff) and not args.median_only:
            ax.axvline(cutoff.index[0], color=color, linestyle=":", alpha=0.5)

for ax, col in zip(axes, value_cols):
    ax.set_xlabel("epoca")
    ax.set_ylabel(col)
    ax.set_title("training loss" if col == "loss" else col)
    ax.grid(True, alpha=0.3)
    ax.legend()

for ax in axes[len(value_cols):]:
    ax.axis("off")

fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=150)
    print(f"Salvato in {args.save}")
else:
    plt.show()
