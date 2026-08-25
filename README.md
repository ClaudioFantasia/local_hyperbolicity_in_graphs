# Local Hyperbolicity in Graphs

Research code for studying local (Gromov) hyperbolicity in graphs. Exact
4-point Gromov hyperbolicity is intractable beyond a few hundred nodes, so
this repo builds local, soft-relaxed approximations of it (see
`src/optimization/local.py::KL_score`) that can be computed per-node on
larger graphs (Cora, CiteSeer, ZINC) and used as a feature for downstream
GNN tasks.

For the full architecture writeup (module-by-module, what's active vs.
legacy, conventions to know) see `CLAUDE.md`.

## Setup

Use the `hyperLocal` conda environment. Core dependencies: `networkx`,
`numpy`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`, `pandas`,
`scikit-learn`, `torch`, `torch_geometric`.

## Running things

Synthetic-graph experiments (driven by `configs/optimization_parameters.yaml`):

```bash
python experiments/run_experiment.py --method entropic
```

Dataset pipeline (`experiments/datasets/`, select the dataset with
`--dataset cora|citeseer|mutag|...`):

```bash
# 1. Compute per-node local hyperbolicity scores
#    -> data/hyperbolic_features/<dataset>_node_metrics.csv
python experiments/datasets/generate_features.py --dataset cora

# 2. Train a GCN baseline (bag-of-words / custom features / both).
#    --curves salva le curve per epoca in data/curves/, con nome automatico
python experiments/datasets/node_classification.py --dataset cora --features bow --curves

# 3. Link prediction, e classificazione di grafi sui dataset TU
python experiments/datasets/link_prediction.py --dataset cora --features concat
python experiments/datasets/graph_classification.py --dataset mutag --features concat

# 4. Plot delle curve salvate al punto 2
python experiments/datasets/plot_curves.py data/curves/nodeclass_cora_gcn_bow.csv
```

ZINC molecular regression (self-contained, not wired to the hyperbolicity code):

```bash
python experiments/ZINC/zinc_gnn.py --epochs 100 --hidden_dim 128 --num_layers 4
```

`experiments/filtration.py` and `experiments/spatial_sparsity.py` are
currently broken (stale imports from before `src/optimization/local.py`
was rewritten) and not maintained — don't trust them to run as-is.
