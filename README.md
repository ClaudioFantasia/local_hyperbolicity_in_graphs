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

Cora / CiteSeer pipeline (`experiments/citation/`, select the dataset with
`--dataset cora|citeseer`):

```bash
# 1. Compute per-node local hyperbolicity scores -> <dataset>_node_metrics.csv
python experiments/citation/generate_hyperbolic_features.py --dataset cora

# 2. Train a GCN baseline (bag-of-words / custom features / both)
python experiments/citation/node_classification.py --dataset cora --features bow

# 3. k-hop neighborhood plots
python experiments/citation/visualize.py --dataset cora

# 4. Cora-only link prediction baseline (heuristics + a small GCN)
python experiments/citation/link_prediction_baseline.py
```

ZINC molecular regression (self-contained, not wired to the hyperbolicity code):

```bash
python experiments/ZINC/zinc_gnn.py --epochs 100 --hidden_dim 128 --num_layers 4
```

`experiments/filtration.py` and `experiments/spatial_sparsity.py` are
currently broken (stale imports from before `src/optimization/local.py`
was rewritten) and not maintained — don't trust them to run as-is.
