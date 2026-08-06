# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code studying **local (Gromov) hyperbolicity** in graphs. Gromov
4-point hyperbolicity (`delta`) is normally computed globally over all
4-node subsets of a graph, which is intractable beyond a few hundred
nodes. This repo develops *local, differentiable/soft relaxations* of that
measure (KL-divergence / entropic / softmax aggregations over a node's
k-hop neighborhood) so it can be computed as a per-node score on larger
graphs (Cora, CiteSeer, ZINC), then used as a feature for downstream GNN
tasks (node classification, link prediction, regression).

There is no build system, package manifest, test suite, or linter in this
repo — it's an active research/experimentation codebase driven by ad-hoc
scripts and Jupyter notebooks. Treat correctness verification as "run the
script and inspect output," not "run pytest."

## Who you're working with

The user is a researcher in AI/ML with a computer engineering background —
not a professional software engineer. Their focus is the math and research
content (hyperbolicity theory, scoring methods, experiment results), not
software architecture. They'd rather read and reason about straightforward
code than navigate abstractions.

## Code style: keep it simple

The existing code in this repo (`src/`, `experiments/`) is intentionally
plain: flat procedural scripts, minimal abstraction, few helper functions,
little error handling. Match that style rather than "improving" it.

- For small/one-off asks (a short script, a quick analysis snippet), write
  the most direct code that does the job. Don't add `argparse`/`sys.argv`
  handling, `input()` fallbacks, type hints, docstrings, or
  `if __name__ == "__main__"` guards unless the user asks for them or the
  file you're editing already uses that pattern.
- Don't wrap a single sequential task in a function "for cleanliness" if
  it's only called once — top-level script code is fine here.
- Skip error handling and input validation for research scripts the user
  runs themselves — trust the inputs, don't guard against malformed args
  or bad files unless asked.
- When extending `src/optimization` or `src/graphs`, match the existing
  terseness (short functions, direct numpy/scipy calls) rather than
  introducing new abstraction layers, config objects, or class hierarchies.
- When unsure whether something warrants structure (a one-off snippet vs.
  a script that will be reused across experiments), default to the
  simpler version — the user will ask for more structure if they want it.

## Running code

There is no `requirements.txt`/`pyproject.toml`. Core dependencies used
across the codebase (install as needed): `networkx`, `numpy`, `scipy`,
`matplotlib`, `pyyaml`, `tqdm`, `pandas`, `scikit-learn`, `torch`,
`torch_geometric`.

Just use the `hyperLocal` conda environment.  

Scripts add the repo root to `sys.path` manually (see the
`sys.path.append(...)` line near the top of each `experiments/*.py` file),
so they can be run directly regardless of working directory:

```bash
python experiments/run_experiment.py --method entropic
python experiments/citation/generate_hyperbolic_features.py --dataset cora
python experiments/citation/node_classification.py --dataset cora --task classification --features bow
python experiments/citation/node_classification.py --dataset citeseer --features concat --custom-features-path <path>
python experiments/citation/visualize.py --dataset cora
python experiments/citation/link_prediction_baseline.py
python experiments/ZINC/zinc_gnn.py --epochs 100 --hidden_dim 128 --num_layers 4
```

Read each script's module docstring/`--help` first — several
(`node_classification.py`, `zinc_gnn.py`) have detailed usage docs at the
top covering feature-file formats, seed handling, and CLI flags.

`experiments/filtration.py` and `experiments/spatial_sparsity.py` are
currently broken and not maintained — don't trust them to run as-is.
`filtration.py` imports `optimization.local` missing the `src.` prefix,
unpacks `load_optimization_config()`'s return value into 4 names when it
now returns 6, calls an undefined `hierarchical` graph generator, and
imports `compute_gromov_hyperbolicity` from a `src/hyperbolicity/` module
that doesn't exist. `spatial_sparsity.py` imports `new_score_KL_divergence`,
a name that no longer exists in `src/optimization/local.py` (it predates
the current active pipeline), and two of its code paths reference an
undefined `dist_matrix`. Both were left as-is during the 2026-08 cleanup
pass rather than fixed, since nothing else in the repo depends on them.

## Architecture

**`src/graphs/`** — graph construction and drawing.
- `utils.py::create_graph(type, seed, **kwargs)` is the universal entry
  point for building any graph used in experiments (star, tree, cycle,
  path, complete, erdos_renyi, lattice, sbm, geometric, tree_of_cycles,
  tree_with_grid, molecule_like, hierarchical). Dispatches on `type` and
  reads generator-specific params out of `**kwargs` (typically loaded
  straight from `configs/optimization_parameters.yaml`). Returns `(G, pos)`.
- `visualization.py` — matplotlib helpers (`draw_graphs`,
  `draw_graph_with_values` for heatmap-style node coloring,
  `draw_quadruples` for visualizing the 4-node subsets driving a
  hyperbolicity score, `plot_hist`).

**`src/optimization/`** — the core hyperbolicity math.
- `objectives.py::gromov_energy(quads, dist_matrix)` computes the exact
  4-point Gromov delta for a batch of node quadruples given an all-pairs
  shortest-path distance matrix — the ground-truth quantity everything
  else approximates locally. Also has `compute_gromov_hyperbolicity(G)`
  (exact, only tractable up to ~300 nodes) and spectral/diffusion energy
  helpers (heat-kernel / denoising diffusion signals over the graph
  Laplacian) used by some of the weighting schemes.
- `local.py` — **the active/current pipeline is only the top portion**:
  `get_neighborhood` (k-hop neighborhood, `'full_neighborhood'` or sampled
  `'increasing_neighborhood'` strategy), `sampling_quads` (exact or
  randomly-sampled 4-subsets, capped at `comb(100,4)`), and `KL_score`
  (the current soft/temperature-weighted local hyperbolicity score —
  log-sum-exp over sampled quads, weighted by distance-to-target and
  Gromov energy). Below a marked-off comment block is earlier experimental
  scoring code (`score_KL_divergence`, `score_entropic`, `score_softing`,
  `_get_quads_and_energies`, `score_KL_divergence_batched`,
  `precompute_energies`) kept only because `experiments/spatial_sparsity.py`
  and `experiments/filtration.py` still import from it — not the primary
  API for new work, and not maintained (both those scripts are currently
  broken, see above). Everything else that was unused anywhere in the repo
  (`old_get_neighborhood`, `score_max`, `score_softmax`,
  `batched_subgraph_score_KL_divergence`, plus the standalone
  `khop_local.py` scratch copy that used to sit at the repo root) was
  removed in the 2026-08 cleanup pass.
- `solver.py` — the regularized-optimization side: given a cost vector
  (Gromov energies) and a reference distribution, solves for a
  simplex-constrained `mu` under entropic (`solve_entropic_regularization`,
  softmax), L2 (`solve_l2_regularization`), or KL (`solve_KL_regularization`)
  regularization. Used to turn per-quad Gromov energies into a soft
  "importance" distribution over quads, e.g. for the `score_entropic`
  path and `experiments/spatial_sparsity.py`'s node-contribution analysis.

**`src/utils/config.py`** — loads `configs/optimization_parameters.yaml`
(the single config file, with top-level `graph:` and `optimization:`
sections). `load_graph_config()` returns the raw `graph` dict for
`create_graph(**kwargs)`. `load_optimization_config()` returns a
*positional* tuple `(temperature, geometric_temperature, lambda_reg, sigma,
k, *target_nodes)` — note it unpacks `target_nodes` (a list in the yaml)
into trailing positional values, so callers must match that exact order.

**`experiments/`** — runnable scripts, one pipeline per file/dataset:
- `run_experiment.py`, `filtration.py`, `spatial_sparsity.py` — synthetic
  graphs (from `create_graph`) driven by `configs/optimization_parameters.yaml`,
  comparing local hyperbolicity scoring methods (`KL_divergence`,
  `entropic`, `softing`/`softmax`) and visualizing per-node heatmaps /
  score distributions / quad-level contributions.
- `citation/` — one dataset-parametrized pipeline for both Cora and
  CiteSeer, selected via `--dataset cora|citeseer` (merged from separate
  per-dataset scripts in the 2026-08 cleanup pass, since they were ~95%
  identical): `generate_hyperbolic_features.py` (computes per-node local
  hyperbolicity scores on the Planetoid graph via `KL_score`,
  largest-connected-component only, writes results to
  `<dataset>_node_metrics.csv` next to the script), `node_classification.py`
  (configurable-depth GCN baseline with early stopping that consumes those
  features — supports `bow` / `custom` / `concat` feature modes and
  averages results over multiple seeds for paired comparisons), and
  `visualize.py` (k-hop neighborhood/growth-curve plots). Also
  `link_prediction_baseline.py`, which stays Cora-only (no CiteSeer
  equivalent exists).
- `ZINC/zinc_gnn.py` — separate, self-contained GINE-based GNN pipeline
  for molecular property regression on the ZINC-12k benchmark; not wired
  to the hyperbolicity code.

**`configs/optimization_parameters.yaml`** — the single source of
experiment parameters: `graph.type` selects which `create_graph` generator
runs and its shape params (`n`, `tree_height`, `leaves_per_node`,
`size_grid`, SBM block sizes, etc.); `optimization.*` holds scoring
hyperparameters (`temperature`, `geometric_temperature`, `lambda_reg`,
`alpha_diffusion`, `sigma`, `time`, `k`, `target_nodes`). Editing this file
is the normal way to change which synthetic graph / scoring config the
top-level `experiments/*.py` scripts (`run_experiment.py`, `filtration.py`,
`spatial_sparsity.py`; not `citation/` or `ZINC/`) run against.

**`data/`** — downloaded Planetoid datasets (`data/Cora`, `data/citeseer`,
raw + processed PyG format) and generated figures (`data/LSE/<graph_type>/<method>/...`,
`data/newFig/...`). Gitignored — treat as cache/output, not source.

**`notebooks/`** — exploratory Jupyter notebooks, numbered roughly in the
order the underlying ideas were developed (hyperbolicity testing →
connectivity → local hyperbolicity → spectral regularization → local
regularization → robustness analysis). Useful for understanding the
motivation/derivation behind the `src/optimization` code in more depth
than the scripts alone show.

## Conventions to know

- All-pairs shortest-path distance matrices (`compute_distance_nodes`,
  duplicated in both `src/graphs/utils.py` and `src/optimization/objectives.py`)
  are indexed by *position in `sorted(G.nodes())` / `list(G.nodes())`*, not
  by raw node id — quad tuples and target indices passed into scoring
  functions must already be in that index space.
- `gromov_energy` expects each quad as `(x, y, z, w)` and returns
  `(largest_pair_sum - middle_pair_sum) / 2` across the three ways to pair
  up 4 points — the standard 4-point Gromov delta.
- Quad results are memoized in a `quad_cache` dict keyed by
  `tuple(sorted(quad))`, passed explicitly through call chains rather than
  stored on an object — when adding a new scoring function, follow this
  same explicit-cache-passing pattern rather than introducing global state.
