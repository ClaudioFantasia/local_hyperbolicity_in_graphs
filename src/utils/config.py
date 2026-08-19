"""
Loads configs/optimization_parameters.yaml, which has two top-level
sections: `graph` (kwargs for src.graphs.utils.create_graph) and
`optimization` (hyperparameters for the scoring functions in
src/optimization).
"""

import os

import yaml

# absolute, so the scripts in experiments/*/ find the config whatever the
# working directory is
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "optimization_parameters.yaml")


def load_config(path=DEFAULT_CONFIG):
    """The whole yaml file as a dict with 'graph' and 'optimization' keys."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_graph_config(path=DEFAULT_CONFIG):
    """The 'graph' section only, e.g. create_graph(**load_graph_config())."""
    return load_config(path)["graph"]


def load_optimization_config(path=DEFAULT_CONFIG):
    """
    The 'optimization' section only, as a plain dict -- access whichever
    keys you need by name, e.g. cfg["temperature"], cfg["k"].

    (Older versions of this function unpacked the section into a fixed
    positional tuple, which broke silently whenever a key was added,
    removed, or `target_nodes` didn't have exactly one entry. Returning
    the dict avoids that footgun.)
    """
    return load_config(path)["optimization"]
