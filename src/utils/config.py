import yaml

def load_graph_config(path="configs/optimization_parameters.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)["graph"]
    return cfg

def load_optimization_config(path="configs/optimization_parameters.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)["optimization"]
    return cfg["temperature"], cfg["lambda_reg"], cfg["k"], *cfg["localized_nodes"]
