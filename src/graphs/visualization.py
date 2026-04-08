import networkx as nx
import matplotlib.pyplot as plt


def draw_layout(G, type = 'spring', pos = None, seed = 42):
    """
    Creating the 'pos' to run draw_graph
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)
    else:
        # this is useful because it maintain the old layout, changing just a bit
        pos = nx.spring_layout(G, pos=pos, seed=seed)
    return pos

def draw_graph(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=500,
    font_size=10,
    title=None
):
    plt.figure(figsize=(6, 6))
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color)
    
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.show()
    #TODO: just check if it can be better to return fig
    return   