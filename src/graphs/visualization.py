import networkx as nx
import matplotlib.pyplot as plt
import math

def draw_layout(G, type = 'spring', pos = None, seed = 42):
    """
    Creating the 'pos' to run draw_graph
    """
    if type == 'spring':
        pos = nx.spring_layout(G, pos=pos, seed=seed)
    return pos


def draw_graphs(graphs, poses, titles=None, 
                    highlight_nodes=None, highlight_edges=None,
                    node_params=None, edge_params=None, 
                    highlight_params=None, base_figsize=(6, 6)):
    
    # Normalizzazione input (singolo grafo -> lista)
    if isinstance(graphs, nx.Graph):
        graphs, poses = [graphs], [poses]
        titles = [titles] if titles else [None]
        highlight_nodes = [highlight_nodes] if highlight_nodes else [None]
        highlight_edges = [highlight_edges] if highlight_edges else [None]

    n = len(graphs)
    n_cols = min(n, 3)
    n_rows = math.ceil(n / n_cols)
    
    # Setup stili di default
    n_params = {'node_color': 'lightblue', 'node_size': 500, 'font_size': 10}
    if node_params: n_params.update(node_params)
    
    e_params = {'edge_color': 'gray', 'alpha': 0.7}
    if edge_params: e_params.update(edge_params)
    
    h_params = {'node_color': 'red', 'edge_color': 'red', 'width': 2, 'node_size': 500}
    if highlight_params: h_params.update(highlight_params)

    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(base_figsize[0]*n_cols, base_figsize[1]*n_rows))
    axes = [axes] if n == 1 else axes.flatten()

    for i in range(n):
        ax, G, pos = axes[i], graphs[i], poses[i]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=n_params['node_color'], node_size=n_params['node_size'])
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=e_params['edge_color'], alpha=e_params['alpha'])
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=n_params['font_size'])
        
        
        # 2. Highlight Nodi (se presenti per questo grafo)
        if highlight_nodes and i < len(highlight_nodes) and highlight_nodes[i]:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=highlight_nodes[i],
                                   node_color=h_params['node_color'], 
                                   node_size=h_params['node_size'])
            
        # 3. Highlight Archi (se presenti per questo grafo)
        if highlight_edges and i < len(highlight_edges) and highlight_edges[i]:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=highlight_edges[i],
                                   edge_color=h_params['edge_color'], 
                                   width=h_params['width'])
        
        if titles and i < len(titles) and titles[i]:
            ax.set_title(titles[i])
        ax.set_axis_off()

    # Nascondi assi extra
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()
    return fig, axes


def draw_quadruples(G, pos, quadruples):
    """
    Wrapper specifico per visualizzare le quadruple critiche di Gromov.
    """
    graphs = [G] * len(quadruples)
    poses = [pos] * len(quadruples)
    titles = [f"Quadruple {q}" for q in quadruples]
    
    highlight_nodes = [list(q) for q in quadruples]
    

    return draw_graphs(
        graphs, poses, 
        titles=titles, 
        highlight_nodes=highlight_nodes,
    )


