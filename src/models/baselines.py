import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

class SimpleGCN(torch.nn.Module):
    """
    Una Graph Convolutional Network (GCN) classica.
    Spesso usata come baseline. Soffre di oversquashing su grafi tree-like
    o con bottleneck quando si aumenta il numero di layer.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # Primo layer di input
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Layer nascosti
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Layer di output
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        """
        x: tensore delle feature dei nodi di shape [num_nodes, in_channels].
           In PyTorch Geometric (PyG), i batch di grafi disconnessi vengono
           trattati unendo i nodi in una singola grande matrice x.
        edge_index: tensore della connettività (archi) di shape [2, num_edges].
                    Questo è il formato COO (coordinate format, riga/colonna),
                    necessario per i calcoli efficienti delle sparse matrix in PyG.
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            
        # Nessuna funzione di attivazione nell'ultimo layer se stiamo 
        # calcolando dei logits diretti.
        x = self.convs[-1](x, edge_index)
        return x


class SimpleGIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN).
    Più espressiva della GCN: usa una piccola MLP all'interno di ogni operazione 
    di calcolo prima di aggregare i messaggi. Molto potente per alcuni task
    ma comunque soggetta a oversquashing in setting long-range.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            
            # In GIN, ogni layer apprende una sequenza MLP per mappare le feature
            nn = Sequential(
                Linear(in_dim, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_dim)
            )
            # train_eps=True permette di apprendere il parametro epsilon di GIN
            self.convs.append(GINConv(nn, train_eps=True))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        return x
