import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

class HyperbolicityGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HyperbolicityGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        # Node-level prediction
        out = self.lin(x)
        return out

def train():
    dataset_path = 'data/dataset/synthetic_hyperbolicity2.pt'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}!")
        return
        
    dataset = torch.load(dataset_path, weights_only=False)
    
   
    train_dataset = dataset[:40]
    test_dataset = dataset[40:]
    
    print(len(test_dataset))

    in_channels = train_dataset[0].num_node_features
    model = HyperbolicityGNN(in_channels=in_channels, hidden_channels=32, out_channels=1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training Loop
    model.train()
    epochs = 2000
    for epoch in range(1, epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
        for data in train_dataset:
            # Predict local hyperbolicity for all nodes in the graph
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            total_loss += loss.item()
            
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {total_loss/len(train_dataset):.4f}')
            
    # Evaluation
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_dataset:
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            test_loss += loss.item()
            all_preds.extend(out.flatten().numpy())
            all_targets.extend(data.y.flatten().numpy())
            
    print(f'\nTest MSE: {test_loss/len(test_dataset):.4f}')
    
    # Visualize Test Predictions vs Ground Truth
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_preds, alpha=0.6, color='blue', label='Predictions')
    target_range = [min(all_targets), max(all_targets)]
    plt.plot(target_range, target_range, color='red', linestyle='--', label='Ideal')
    plt.xlabel('True Local Hyperbolicity')
    plt.ylabel('Predicted Local Hyperbolicity')
    plt.title('GNN Predictions vs Ground Truth on Test Graphs')
    plt.legend()
    plt.grid(True)
    os.makedirs('data/figures', exist_ok=True)
    plt.savefig('data/figures/gnn_test_predictions1.png')
    print("Saved evaluation plot to data/figures/gnn_test_predictions.png")

if __name__ == "__main__":
    train()
