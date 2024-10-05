import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def main():
    # Load the Cora dataset
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Initialize the model
    model = GCN(dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)

    # Pretrained weights path (you would need to have this file)
    pretrained_path = "path/to/pretrained/gcn_cora.pth"

    # Load pretrained weights if available
    try:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights.")
    except FileNotFoundError:
        print("Pretrained weights not found. Starting with random initialization.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # Final evaluation
    train_acc, val_acc, test_acc = test(model, data)
    print(f'Final results:')
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

if __name__ == '__main__':
    main()