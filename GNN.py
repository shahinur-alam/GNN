import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Step 1: Load the dataset (Cora)
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# Step 2: Define the GCN model architecture
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 3: Initialize the model with the appropriate number of features and classes
model = GCN(in_channels=dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)

# Step 4: Load pretrained weights (if they are saved in 'pretrained_model.pth')
model.load_state_dict(torch.load('pretrained_model.pth'))

# Step 5: Move model to evaluation mode for inference
model.eval()

# Step 6: Perform inference on the dataset (Cora graph)
data = dataset[0]  # Get the Cora graph
with torch.no_grad():
    out = model(data)
    _, pred = out.max(dim=1)  # Get the predicted class labels

# Step 7: Calculate accuracy on the entire dataset
correct = pred.eq(data.y).sum().item()
accuracy = correct / data.num_nodes
print(f'Accuracy of the pretrained model: {accuracy:.4f}')

# Step 8: (Optional) Fine-tune the model further with training
# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Set the model to training mode
model.train()

# Number of epochs for fine-tuning
epochs = 100

# Fine-tuning loop
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    out = model(data)  # Forward pass
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute loss only on the training set
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Step 9: Evaluate the model after fine-tuning
model.eval()
with torch.no_grad():
    out = model(data)
    _, pred = out.max(dim=1)
    correct = pred.eq(data.y).sum().item()
    accuracy = correct / data.num_nodes
    print(f'Accuracy after fine-tuning: {accuracy:.4f}')
