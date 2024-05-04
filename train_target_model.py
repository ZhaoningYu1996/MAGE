import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from utils.utils import clean_dataset, to_smiles
from utils.model import GCN
import argparse
import random

# Create an argument parser
parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('--data_name', type=str, default='Mutagenicity', help='Name of the dataset')

# Parse the arguments
args = parser.parse_args()

# Load the TUDataset
dataset = TUDataset(root='data', name=args.data_name)
print(dataset[0])
# Call the clean_dataset function to update the dataset with the cleaned data
cleaned_dataset, cleaned_smiles = clean_dataset(dataset, args.data_name, False)

print(len(cleaned_dataset), len(dataset))

# shuffle the data before splitting
random.shuffle(cleaned_dataset)

# random split the dataset into training and test sets
# 80% training, 20% test
train_dataset = cleaned_dataset[:int(0.8 * len(cleaned_dataset))]
test_dataset = cleaned_dataset[int(0.8 * len(cleaned_dataset)):]

# Save the training and test datasets into a file
torch.save(train_dataset, f'checkpoints/datasets/{args.data_name}_train.pt')
torch.save(test_dataset, f'checkpoints/datasets/{args.data_name}_test.pt')

# Create a DataLoader for the training and test datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
hidden_dim = 64
model = GCN(input_channels=dataset.num_node_features, hidden_channels=hidden_dim, output_channels=dataset.num_classes).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Define a train function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)

# Define a test function
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# Train the model and save the best model
best_val_acc = test_acc = 0
for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(test_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = test(test_loader)
        torch.save(model.state_dict(), f'checkpoints/models/{args.data_name}_model.pth')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')