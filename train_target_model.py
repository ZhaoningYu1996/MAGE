import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from utils.utils import clean_dataset, to_smiles
from utils.model import GCN
import argparse
import random

# Create an argument parser
parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('--data_name', type=str, default='NCI-H23', help='Name of the dataset')

# Parse the arguments
args = parser.parse_args()

# Load the TUDataset
dataset = TUDataset(root='data', name=args.data_name)
print(dataset[0])
# Call the clean_dataset function to update the dataset with the cleaned data
cleaned_dataset_0, cleaned_dataset_1, all_id = clean_dataset(dataset, args.data_name, False)

# make the size of two datasets the same
# sort two datasets by the number of nodes, descending
print(len(cleaned_dataset_0), len(cleaned_dataset_1))
cleaned_dataset_0, id_0 = zip(*sorted(zip(cleaned_dataset_0, all_id[0]), key=lambda x: -x[0].num_nodes))
cleaned_dataset_1, id_1 = zip(*sorted(zip(cleaned_dataset_1, all_id[1]), key=lambda x: -x[0].num_nodes))
print(cleaned_dataset_0[:2], id_0[:2])
print(stop)

if len(cleaned_dataset_0) > len(cleaned_dataset_1):
    cleaned_dataset_0 = list(cleaned_dataset_0[:len(cleaned_dataset_1)])
    all_id = list(id_0[:len(cleaned_dataset_1)]) + list(id_1)
else:
    cleaned_dataset_1 = list(cleaned_dataset_1[:len(cleaned_dataset_0)])
    all_id = list(id_0) + list(id_1[:len(cleaned_dataset_0)])

print(len(cleaned_dataset_0),len(cleaned_dataset_1), len(dataset))
# print(stop)

# Combine the two datasets, and shuffle the data
cleaned_dataset = list(cleaned_dataset_0) + list(cleaned_dataset_1)

# shuffle the data before splitting
combined = list(zip(cleaned_dataset, all_id))
random.shuffle(combined)
cleaned_dataset, all_id = zip(*combined)

# random split the dataset into training and test sets
# 80% training, 20% test
train_dataset = list(cleaned_dataset[:int(0.8 * len(cleaned_dataset))])
test_dataset = list(cleaned_dataset[int(0.8 * len(cleaned_dataset)):])

train_id = list(all_id[:int(0.8 * len(all_id))])
print(train_id)
test_id = list(all_id[int(0.8 * len(all_id)):])
# Save the training and test datasets into a file
torch.save(cleaned_dataset, f'checkpoints/datasets/{args.data_name}.pt')
torch.save(train_id, f'checkpoints/datasets/{args.data_name}_train_id.pt')
torch.save(test_id, f'checkpoints/datasets/{args.data_name}_test_id.pt')
# torch.save(cleaned_dataset_1, f'checkpoints/datasets/{args.data_name}_1.pt')
# print(stop)
# Create a DataLoader for the training and test datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
hidden_dim = 64
model = GCN(input_channels=dataset.num_node_features, hidden_channels=hidden_dim, output_channels=dataset.num_classes).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
best_val_acc = 0
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(test_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # test_acc = test(test_loader)
        torch.save(model.state_dict(), f'checkpoints/models/{args.data_name}_model.pth')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')