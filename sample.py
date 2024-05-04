# In this file, we will import a pretrained GNN and use MAGE class to train an explanation model.

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from utils.model import GCN
from torch_geometric.data import DataLoader
from mage import MAGE
import argparse


# Create an argument parser
parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('--data_name', type=str, default='AIDS', help='Name of the dataset')
parser.add_argument('--input_channels', type=int, default=38, help='Number of input channels')
parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('--output_channels', type=int, default=2, help='Number of output channels')
parser.add_argument('--target_model', type=str, default='checkpoints/models/AIDS_model.pth', help='Path to the pretrained GNN model')
parser.add_argument('--dataset', type=str, default='checkpoints/datasets/AIDS_train.pt', help='Path to the dataset')
parser.add_argument('--label', type=int, default=0, help='Label of the data')

# Parse the arguments
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Initialize the GNN model
model = GCN(input_channels=args.input_channels, hidden_channels=args.hidden_channels, output_channels=args.output_channels).to(device)

# Load the pretrained GNN model
model.load_state_dict(torch.load(args.target_model))
model.eval()

# Load the dataset from checkpoints
dataset = torch.load(args.dataset)
print(len(dataset))

new_dataset = []
count = 0
prob = 0
for data in dataset:
    batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
    pred = model(data.x.to(device), data.edge_index.to(device), batch=batch)
    if pred.argmax().item() == args.label:
        if pred.softmax(1)[0][args.label].item() > 0.9:
            new_dataset.append(data)
            count += 1
            prob += pred.softmax(1)[0][args.label].item()

print(f"Label: {args.label}")

# Initialize the Mage class
mage = MAGE(gnn=model, model=model, dataset=new_dataset, data_name=args.data_name, add_H=False, label=args.label, hidden_channels=args.hidden_channels, output_channels=args.output_channels, device=device)

path_dict = {
    'T_encoder': f'checkpoints/models/{args.data_name}_label_{args.label}_T_encoder.pth', 
    'pred_node_topo': f'checkpoints/models/{args.data_name}_label_{args.label}_pred_node_topo.pth', 
    'pred_node_label': f'checkpoints/models/{args.data_name}_label_{args.label}_pred_node_label.pth', 
    'linear_topo': f'checkpoints/models/{args.data_name}_label_{args.label}_linear_topo.pth', 
    'linear_label': f'checkpoints/models/{args.data_name}_label_{args.label}_linear_label.pth', 
    'T_mean': f'checkpoints/models/{args.data_name}_label_{args.label}_T_mean.pth', 
    'T_var': f'checkpoints/models/{args.data_name}_label_{args.label}_T_var.pth'}

mage.load(path_dict)

sampled_data, pred_prob, invalid_count, tree_acc = mage.sample(100)

# Count how many unique smiles in saampled_data
unique_smiles = set()
for data in sampled_data:
    unique_smiles.add(data)

print(f'Unique smiles: {len(unique_smiles)}')
print(f"Unique rate: {len(unique_smiles)/len(sampled_data)}")
# print(f'Sampled data: {sampled_data}')
print(f'Invalid count: {invalid_count}')
print(f'Predicted probability: {pred_prob}')
print(f'Tree accuracy: {tree_acc}')

