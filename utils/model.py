import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,  output_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index=None, batch=None, return_embedding=False, classifier=False):
        if classifier:
            return self.lin(x)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch != None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x
        x = self.lin(x)
        
        return x
    
class TGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,  output_channels):
        super(TGCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        # self.conv1 = Linear(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None, return_embedding=False, classifier=False):
        if classifier:
            return self.lin(x)
        # 1. Obtain node embeddings 
        # print(motif_embedding)
        # x = self.conv1(x)
        if edge_weight != None:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = x.relu()
        if edge_weight != None:
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        x = x.relu()
        if edge_weight != None:
            x = self.conv3(x, edge_index, edge_weight)
        else:
            x = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch != None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
            return x
        x = self.lin(x)
        
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels, output_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_channels, hidden_channels, heads=heads, add_self_loops=False, bias=False)

    def forward(self, x, edge_index, return_weight=False):

        x, edge_tuple = self.conv1(x, edge_index, return_attention_weights=return_weight)

        return x, edge_tuple