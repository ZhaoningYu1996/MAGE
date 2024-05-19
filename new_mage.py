# Please write an official comment and the code for the Mage class here. The class is taking a pretrained GNN, and the dataset that used to pretrain the GNN. The Mage class is responsible for create a graph encoder, a tree encoder, a tree decoder, and a graph decoder to generate model level explanation for the target GNN.
# The class should have the following methods:
# - encode_graph: This method should take a graph as input and return the encoded graph.
# - encode_tree: This method should take a tree as input and return the encoded tree.
# - decode_tree: This method should take an encoded tree as input and return the decoded tree.
# - decode_graph: This method should take an encoded graph as input and return the decoded graph.
# - explain: This method should sample explanations for the target GNN.

# Start coding here
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from utils.model import TGCN
from utils.tree import Tree, TreeDataset
from utils.utils import to_tudataset, get_mol, sanitize_mol, get_smiles, sanitize_smiles, can_assemble
from utils.loader import custom_collate, DataLoader
from utils.motif_filter import motif_filter
from torch_geometric.data import Data, Batch
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.model_selection import train_test_split
import os

class MAGE:
    def __init__(self, gnn, model, dataset, whole_dataset, smiles_set, data_name, add_H, hidden_channels, output_channels, label, device):
        self.gnn = gnn
        self.dataset = dataset
        self.whole_dataset= whole_dataset
        self.smiles_set = smiles_set
        self.data_name = data_name
        self.add_H = add_H
        self.label = label
        self.motif_id = {}
        self.id_motif = {}
        self.device = device
        self.hidden_channels = hidden_channels
        self.model = model.to(self.device) # Pretrained target GNN model as the graph encoder
        for param in self.model.parameters(): # Freeze the parameter of self.model
            param.requires_grad = False
            
        # self.G_mean = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        # self.G_var = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.T_mean = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.T_var = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.T_encoder = TGCN(hidden_channels, hidden_channels, output_channels).to(self.device)
        self.pred_node_topo = nn.Linear(hidden_channels, 2).to(self.device)
        self.linear_topo = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        self.linear_label = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        # self.pred_node_label = nn.Linear(self.hidden_channels, self.motif_embedding.size(0)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.get_motif_embedding()
        

    def rsample(self, embedding, W_mean, W_var):
        # Compute the mean and log variance
        mu = W_mean(embedding)
        # log_var = W_var(embedding)
        log_var = -torch.abs(W_var(embedding)) # Use the nagetive absolute value of the log variance, with more control
        # Sample from the Gaussian distribution using reparameterization trick
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # `randn_like` generates a tensor with the same size as std, sampled from a standard normal distribution
        z = mu + eps * std  # Reparameterization trick
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
        return z, kl_loss / embedding.size(0)
    
    def get_tree(self, data, test=False):
        # Get the tree from the dataset
        tree = Tree(data, self.data_name, self.add_H)
        tree.transform()
        # print(stop)
        node_motif_map = defaultdict(set)
        node_indices = []
        for i, motif in enumerate(tree.fragments):
            node_indices.append(tree.atom_list[i])
            if motif not in self.motif_id:
                if test:
                    return None
                self.motif_id[motif] = len(self.motif_id)
                self.id_motif[self.motif_id[motif]] = motif
            for node in tree.atom_list[i]:
                node_motif_map[node].add(i)
        # Create x for the tree, each node is a motif, the node feature is the motif id
        x = torch.tensor([[self.motif_id[motif]] for motif in tree.fragments], dtype=torch.long)
        edge_index = []

        # Need to think about singlton nodes.
        for bond in tree.bond_list:
            set1 = node_motif_map[bond[0]]
            set2 = node_motif_map[bond[1]]

            for node1 in set1:
                for node2 in set2:
                    edge_index.append([node1, node2])
                    edge_index.append([node2, node1])
        if len(edge_index) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        tree_data = Data(x=x, edge_index=edge_index, node_ori_map=node_indices, data=data)
        
        return tree_data
    
    def get_motif_embedding(self):

        trees = [self.get_tree(graph) for graph in tqdm(self.dataset)]
        self.trees = trees
        # Get the motif embedding from the target GNN model
        print("Getting motif embedding!")
        self.motif_embedding = []
        
        self.model.eval()
        with torch.no_grad():
            for motif in tqdm(self.motif_id.keys()):
                mol = sanitize_mol(get_mol(motif, self.data_name), self.add_H)
                data = to_tudataset(mol, self.data_name)
                data.to(self.device)
                batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
                motif_embedding = self.model(data.x, data.edge_index, batch, return_embedding=True)
                # print(motif_embedding.shape)
                self.motif_embedding.append(motif_embedding)
                
        self.motif_embedding = torch.stack(self.motif_embedding).squeeze().to(self.device)
        self.pred_node_label = nn.Linear(self.hidden_channels, self.motif_embedding.size(0)).to(self.device)

    def encode_tree(self, tree):
        # print("Start encode the tree!")
        T_emb = self.T_encoder(self.motif_embedding[tree.x.view(-1)], tree.edge_index, batch=tree.batch, return_embedding=True)
        return T_emb

    def decode_tree(self, trees, graphs, z_tree, max_iter):
        tree_list = Batch.to_data_list(trees)
        topo_loss = 0
        label_loss = 0
        emb_loss = 0
        pred_loss = 0
        total_acc_correct = 0
        total_acc_wrong = 0
        # ori_tree_emb = self.encode_tree(trees)
        # graph_emb = self.encode_graph(graphs)
        # emb_loss += self.mse_loss(ori_tree_emb, graph_emb)
        total_pred_prob = 0
        count_graph_pred = 0
        count_topo = 0
        count_label = 0
        count_negative = 0
        count_positive = 0
        count_embedding = 0
        
        # For each tree, calculate the loss
        for i, data in enumerate(tree_list):
            # tree_emb = torch.randn(1, self.hidden_channels).to(self.device)
            tree_emb = z_tree[i].unsqueeze(0)
            h_curr = torch.zeros_like(tree_emb)
            if data.x.shape[0] == 1:
                topo_pred = self.pred_node_topo(self.linear_topo(tree_emb.add(h_curr)))
                label_pred = self.pred_node_label(self.linear_label(tree_emb.add(h_curr)))

                topo_loss += self.criterion(topo_pred, torch.tensor([1]).to(self.device))
                label_loss += self.criterion(label_pred, data.x[0])
                count_topo += 1
                count_label += 1
                count_positive += 1

                tree_pred_loss, acc_correct, acc_wrong, new_tree, _ = self.sample_tree(tree_emb, max_iter=max_iter)
                pred_loss += tree_pred_loss
                total_acc_correct += acc_correct
                total_acc_wrong += acc_wrong
                continue
            
            # Use BFS to traverse the tree
            queue = deque([(-1, 0)])
            node_neighbors = defaultdict(set)
            for j in range(data.edge_index.shape[1]):
                node_neighbors[data.edge_index[0, j].item()].add(data.edge_index[1, j].item())
                node_neighbors[data.edge_index[1, j].item()].add(data.edge_index[0, j].item())
            visited = set()
            node_order = {}
            curr_x = torch.empty((0, 1), dtype=torch.long).to(self.device)
            curr_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            while queue:
                # print(f"Queue: {queue}")
                length = len(queue)
                for j in range(length):
                    node = queue.popleft()
                    visited.add(node[1])
                    if node[1] != -2:
                        node_order[node[1]] = len(node_order)
                    if node[0] != -1:
                        # print(curr_x, curr_edge_index)
                        with torch.no_grad():
                            h_curr = self.T_encoder(self.motif_embedding[curr_x.view(-1)], curr_edge_index, return_embedding=True)[node_order[node[0]]]

                    topo_pred = self.pred_node_topo(self.linear_topo(tree_emb.add(h_curr)))
                    count_topo += 1
                    
                    # Change label for the last node in the same level
                    if node[1] == -2:
                        topo_loss += self.criterion(topo_pred, torch.tensor([0], device=self.device))
                        count_negative += 1
                    else:
                        topo_loss += self.criterion(topo_pred, torch.tensor([1], device=self.device))
                        count_positive += 1
                        curr_x = torch.cat((curr_x, data.x[node[1]].unsqueeze(0)), dim=0)
                        if node[0] != -1:
                            curr_edge_index = torch.cat((curr_edge_index, torch.tensor([[node_order[node[0]], node_order[node[1]]], [node_order[node[1]], node_order[node[0]]]], dtype=torch.long, device=self.device)), dim=1)

                        for neighbor in node_neighbors[node[1]]:
                            if neighbor not in visited:
                                queue.append((node[1], neighbor))
                        queue.append((node[1], -2))
                        
                        label_pred = self.pred_node_label(self.linear_label(tree_emb.add(h_curr)))
                        label_loss += self.criterion(label_pred, data.x[node[1]])
                        count_label += 1
            
            # Freeze teacher forcing
            tree_pred_loss, acc_correct, acc_wrong, new_tree, _ = self.sample_tree(tree_emb, max_iter=max_iter)
            pred_loss += tree_pred_loss
            total_acc_correct += acc_correct
            total_acc_wrong += acc_wrong
            # print(stop)

            smiles, pred_prob, graph_embedding = self.decode_graph(new_tree)
            if smiles:
                batch = torch.zeros(new_tree.x.size(0), dtype=torch.long).to(self.device)
                # with torch.no_grad():
                tree_emb = self.T_encoder(self.motif_embedding[new_tree.x.view(-1)], new_tree.edge_index, batch=batch, return_embedding=True)
                emb_loss += self.mse_loss(tree_emb, graph_embedding)
                count_embedding += 1
        
            # if smiles:
            #     count_graph_pred += 1
            #     total_pred_prob += pred_prob
        # print(topo_loss, label_loss)
        # return topo_loss / count_topo, label_loss / count_label, emb_loss / len(tree_list), pred_loss / len(tree_list), total_acc_correct, total_acc_correct+total_acc_wrong, total_pred_prob, count_graph_pred
        # return emb_loss, pred_loss, total_acc_correct, total_acc_correct+total_acc_wrong
        # print(f"Number of valid graph decoding: {count_embedding}")
        # return topo_loss / count_topo, label_loss / count_label, pred_loss / len(tree_list), total_acc_correct, total_acc_correct+total_acc_wrong, count_positive, count_negative 
        return topo_loss / count_topo, label_loss / count_label, pred_loss / len(tree_list), emb_loss, total_acc_correct, total_acc_correct+total_acc_wrong, count_positive, count_negative 
          
    def sample_tree(self, z_tree, max_iter, test=False):
        iter = 0
        queue = deque([(-1, 0)])

        curr_x = torch.empty((0, self.motif_embedding.size(0)), dtype=torch.long).to(self.device)
        curr_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        curr_edge_weight = torch.empty(0, dtype=torch.float).to(self.device)

        h_curr = torch.zeros_like(z_tree)
        visited = set()
        node_neighbors = defaultdict(list)
        curr_node = -1
        tree_pred_loss = 0
        # print("start")
        while queue and iter < max_iter:
            length = len(queue)
            for j in range(length):
                iter += 1
                node = queue.popleft()
                if node[0] != curr_node:
                    print("Huge error!")
                if node[0] != -1:
                    # print("update h_curr")
                    # print(curr_x.size())
                    # print(curr_edge_index)
                    # print(curr_edge_weight)
                    with torch.no_grad():
                        h_curr = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, curr_edge_weight, return_embedding=True)[node[0]]
                    # batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                    # h_curr = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, batch=batch, return_embedding=True)
                # with torch.no_grad():
                topo_pred = self.pred_node_topo(self.linear_topo(z_tree.add(h_curr)))
                # print(f"h_curr: {h_curr}")
                # print(topo_pred.softmax(1))
                
                if torch.argmax(topo_pred) == 1 or node[0] == -1:                   
                    
                    label_pred = self.pred_node_label(self.linear_label(z_tree.add(h_curr)))
                    if node[0] == -1:
                        softmax_label = self.straight_through_gumbel_softmax(label_pred, temperature=0.1, first_node=True)
                    else:
                        softmax_label = self.straight_through_gumbel_softmax(label_pred, temperature=0.1)
                    if node[0] == -1:
                        curr_x = torch.cat((curr_x, softmax_label), dim=0)
                        motif = self.id_motif[curr_x[-1].argmax().item()]
                        curr_mol = sanitize_mol(get_mol(motif, self.data_name), self.add_H)
                        batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                        with torch.no_grad():
                            tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, curr_edge_weight, batch=batch, return_embedding=True)
                        pred = self.model(tree_emb, classifier=True)
                        tree_pred_loss += self.criterion(pred, torch.tensor([self.label], device=self.device))
                    else:
                        if test:
                            # print("Test!")
                            values, indices = torch.topk(label_pred, 5)
                            # values, indices = torch.topk(softmax_label, 20)
                            # print(values)
                            # print(stop)
                            selected_motif = None
                            for i in range(len(indices[0])):
                                motif = self.id_motif[indices[0][i].item()]
                                mol = sanitize_mol(get_mol(motif, self.data_name), self.add_H)
                                site_pair = can_assemble(curr_mol, mol)
                                if site_pair:
                                    new_x = torch.nn.functional.one_hot(indices[0][i].view(-1), num_classes=len(self.id_motif))
                                    curr_x = torch.cat((curr_x, new_x), dim=0)
                                    curr_mol = mol
                                    selected_motif = motif
                                    curr_edge_index = torch.cat((curr_edge_index, torch.tensor([[node[0], node[1]], [node[1], node[0]]], dtype=torch.long, device=self.device)), dim=1)
                                    topo_prob = topo_pred.softmax(1)[0,1]
                                    edge_weight = self.gumbel_softmax_edge_weight(topo_prob)
                                    curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                                    curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                                    batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                                    with torch.no_grad():
                                        tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, batch=batch, return_embedding=True)
                                    pred = self.model(tree_emb, classifier=True)
                                    tree_pred_loss += self.criterion(pred, torch.tensor([self.label], device=self.device))
                                    break
                            # if not selected_motif:
                            #     print("Error in sampling")
                        else:
                            motif = self.id_motif[softmax_label.argmax().item()]
                            mol = sanitize_mol(get_mol(motif, self.data_name), self.add_H)
                            # site_pair = can_assemble(curr_mol, mol)
                            # if site_pair:
                            curr_x = torch.cat((curr_x, softmax_label), dim=0)
                            curr_mol = mol
                            curr_edge_index = torch.cat((curr_edge_index, torch.tensor([[node[0], node[1]], [node[1], node[0]]], dtype=torch.long, device=self.device)), dim=1)
                            topo_prob = topo_pred.softmax(1)[0,1]
                            edge_weight = self.gumbel_softmax_edge_weight(topo_prob)
                            curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                            curr_edge_weight = torch.cat((curr_edge_weight, edge_weight.view(-1)), dim=0)
                            batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
                            with torch.no_grad():
                                tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, curr_edge_weight, batch=batch, return_embedding=True)
                            pred = self.model(tree_emb, classifier=True)
                            tree_pred_loss += self.criterion(pred, torch.tensor([self.label], device=self.device))
                        # else:
                        #     print("Error in sampling")
                    
                    if curr_node == -1:
                        queue.append((node[1], curr_x.shape[0]))
                        curr_node += 1
                    else:
                        queue.append((node[0], curr_x.shape[0]))
                    node_neighbors[node[1]].append(curr_x.shape[0] - 1)
                elif torch.argmax(topo_pred) == 0:
                    # if node[0] == -1:
                    curr_node += 1
                    if curr_node < curr_x.shape[0]:
                        # print("------------->")
                        # print(curr_node)
                        # print(curr_x)
                        # print(curr_edge_index)
                        visited.add(node[0])
                        queue.append((curr_node, curr_x.shape[0]))
                        # print(f"Queue: {queue}")
                    else:
                        break
        # if iter >= max_iter:
        #     print("Exceed max_iter!")
        batch = torch.zeros(curr_x.shape[0], dtype=torch.long).to(self.device)
        # print(f"tree_x: {curr_x.size()}")
        
        with torch.no_grad():
            tree_emb = self.T_encoder(torch.matmul(curr_x, self.motif_embedding), curr_edge_index, curr_edge_weight, batch=batch, return_embedding=True)
        pred = self.model(tree_emb, classifier=True)
        # print(pred.size())
        tree_pred_loss = self.criterion(pred, torch.tensor([self.label], device=self.device))
        acc_correct, acc_wrong = 0, 0
        if pred.argmax() == self.label:
            acc_correct = 1
        else:
            acc_wrong = 1
        # print(curr_x.size())
        # print(torch.argmax(curr_x, dim=1).view(-1, 1))
        # print(stop)
        new_tree = Data(x=torch.argmax(curr_x, dim=1).view(-1, 1), edge_index=curr_edge_index)
        # print(f"Generated Tree: {new_tree.edge_index}")
        # if new_tree.x.size(0) == 1:
        #     return self.sample_tree(z_tree, max_iter, test)

        return tree_pred_loss, acc_correct, acc_wrong, new_tree, pred.softmax(1)[0][self.label].item()
            
    
    def encode_graph(self, graph):
        G_emb = self.model(graph.x, graph.edge_index, graph.batch, return_embedding=True)
        return G_emb

    def decode_graph(self, tree):
        queue = deque([])
        # print("Start decoding graph")
        # print(f"edge_index: {tree.edge_index}")
        # print(f"number of nodes: {tree.x.size(0)}")

        # odd_edges = tree.edge_index[:, 0::2]
        # scores_0 = self.node_scores[tree.x[odd_edges[0]].view(-1)]
        # scores_1 = self.node_scores[tree.x[odd_edges[1]].view(-1)]
        # max_scores = torch.max(scores_0, scores_1)
        # # if scores_1 > scores_0, swap corresponding edges indices in odd_edges
        # swap_indices = scores_1 > scores_0
        # odd_edges[:, swap_indices] = odd_edges.flip(0)[:, swap_indices]
        # # print(f"max_scores: {max_scores}")
        # sorted_indices = torch.argsort(max_scores, descending=True)
        # sorted_edge_index = odd_edges[:, sorted_indices]
        # print(f"sorted_edge_index: {sorted_edge_index}")
        # for i in range(sorted_edge_index.size(1)):
        #     queue.append((sorted_edge_index[0, i].item(), sorted_edge_index[1, i].item()))
        
        for i in range(0, tree.edge_index.shape[1], 2):
        # for i in range(tree.edge_index.shape[1]-1, -1, -2):
            queue.append((tree.edge_index[0, i].item(), tree.edge_index[1, i].item()))
        curr_mol = None
        visited = set()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
        # bond_types = [Chem.rdchem.BondType.SINGLE]
        # print("Start decoding graph!")
        if not queue:
            motif = self.id_motif[tree.x[0].item()]
            smiles = sanitize_smiles(motif, self.add_H)
            mol = sanitize_mol(get_mol(smiles, self.data_name), self.add_H)
            data = to_tudataset(mol, self.data_name)
            data.to(self.device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
            pred = self.model(data.x, data.edge_index, batch)
            pred = F.softmax(pred, dim=1)
            embedding = self.model(data.x, data.edge_index, batch, return_embedding=True)
            return smiles, pred[0, self.label].item(), embedding
        while queue:
            node1, node2 = queue.popleft()
            
            motif2 = self.id_motif[tree.x[node2].item()]
            motif2 = sanitize_mol(get_mol(motif2, self.data_name), self.add_H)
            curr_cand = []
            if curr_mol and not node1 in visited and not node2 in visited:
                print(node1, node2)
                print(visited)
                print(curr_mol)
                print(queue)
                print("Error!")
            elif not curr_mol:
                motif1 = self.id_motif[tree.x[node1].item()]
                motif1 = sanitize_mol(get_mol(motif1, self.data_name), self.add_H)
                # Create a dictionary store the original motif index of each node in motif1
                atom_motif_id_mapping = {}
                for atom in motif1.GetAtoms():
                    atom_motif_id_mapping[atom.GetIdx()] = node1

                num_atoms = len(atom_motif_id_mapping)
                # Add node in motif2 into atom_motif_id_mapping
                for atom in motif2.GetAtoms():
                    atom_motif_id_mapping[atom.GetIdx()+num_atoms] = node2

                atom_pairs = [(i, j) for i in range(motif1.GetNumAtoms()) for j in range(motif2.GetNumAtoms())]
                for atom1, atom2 in atom_pairs:
                    for bond_type in bond_types:
                        cand = self.combine_motifs(motif1, motif2, atom1, atom2, bond_type)
                        if cand:
                            curr_cand.append(cand)
            else:
                atom_in_motif1 = []
                num_atoms = len(atom_motif_id_mapping)
                # Add node in motif2 into atom_motif_id_mapping
                for atom in motif2.GetAtoms():
                    atom_motif_id_mapping[atom.GetIdx()+num_atoms] = node2
                
                for key, value in atom_motif_id_mapping.items():
                    if value == node1:
                        atom_in_motif1.append(key)

                atom_pairs = [(atom_in_motif1[i], j) for i in range(len(atom_in_motif1)) for j in range(motif2.GetNumAtoms())]
                for atom1, atom2 in atom_pairs:
                    for bond_type in bond_types:
                        cand = self.combine_motifs(curr_mol, motif2, atom1, atom2, bond_type)
                        if cand:
                            curr_cand.append(cand)
            if not curr_cand:
                # print("Error in assemble")
                break
            max_score = 0.0
            # print("hhh")
            for cand in curr_cand:
                data = to_tudataset(cand, self.data_name)
                data.to(self.device)
                batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
                pred = self.model(data.x, data.edge_index, batch)
                # softmax pred and get the probability of the label
                pred = F.softmax(pred, dim=1)
                # print(pred)
                # print(stop)
                if pred[0, self.label] > max_score:
                    max_score = pred[0, self.label]
                    curr_mol = cand
            # print(f"max_score: {max_score}")
            visited.add(node1)
            visited.add(node2)
        try:
            smiles = sanitize_smiles(get_smiles(curr_mol), self.add_H)
        except:
            # print(curr_cand)
            return None, 0, None
        # smiles = sanitize_smiles(get_smiles(curr_mol), self.add_H)
        data = to_tudataset(curr_mol, self.data_name)
        data.to(self.device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
        pred = self.model(data.x, data.edge_index, batch)
        pred = F.softmax(pred, dim=1)
        embedding = self.model(data.x, data.edge_index, batch, return_embedding=True)
        # print(smiles)
        # print(stop)
        # print(pred[0, self.label].item())
        return smiles, pred[0, self.label].item(), embedding
    
    def combine_motifs(self, motif1, motif2, atom_idx1, atom_idx2, bond_type):
        combined_mol = Chem.CombineMols(motif1, motif2)
        editable_mol = Chem.EditableMol(combined_mol)
        
        # Add a bond between specified atom indices from each molecule
        num_atoms1 = motif1.GetNumAtoms()
        editable_mol.AddBond(atom_idx1, num_atoms1 + atom_idx2, bond_type)
        
        # Attempt to sanitize the molecule, returns None if unsuccessful
        new_mol = editable_mol.GetMol()

        return sanitize_mol(new_mol, self.add_H)
    
    def straight_through_gumbel_softmax(self, logits, temperature=0.5, first_node=False):
        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
        # print(self.mask.bool().size())
        if first_node:
            y_soft = torch.softmax(torch.where(self.first_node_mask.bool(), (logits + gumbels) / temperature, torch.tensor(float('-inf'))), dim=-1)
        else:
            # y_soft = torch.softmax((logits + gumbels) / temperature, dim=-1)
            y_soft = torch.softmax(torch.where(self.motif_mask.bool(), (logits + gumbels) / temperature, torch.tensor(float('-inf'))), dim=-1)

            # negative_infinity = torch.tensor(float('-inf'), device=logits.device)  # Ensure tensor is on the same device as logits
            # adjusted_logits = (logits + gumbels) / temperature
            # adjusted_logits = torch.where(self.motif_mask, adjusted_logits, negative_infinity)
            # y_soft = torch.softmax(adjusted_logits, dim=-1)
        y_hard = torch.zeros_like(logits).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)

        # Straight-through estimator trick
        y = y_hard - y_soft.detach() + y_soft
        # y = y_soft
        return y
    
    def gumbel_softmax_edge_weight(self, prob):
        y_soft = prob*2
        y_hard = torch.ones_like(prob)
        y = y_hard - y_soft.detach() + y_soft
        return y
    
    def get_motif_mask(self):
        mask = torch.zeros((1, len(self.motif_id)), dtype=torch.long).to(self.device)
        mask_pred = torch.zeros((1, len(self.motif_id)), dtype=torch.float).to(self.device)
        for i in range(self.motif_embedding.size(0)):
            emb = self.motif_embedding[i].unsqueeze(0)
            pred = self.model(emb, classifier=True)
            pred = F.softmax(pred, dim=1)
            mask_pred[0][i] = pred[0, self.label]
            
            if pred[0, self.label] > 0.99:
                mask[0][i] = 1
        self.first_node_mask = mask.to(self.device)
        # self.mask_pred = mask_pred

        if os.path.exists("checkpoints/motif_selection/"+self.data_name+"_motif_"+str(self.label)+".pt"):
            selected_motif = torch.load("checkpoints/motif_selection/"+self.data_name+"_motif_"+str(self.label)+".pt")
        else:
            motif_filter(self.whole_dataset, self.data_name, self.smiles_set, self.model, 2, self.device)
            selected_motif = torch.load("checkpoints/motif_selection/"+self.data_name+"_motif_"+str(self.label)+".pt")
        # print(f"Selected_motif: {selected_motif}")
        mask = torch.zeros((1, len(self.motif_id)), dtype=torch.long).to(self.device)

        for i, smiles in enumerate(self.motif_id.keys()):
            if smiles in selected_motif:
                mask[0][self.motif_id[smiles]] = 1
        self.motif_mask = mask.bool()


    
    def train(self, epochs, batch_size, lr, max_iter, path_dict, t_encoder_path):
        # Train the MAGE model
        # For each graph use get_tree to get the tree
        self.get_motif_mask()
        print(torch.sum(self.first_node_mask))
        
        # load the pretrained T_encoder
        self.T_encoder.load_state_dict(torch.load(t_encoder_path))
        # fix parameters of T_encoder
        # for param in self.T_encoder.parameters():
        #     param.requires_grad = False
        
        # check if fix is ok
        for param in self.T_encoder.parameters():
            print(param.requires_grad)
        
        import matplotlib.pyplot as plt

        # Calculate the count distribution
        count_distribution = [tree.x.shape[0] for tree in self.trees]

        # Plot the count distribution
        plt.hist(count_distribution, bins=10)
        plt.xlabel('Tree Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Tree Length')
        plt.savefig('plot/distribution.pdf')
        plt.close()
        # print(stop)
        
        # self.get_motif_embedding()
        # Create the dataloader for the trees
        # tree_dataset = TreeDataset(trees)
        train_loader = DataLoader(self.trees, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        # Test the dataloader
        optimizer = torch.optim.Adam(list(self.T_encoder.parameters()) + list(self.pred_node_topo.parameters()) + list(self.pred_node_label.parameters()) + list(self.linear_topo.parameters()) + list(self.linear_label.parameters()) + list(self.T_mean.parameters()) + list(self.T_var.parameters()), lr=lr)
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            total_topo_loss = 0
            total_label_loss = 0
            total_pred_loss = 0
            total_kl_loss = 0
            total_emb_loss = 0
            total_acc = 0
            total_count = 0
            total_graph_pred = 0
            total_graph_sample_count = 0
            for t_data, g_data in tqdm(train_loader):
                # print("------------------------->")
                # for name, param in self.T_encoder.named_parameters():
                #     print('Name:', name)
                #     print('Type:', type(param))
                #     print('Size:', param.size())
                #     print('Values:', param.data)
                t_data.to(self.device)
                g_data.to(self.device)
                
                # graph_emb = self.encode_graph(g_data)
                # z_graph, kl_graph = self.rsample(graph_emb, self.G_mean, self.G_var)
                with torch.no_grad():
                    tree_emb = self.encode_tree(t_data)
                z_tree, kl_tree = self.rsample(tree_emb, self.T_mean, self.T_var)
                
                # topo_loss, label_loss, emb_loss, pred_loss, acc, count, graph_pred, graph_sample_count = self.decode_tree(t_data, g_data, z_tree)
                # emb_loss, pred_loss, acc, count = self.decode_tree(t_data, g_data, z_tree)
                # topo_loss, label_loss, pred_loss, acc, count, count_positive, count_negative = self.decode_tree(t_data, g_data, z_tree, max_iter)
                topo_loss, label_loss, pred_loss, emb_loss, acc, count, count_positive, count_negative = self.decode_tree(t_data, g_data, z_tree, max_iter)
                # loss = 0.1*topo_loss + 0.2*label_loss + 0.1*kl_tree + 0.2*emb_loss + 0.4*pred_loss
                # loss = kl_tree + 5*emb_loss + 10*pred_loss
                # loss = topo_loss + label_loss + kl_tree + 10*pred_loss
                # loss = 0.3*pred_loss + 0.7*emb_loss
                # loss = emb_loss
                # loss = topo_loss + label_loss + kl_tree + 10*pred_loss + emb_loss
                # loss = topo_loss + kl_tree + 10*pred_loss + emb_loss
                loss = kl_tree + pred_loss + emb_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_topo_loss += topo_loss.item()
                total_label_loss += label_loss.item()
                total_kl_loss += kl_tree.item()
                if emb_loss != 0:
                    total_emb_loss += emb_loss.item()
                # print(pred_loss)
                total_pred_loss += pred_loss.item()
                total_acc += acc
                total_count += count
                # total_graph_pred += graph_pred
                # total_graph_sample_count += graph_sample_count
            # print(f'Epoch {epoch}, Loss: {total_loss}, Topo Loss: {total_topo_loss}, Label Loss: {total_label_loss}, KL Loss: {total_kl_loss}, Emb Loss: {total_emb_loss}, Pred Loss: {total_pred_loss}, Acc: {total_acc / total_count}, Acc_count: {total_acc}.')
            # print(f'Epoch {epoch}, Loss: {total_loss}, Emb Loss: {total_emb_loss}, Pred Loss: {total_pred_loss}, Acc: {total_acc / total_count}, Acc_count: {total_acc}.')
            # print(f'Epoch {epoch}, Loss: {total_loss}, Topo Loss: {total_topo_loss}, Label Loss: {total_label_loss}, KL Loss: {total_kl_loss}, Pred Loss: {total_pred_loss}, Acc: {total_acc / total_count}, Acc_count: {total_acc}.')
            print(f'Epoch {epoch}, Loss: {total_loss}, Topo Loss: {total_topo_loss}, Label Loss: {total_label_loss}, KL Loss: {total_kl_loss}, Emb Loss: {total_emb_loss}, Pred Loss: {total_pred_loss}, Acc: {total_acc / total_count}, Acc_count: {total_acc}.')
            if total_loss < best_loss:
                best_loss = total_loss
                self.save(path_dict)

    def load(self, path_dict):
        # Load the model from the path
        self.T_encoder.load_state_dict(torch.load(path_dict['T_encoder']))
        self.pred_node_topo.load_state_dict(torch.load(path_dict['pred_node_topo']))
        self.pred_node_label.load_state_dict(torch.load(path_dict['pred_node_label']))
        self.linear_topo.load_state_dict(torch.load(path_dict['linear_topo']))
        self.linear_label.load_state_dict(torch.load(path_dict['linear_label']))
        self.T_mean.load_state_dict(torch.load(path_dict['T_mean']))
        self.T_var.load_state_dict(torch.load(path_dict['T_var']))

        for param in self.T_encoder.parameters():
            print(param.requires_grad)
        self.T_encoder.eval()
        self.pred_node_topo.eval()
        self.pred_node_label.eval()
        self.linear_topo.eval()
        self.linear_label.eval()
        self.T_mean.eval()
        self.T_var.eval()

    def save(self, path_dict):
        # Save the model to the path
        # torch.save(self.T_encoder.state_dict(), path_dict['T_encoder'])
        torch.save(self.pred_node_topo.state_dict(), path_dict['pred_node_topo'])
        torch.save(self.pred_node_label.state_dict(), path_dict['pred_node_label'])
        torch.save(self.linear_topo.state_dict(), path_dict['linear_topo'])
        torch.save(self.linear_label.state_dict(), path_dict['linear_label'])
        torch.save(self.T_mean.state_dict(), path_dict['T_mean'])
        torch.save(self.T_var.state_dict(), path_dict['T_var'])

    def sample(self, num_samples, max_iter):
        # Sample explanations for the target GNN
        self.get_motif_mask()
        print(torch.sum(self.first_node_mask))
        print(stop)
        
        node_scores = []
        for motif in self.motif_id.keys():
            smiles = sanitize_smiles(motif, self.add_H)
            mol = sanitize_mol(get_mol(smiles, self.data_name), self.add_H)
            data = to_tudataset(mol, self.data_name)
            data.to(self.device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long).to(self.device)
            pred = self.model(data.x, data.edge_index, batch)
            pred = F.softmax(pred, dim=1)
            node_scores.append(pred[0, self.label].item())
        self.node_scores = torch.tensor(node_scores).to(self.device)
        
        output = []
        total_pred_prob = []
        count_invalid = 0
        tree_correct = 0
        tree_wrong = 0
        total_prob = 0
        count_single_node_tree = 0
        for _ in tqdm(range(num_samples)):
        # while len(output) < num_samples:
            z_tree = torch.randn(1, self.hidden_channels).to(self.device)
            tree_pred_loss, acc_correct, acc_wrong, new_tree, prob = self.sample_tree(z_tree, max_iter=max_iter, test=True)
            if new_tree.edge_index.size(1) == 0:
                count_single_node_tree += 1
                # continue
            # print(f"Tree prob: {prob}")
            tree_correct += acc_correct
            tree_wrong += acc_wrong
            if len(new_tree.x) == 0:
                print("Wrong tree")
                continue
            smiles, pred_prob, _ = self.decode_graph(new_tree)
            total_pred_prob.append(pred_prob)
            total_prob += prob
            output.append(smiles)
            if not smiles:
                count_invalid += 1
        print(f"Number of single node tree: {count_single_node_tree}")
        return output, total_pred_prob, count_invalid, total_prob / (tree_correct + tree_wrong)

    def test(self, batch_size, test_set):
        test_trees = [self.get_tree(graph, True) for graph in tqdm(test_set)]
        test_trees = [tree for tree in test_trees if tree]
        test_loader = DataLoader(test_trees, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        all_count = 0
        for t_data, g_data in tqdm(test_loader):
            t_data.to(self.device)
            g_data.to(self.device)
            
            graph_emb = self.encode_graph(g_data)
            # z_graph, kl_graph = self.rsample(graph_emb, self.G_mean, self.G_var)
            tree_emb = self.encode_tree(t_data)
            pred = self.model(tree_emb, classifier=True)
            pred.softmax(1)
            pred_graph = self.model(graph_emb, classifier=True)
            pred_graph.softmax(1)
            # Count how may same prediction between pred and pred_graph
            count = 0
            for i in range(pred.size(0)):
                print(pred[i], pred_graph[i])
                if torch.argmax(pred[i]) == torch.argmax(pred_graph[i]):
                    count += 1
            all_count += count
        print(f"Accuracy: {all_count / len(test_trees)}")
            # z_tree, kl_tree = self.rsample(tree_emb, self.T_mean, self.T_var)
            
            # topo_loss, label_loss, emb_loss, pred_loss, acc, count, graph_pred, graph_sample_count = self.decode_tree(t_data, g_data, z_tree)
        
    def train_t_encoder(self, epochs, lr, batch_size, save_path):
        # Train the teacher encoder
        # split the dataset into train and test
        train_trees, test_trees = train_test_split(self.trees, test_size=0.2, shuffle=True, random_state=42)
        train_loader = DataLoader(train_trees, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_trees, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        optimizer = torch.optim.Adam(self.T_encoder.parameters(), lr=lr)
        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            for t_data, g_data in tqdm(train_loader):
                t_data.to(self.device)
                g_data.to(self.device)
                tree_emb = self.encode_tree(t_data)
                graph_emb = self.encode_graph(g_data)
                loss = self.mse_loss(tree_emb, graph_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}')
            self.T_encoder.eval()
            total_loss = 0
            for t_data, g_data in tqdm(test_loader):
                t_data.to(self.device)
                g_data.to(self.device)
                tree_emb = self.encode_tree(t_data)
                graph_emb = self.encode_graph(g_data)
                loss = self.mse_loss(tree_emb, graph_emb)
                total_loss += loss.item()
            print(f'Test Loss: {total_loss/len(test_loader)}')
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.T_encoder.state_dict(), save_path)