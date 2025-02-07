# prepare_dataset.py
# -*- coding: utf-8 -*-
import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from dataset import CustomHyperedgeDataset

import torch

# Import functions from hypergraph.py
from hypergraph import (
    build_meal_hypergraph_with_keys,
    build_hypergraph_and_edge_features,
    load_demographic_data,
    build_pyg_data_from_hypergraph,
    add_nutrition_nodes
)

#############################################
# Load data and build the graph (Parts 1¨C6)
#############################################

# Set file paths (update these paths as needed)
folder_path = "/home/jwu576/F+N/NHANES data/NHANES/DietaryRecall"
demo_folder = "/home/jwu576/F+N/NHANES data/NHANES/demographics"
demo_suffix = "_A"

# Build hypergraph
meal_dict = build_meal_hypergraph_with_keys(folder_path)
hypergraph, edge_keys = build_hypergraph_and_edge_features(meal_dict)
print(f"Hypergraph built with {len(hypergraph.nodes)} nodes and {len(hypergraph.edges)} edges.")

# Load demographic data and build the PyG mixed graph
demo_data = load_demographic_data(demo_folder, demo_suffix)
pyg_data = build_pyg_data_from_hypergraph(hypergraph, demo_data)

# Optionally add nutrition nodes
pyg_data = add_nutrition_nodes(pyg_data, folder_path)
print(f"PyG graph has {pyg_data.num_nodes} nodes after adding nutrition nodes.")

#############################################
# Generate Hyperedge Prediction Samples
#############################################

class HyperedgePredictionDataset(Dataset):
    def __init__(self, hypergraph, node_mapping, num_negatives=1):
        self.positive_samples = []
        self.negative_samples = []
        self.samples = []
        
        for edge_id, nodes in hypergraph.incidence_dict.items():
            pos_sample = [node_mapping[node] for node in nodes if node in node_mapping]
            if len(pos_sample) < 2:
                continue
            self.positive_samples.append(pos_sample)

            # Generate negative samples
            for _ in range(num_negatives):
                neg_sample = pos_sample.copy()
                replace_idx = random.randint(0, len(neg_sample) - 1)
                while True:
                    new_node = random.randint(0, len(node_mapping) - 1)
                    if new_node not in neg_sample:
                        break
                neg_sample[replace_idx] = new_node
                self.negative_samples.append(neg_sample)
        
        # Combine positive and negative samples; label 1 for positive, 0 for negative.
        self.samples = [(sample, 1) for sample in self.positive_samples] + \
                       [(sample, 0) for sample in self.negative_samples]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample_tensor = torch.tensor(sample, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return sample_tensor, label_tensor

# complete sample set
num_negatives = 1
full_dataset_obj = HyperedgePredictionDataset(hypergraph, pyg_data.node_mapping, num_negatives=num_negatives)
all_samples = full_dataset_obj.samples  # list of (sample, label)

#  50% 25% 25%
train_samples, temp_samples = train_test_split(all_samples, test_size=0.5, random_state=42)
val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

print(f"Total samples: {len(all_samples)}")
print(f"Train samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")
print(f"Test samples: {len(test_samples)}")

# 
train_dataset = CustomHyperedgeDataset(train_samples)
val_dataset   = CustomHyperedgeDataset(val_samples)
test_dataset  = CustomHyperedgeDataset(test_samples)

# 
def collate_fn(padding_value):
    def inner_collate(batch):
        samples, labels = zip(*batch)
        lengths = [s.size(0) for s in samples]
        max_len = max(lengths)
        
        padded_samples = []
        for s in samples:
            pad_size = max_len - s.size(0)
            padded = torch.cat([s, torch.full((pad_size,), padding_value, dtype=torch.long)]) if pad_size > 0 else s
            padded_samples.append(padded.unsqueeze(0))
        
        padded_samples = torch.cat(padded_samples, dim=0)
        labels_tensor = torch.stack(labels)
        return padded_samples, lengths, labels_tensor
    return inner_collate

# 
torch.save(train_dataset, "train_dataset.pt")
torch.save(val_dataset, "val_dataset.pt")
torch.save(test_dataset, "test_dataset.pt")
print("Datasets saved successfully.")

