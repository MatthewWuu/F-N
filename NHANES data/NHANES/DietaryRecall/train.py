# train_model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from dataset import CustomHyperedgeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes = 1292025  
embed_dim = 256 

# Load datasets (these were saved in prepare_dataset.py)
train_dataset = torch.load("train_dataset.pt")
val_dataset   = torch.load("val_dataset.pt")
test_dataset  = torch.load("test_dataset.pt")

# Define a collate_fn factory function that takes a padding value
def collate_fn(padding_value):
    def inner_collate_fn(batch):
        samples, labels = zip(*batch)
        lengths = [s.size(0) for s in samples]
        max_len = max(lengths)
    
        padded_samples = []
        for s in samples:
            pad_size = max_len - s.size(0)
            # Use the passed padding_value to fill the gap
            if pad_size > 0:
                padded = torch.cat([s, torch.full((pad_size,), padding_value, dtype=torch.long)])
            else:
                padded = s
            padded_samples.append(padded.unsqueeze(0))
    
        padded_samples = torch.cat(padded_samples, dim=0)
        labels_tensor = torch.stack(labels)
        return padded_samples, lengths, labels_tensor
    return inner_collate_fn

# Generate the actual collate function using num_nodes as the padding value
data_collate_fn = collate_fn(padding_value=num_nodes)

# Create DataLoaders and set pin_memory=True to speed up data transfer to GPU
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collate_fn, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=data_collate_fn, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collate_fn, pin_memory=True)

# Define the HyperedgePredictor model
class HyperedgePredictor(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(HyperedgePredictor, self).__init__()
        self.num_nodes = num_nodes  # Save num_nodes as an instance attribute
        self.node_emb = nn.Embedding(num_nodes + 1, embed_dim)  # +1 for the padding index
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, padded_samples, lengths):
        emb = self.node_emb(padded_samples)
        # Use self.num_nodes to build a mask for padded elements
        mask = (padded_samples != self.num_nodes).unsqueeze(-1).float()
        emb = emb * mask
        
        sum_emb = emb.sum(dim=1)
        lengths_tensor = torch.tensor(lengths, dtype=torch.float).unsqueeze(1).to(emb.device)
        mean_emb = sum_emb / lengths_tensor
        scores = self.classifier(mean_emb).squeeze(-1)
        return scores

# Evaluate the model on a given dataloader
def evaluate_model(model, dataloader):
    model.eval()
    all_scores, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            padded_samples, lengths, labels = batch
            # Move batch data to the device
            padded_samples = padded_samples.to(device)
            labels = labels.to(device)
            outputs = model(padded_samples, lengths)
            all_scores.extend(outputs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    auc = roc_auc_score(all_labels, all_scores)
    predictions = [1 if s > 0.5 else 0 for s in all_scores]
    acc = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')

    print(f"AUC-ROC: {auc:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return auc, acc, precision, recall, f1

# Initialize model, loss function, and optimizer, and move the model to the device
predictor_model = HyperedgePredictor(num_nodes, embed_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(predictor_model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.0
    predictor_model.train()

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
        for batch in tepoch:
            padded_samples, lengths, labels = batch
            # Move batch data to the device
            padded_samples = padded_samples.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = predictor_model(padded_samples, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * padded_samples.size(0)
            tepoch.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
    print("Validation:")
    evaluate_model(predictor_model, val_loader)

print("Training completed.")
print("Final Test Evaluation:")
evaluate_model(predictor_model, test_loader)
