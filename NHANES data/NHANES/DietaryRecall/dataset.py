# dataset.py
from torch.utils.data import Dataset
import torch


# Define a simple wrapper dataset class to hold a list of samples.

class CustomHyperedgeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        return torch.tensor(sample, dtype=torch.long), torch.tensor(label, dtype=torch.float)
