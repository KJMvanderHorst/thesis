import torch
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = torch.tensor(inputs).float()  # Shape: [num_samples, time_steps]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]
