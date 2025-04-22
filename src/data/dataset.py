import torch
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self, composite_signals, components):
        """
        Args:
            composite_signals (list or ndarray): The composite signals (e.g., mixed signals).
            components (list or ndarray): The individual components of the signals.
        """
        self.composite_signals = torch.tensor(composite_signals).float()  # Shape: [num_samples, time_steps]
        self.components = torch.tensor(components).float()  # Shape: [num_samples, time_steps, num_components]

    def __len__(self):
        return len(self.composite_signals)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (composite_signal, components) for the given index.
        """
        return self.composite_signals[idx], self.components[idx]
