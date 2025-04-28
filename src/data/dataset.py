import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.signals_bandwidth import compute_bandwidths

class SignalDataset(Dataset):
    """
    A dataset class to load and access composite signals and their components from an NPZ file.

    Args:
        dataset_path (str): Path to the NPZ file containing the dataset.
    """

    def __init__(self, dataset_path, include_frequency_bands=False):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

        # Load the dataset
        self.data = np.load(dataset_path)
        self.composite_signals = torch.tensor(self.data["composite_signals"]).float()  # Shape: [num_samples, time_steps]
        self.components = torch.tensor(self.data["components"]).float()  # Shape: [num_samples, time_steps, num_components]
        if (include_frequency_bands):
            # Call the function to calculate frequency bands from signals_bandwidth
            self.frequency_bands = compute_bandwidths(self.composite_signals)
            self.frequency_bands = torch.tensor(self.frequency_bands).float()  # Convert to tensor
        else:
            self.frequency_bands = None
        #add frequency bands here

    def __len__(self):
        """
        Returns the number of composite signals in the dataset.
        """
        return len(self.composite_signals)

    def __getitem__(self, idx):
        composite_signal = self.composite_signals[idx]
        components = self.components[idx]
        if self.frequency_bands is not None:
            frequency_bands = self.frequency_bands[idx]
        else:
            frequency_bands = None

        # Add a channel dimension to the composite signal
        composite_signal = np.expand_dims(composite_signal, axis=0)  # Shape: [1, signal_length]

        return composite_signal, components, frequency_bands