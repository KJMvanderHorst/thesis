import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.signals_bandwidth import compute_bandwidths
import pickle

class SignalDataset(Dataset):
    """
    A dataset class to load and access composite signals and their components from an NPZ file.

    Args:
        dataset_path (str): Path to the NPZ file containing the dataset.
    """

    def __init__(self, dataset_path, include_frequency_bands=False):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

        # Load the dataset and do NOT keep the npz file object as an attribute
        with np.load(dataset_path) as data:
            self.composite_signals = torch.tensor(data["composite_signals"]).float()  # Shape: [num_samples, time_steps]
            self.components = torch.tensor(data["components"]).float()  # Shape: [num_samples, time_steps, num_components]
            
        # Handle frequency bands
        self.frequency_bands = None
        if include_frequency_bands:
            precomputed_path = dataset_path.replace(".npz", "_freq_bands.pkl")
            if os.path.exists(precomputed_path):
                # Load precomputed frequency bands
                with open(precomputed_path, "rb") as f:
                    self.frequency_bands = pickle.load(f)
                self.frequency_bands = torch.tensor(self.frequency_bands).float()
            else:
                # Compute frequency bands and save them
                self.frequency_bands = compute_bandwidths(self.composite_signals)
                with open(precomputed_path, "wb") as f:
                    pickle.dump(self.frequency_bands, f)
                self.frequency_bands = torch.tensor(self.frequency_bands).float()

    def __len__(self):
        """
        Returns the number of composite signals in the dataset.
        """
        return len(self.composite_signals)

    def __getitem__(self, idx):

        composite_signal = self.composite_signals[idx].unsqueeze(0)  # [1, signal_length]

        components = self.components[idx]
        if self.frequency_bands is not None:
            frequency_bands = self.frequency_bands[idx]
        else:
            frequency_bands = torch.zeros(1)

        return composite_signal, components, frequency_bands