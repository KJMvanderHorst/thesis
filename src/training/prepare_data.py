import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import SignalDataset


def prepare_data(cfg, data_path):
    """Prepare datasets and DataLoaders."""
    # Resolve paths

    # Load the full dataset
    full_dataset = SignalDataset(data_path, cfg.include_frequency_bands)

    # Split into train and test sets
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Create train and test datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True, num_workers=cfg.params.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.params.batch_size, shuffle=False, num_workers=cfg.params.num_workers)

    return train_loader, test_loader