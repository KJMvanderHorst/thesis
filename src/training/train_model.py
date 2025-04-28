import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import hydra
import os

from src.models.base_model import RRCNNDecomposer
from src.data.dataset import SignalDataset
from src.losses.combined_loss import compute_combined_loss
from src.losses.band_leakage_loss import band_leakage_loss

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base="1.1", config_path="/Users/kaspervanderhorst/Desktop/thesis/src/conf", config_name="config")
def train(cfg):
    # Load the full dataset
    full_dataset = SignalDataset(cfg.data_path, cfg.include_frequency_bands)

    # Split into train and test sets
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Create train and test datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.params.batch_size, shuffle=False)

    # Initialize model
    model = RRCNNDecomposer(n_components=cfg.params.n_components).to(DEVICE)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=cfg.params.learning_rate)

    # Training loop
    for epoch in range(cfg.params.epochs):
        model.train()
        epoch_loss = 0.0

        # Iterate over batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.params.epochs}")):
            # Unpack the batch
            composite_signals, components, frequency_bands = batch

            # Move data to the device
            composite_signals = composite_signals.to(DEVICE)
            frequency_bands = frequency_bands.to(DEVICE)

            # Forward pass
            predicted_components = model(composite_signals)

            # Compute loss
            loss = compute_combined_loss(
                predicted_components,
                loss_list=cfg.params.loss_list,
                frequency_bands=frequency_bands,
                input_signal=composite_signals,
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Log epoch loss
        print(f"Epoch {epoch + 1}/{cfg.params.epochs}, Loss: {epoch_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), cfg.params.model_save_path)
    print(f"Model saved to {cfg.params.model_save_path}")

if __name__ == "__main__":
    train()