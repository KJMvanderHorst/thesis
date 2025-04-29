import os
import torch
import hydra

from torch.optim import Adam
from tqdm import tqdm

from src.models.base_model import RRCNNDecomposer
from src.data.dataset import SignalDataset
from src.losses.combined_loss import compute_combined_loss
from src.training.prepare_data import prepare_data

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base="1.1", config_path="/Users/kaspervanderhorst/Desktop/thesis/src/conf", config_name="config")
def train(cfg):
    print("Training configuration:")
    print(cfg)
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define paths
    data_path = os.path.join(project_root, cfg.data_path)
    model_save_path = os.path.join(project_root, cfg.params.model_save_path)

    # Ensure the model save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Prepare data
    train_loader, test_loader = prepare_data(cfg, data_path=data_path)

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
                loss_weights=cfg.params.loss_weights,
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
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()