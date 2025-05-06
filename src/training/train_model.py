import os
import sys
import torch
import hydra

from torch.optim import Adam
from tqdm import tqdm

from src.models.base_model import RRCNNDecomposer
from src.models.base_model import MultiScaleRRCNNDecomposer
from src.losses.combined_loss import compute_combined_loss
from src.training.prepare_data import prepare_data

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dynamically determine the relative config path
CONFIG_PATH = os.path.relpath(os.path.join(os.path.dirname(__file__), "../conf"))

@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def train(cfg):
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
        #TODO check indiviual losses per epoch

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
                ground_truth=components,
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            """for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    print(f"Gradient for {name}: {grad_mean}")
                    if grad_mean < 1e-6:
                        print(f"Warning: Gradient for {name} is very small!")
                    elif grad_mean > 1e2:
                        print(f"Warning: Gradient for {name} is very large!")
                else:
                    print(f"Gradient for {name}: No gradient")"""
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