import os
import sys
from sklearn.metrics import mean_squared_error
import torch
import hydra
import numpy as np

from torch.optim import Adam
from tqdm import tqdm

from src.models.base_model import RRCNNDecomposer
from src.models.base_model import MultiScaleRRCNNDecomposer
from src.losses.combined_loss import compute_combined_loss
from src.training.prepare_data import prepare_data
from src.losses.band_leakage_loss import save_training_run_losses, clear_training_run_losses

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    "base": RRCNNDecomposer,
    "multi_scale": MultiScaleRRCNNDecomposer,
}

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
    train_loader, _ = prepare_data(cfg, data_path=data_path)

    # Initialize model
    kernel_sizes = get_kernel_sizes(cfg, cfg.model_type)
    model = model_dict[cfg.model_type](
        n_components=cfg.n_components,
        kernel_sizes=kernel_sizes
    ).to(DEVICE)
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=cfg.params.learning_rate)

    # Clear previous training run losses
    clear_training_run_losses()

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

    # Save training run losses
    save_training_run_losses()

def get_kernel_sizes(cfg, model_type):
    for model in cfg.models:
        if model_type in model:
            return model[model_type][0]['kernel_sizes']
    raise ValueError(f"Model type '{model_type}' not found in configuration.")

def predict_test_set(cfg):
    """
    Function to predict the test set using the trained model and store the outputs.
    """

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define paths
    data_path = os.path.join(project_root, cfg.data_path)
    model_save_path = os.path.join(project_root, cfg.params.model_save_path)
    predictions_save_path = os.path.join(project_root, cfg.predictions_save_path)

    # Ensure the predictions save directory exists
    os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)

    # Prepare data
    _, test_loader = prepare_data(cfg, data_path=data_path)

    # Initialize model
    kernel_sizes = get_kernel_sizes(cfg, cfg.model_type)
    model = model_dict[cfg.model_type](
        n_components=cfg.n_components,
        kernel_sizes=kernel_sizes
    ).to(DEVICE)

    # Load the trained model
    model.load_state_dict(torch.load(model_save_path))

    # Set the model to evaluation mode
    model.eval()

    predictions = []

    # Predict on the test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting Test Set")):
            # Unpack the batch
            composite_signals, _, _ = batch

            # Move data to the device
            composite_signals = composite_signals.to(DEVICE)

            # Forward pass
            predicted_components = model(composite_signals)

            # Store predictions
            predictions.append(predicted_components.cpu().numpy())

    # Convert predictions and ground truths to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    # Save predictions to a file
    np.save(predictions_save_path, np.concatenate(predictions, axis=0))
    print(f"Predictions saved to {predictions_save_path}")
    # Compute evaluation metrics
    mse = mean_squared_error(ground_truths, predictions)
    return mse
