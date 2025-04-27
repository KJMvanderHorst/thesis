import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from src.models.base_model import RRCNNDecomposer
from src.data.dataset import SignalDataset
from src.models.losses.combined_loss import compute_combined_loss
from src.models.losses.band_leakage_loss import band_leakage_loss

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5
EPOCHS = 20
loss_list = ['wavelet_coherence', 'mse']  # List of loss functions to use
LEARNING_RATE = 1e-3
N_COMPONENTS = 2  # Number of components for RRCNNDecomposer
DATA_PATH = "src/data/data_storage/composite_signals_20250427T205057.npz"  # Update with your dataset path
MODEL_SAVE_PATH = "models/rrcnn_decomposer_mock_test.pth"  # Specify a file name

def train():
    band_leakage_bool = False  # Set to True if you want to include band leakage loss
    # Load the full dataset
    full_dataset = SignalDataset(DATA_PATH, include_frequency_bands=True)  # Set include_frequency_bands to True

    # Split into train and test sets
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=0.1, random_state=42
    )

    # Create train and test datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = RRCNNDecomposer(n_components=N_COMPONENTS).to(DEVICE)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        # Iterate over batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            # Unpack the batch
            composite_signals, components, frequency_bands = batch

            # Move data to the device
            composite_signals = composite_signals.to(DEVICE)
            frequency_bands = frequency_bands.to(DEVICE)

            # Forward pass
            predicted_components = model(composite_signals)

            # Compute loss
            if not loss_list:
                loss = torch.tensor(0.0, requires_grad=True).to(DEVICE)  # Default to zero loss if loss_list is empty
            else:
                loss = compute_combined_loss(predicted_components, composite_signals, loss_list=loss_list)


            if(band_leakage_bool == True):
                band_leakage = band_leakage_loss(predicted_components, frequency_bands)
                loss = loss + band_leakage

            # Combine losses
            total_loss = loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += total_loss.item()

        # Log epoch loss
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()