import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.models.base_model import RRCNNDecomposer
from src.data.dataset import SignalDataset
from src.models.losses.combined_loss import compute_combined_loss

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
N_COMPONENTS = 2  # Number of components for RRCNNDecomposer
DATA_PATH = "src/data/data_storage/composite_signals_20250422T181928.npz"  # Update with your dataset path
MODEL_SAVE_PATH = "models"  # Update with your desired save path

def train():
    # Load dataset
    train_dataset = SignalDataset(DATA_PATH, train=True)  # Assuming your dataset has a train/test split
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = RRCNNDecomposer(n_components=N_COMPONENTS).to(DEVICE)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        # Iterate over batches
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            signals, targets = batch  # Assuming your dataset returns (input_signal, target_signal)
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            components, reconstruction = model(signals)

            # Compute loss
            loss = compute_combined_loss(components, reconstruction, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Log epoch loss
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()