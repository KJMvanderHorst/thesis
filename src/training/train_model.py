import os
import sys
import torch
import hydra
import wandb

from torch.optim import Adam
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from src.models.base_model import RRCNNDecomposer
from src.models.base_model import MultiScaleRRCNNDecomposer
from src.losses.combined_loss import compute_combined_loss
from src.training.prepare_data import prepare_data
from src.losses.band_leakage_loss import save_training_run_losses, clear_training_run_losses

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dynamically determine the relative config path
CONFIG_PATH = os.path.relpath(os.path.join(os.path.dirname(__file__), "../conf"))

class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    
    def average(self):
        return self.sum / self.count if self.count != 0 else 0

class AverageMeterDict: 
    def __init__(self):
        self.meters = {}
    
    def update(self, values: dict, n=1):
        for k, v in values.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v, n)

    def average(self):
        return {k: meter.average() for k, meter in self.meters.items()}

def save_checkpoint(model: torch.nn.Module, name: str, checkpoint_dir: str, epoch: int, metrics: dict):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    artifact = wandb.Artifact(
        name = name,
        type = "model",
        metadata = {
            "epoch": epoch + 1,
            **metrics
        }
    )

    artifact.add_file(checkpoint_path)
    aliases = ["latest"]
    wandb.log_artifact(artifact, aliases=aliases)

def log_to_wandb(train_metrics: dict, test_metrics: dict, epoch: int):
    train_results = {f"train/{k}": v for k, v in train_metrics.items()}
    test_results = {f"test/{k}": v for k, v in test_metrics.items()}
    wandb.log({
        "epoch": epoch + 1,
        **train_results,
        **test_results
    })


def train_loop(model, train_loader, optimizer, cfg, epoch):
    """
    Training loop for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        cfg (DictConfig): Configuration object containing training parameters.

    """

    model.train()

    metrics = AverageMeterDict()

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
        loss, per_loss_vals = compute_combined_loss(
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

        metrics_dict = {
            "loss": loss,
            **per_loss_vals
        }
        metrics.update(metrics_dict)

    return metrics.average()


def test_loop(model, test_loader, cfg):
    """
    Testing loop for one epoch.
    
    Args:
        model (torch.nn.Module): The model to test.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        cfg (DictConfig): Configuration object containing testing parameters.
    """

    metrics = AverageMeterDict()

    model.eval()

    # Iterate over batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Unpack the batch
            composite_signals, components, frequency_bands = batch

            # Move data to the device
            composite_signals = composite_signals.to(DEVICE)
            frequency_bands = frequency_bands.to(DEVICE)

            # Forward pass
            predicted_components = model(composite_signals)

            # Compute loss
            loss, per_loss_vals = compute_combined_loss(
                predicted_components,
                loss_list=cfg.params.loss_list,
                frequency_bands=frequency_bands,
                input_signal=composite_signals,
                loss_weights=cfg.params.loss_weights,
                ground_truth=components,
            )

            metrics_dict = {
                "loss": loss,
                **per_loss_vals
            }
            metrics.update(metrics_dict)

    return metrics.average()

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

    # Setup WandB
    if OmegaConf.select(cfg, "wandb.enabled"):
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.run.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            group=cfg.wandb.group
        )

    # Prepare data
    train_loader, test_loader = prepare_data(cfg, data_path=data_path)

    # Initialize model
    model = MultiScaleRRCNNDecomposer(n_components=cfg.params.n_components).to(DEVICE)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=cfg.params.learning_rate)

    # Clear previous training run losses
    clear_training_run_losses()

    # Training loop
    for epoch in range(cfg.params.epochs):
        model.train()
        epoch_loss = 0.0

        # Run training loop
        train_results = train_loop(model, train_loader, optimizer, cfg, epoch)
        test_results = test_loop(model, test_loader, cfg)

        # Log and save checkpoint using wandb
        if OmegaConf.select(cfg, "wandb.enabled"):
            # Log metrics
            log_to_wandb(train_results, test_results, epoch)

            # Save checkpoint
            save_checkpoint(
                model, 
                cfg.run.experiment_name,
                model_save_path,
                epoch,
                metrics={
                    **train_results,
                    **test_results
                }
            )

        # Log epoch loss
        print(f"Epoch {epoch + 1}/{cfg.params.epochs}, Loss: {epoch_loss / len(train_loader)}")

    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save training run losses
    save_training_run_losses()

if __name__ == "__main__":
    train()
