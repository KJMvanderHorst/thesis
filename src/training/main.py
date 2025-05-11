import os
import hydra
import wandb

from src.training.train_model import train, predict_test_set

# Dynamically determine the relative config path
CONFIG_PATH = os.path.relpath(os.path.join(os.path.dirname(__file__), "../conf"))

@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(cfg):
    """
    Main function to run the training process.
    """
    # Call the train function with the provided configuration
    print("main function called")

    train(cfg)
    mse = predict_test_set(cfg)
    print(f"Mean Squared Error: {mse}")
    return mse

def run_sweep():
    """
    Function to initialize and run a W&B sweep.
    """
    # Path to the sweep configuration file
    sweep_config_path = os.path.join(os.path.dirname(__file__), "../conf/sweep.yaml")

    # Load the sweep configuration from the YAML file
    with open(sweep_config_path, "r") as file:
        sweep_config = file.read()

    # Initialize the W&B sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="thesis-signal-decomposition")

    # Define the training function for the sweep
    def sweep_train():
        # Initialize W&B run
        with wandb.init() as run:
            # Get the W&B config
            config = wandb.config

            # Convert W&B config to Hydra-like structure
            cfg = {
                "params": {
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "loss_weights": {
                        "wavelet_coherence": config.wavelet_coherence,
                        "reconstruction": config.reconstruction,
                        "band_leakage": config.band_leakage,
                    },
                    "model_save_path": config.model_save_path,
                }
            }

            # Call the train function with the W&B config
            train(cfg)

    # Run the sweep agent
    wandb.agent(sweep_id, function=sweep_train)

if __name__ == "__main__":
    # Uncomment the function you want to run
    # main()
    run_sweep()