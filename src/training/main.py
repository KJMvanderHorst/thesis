import os
import hydra
import wandb
import yaml
from omegaconf import OmegaConf
from hydra.experimental import initialize, compose

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
    Function to initialize and run a W&B sweep using Hydra configuration.
    """
    # Path to the sweep configuration file
    sweep_config_path = os.path.join(os.path.dirname(__file__), "../conf/sweep.yaml")

    # Load the sweep configuration from the YAML file
    with open(sweep_config_path, "r") as file:
        sweep_config = yaml.safe_load(file)  # Parse YAML into a dictionary

    # Debugging sweep_config
    print("Sweep Config Type:", type(sweep_config))  # Should now be <class 'dict'>
    print("Sweep Config Content:", sweep_config)

    # Ensure it's a dictionary
    if not isinstance(sweep_config, dict):
        raise ValueError("sweep_config must be a dictionary.")
    
    # Initialize the W&B sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="thesis-signal-decomposition")

    # Define the training function for the sweep
    def sweep_train():
        # Initialize W&B run
        with wandb.init() as run:
            # Get the W&B config
            config = wandb.config

            # Dynamically merge W&B config into Hydra config
            with initialize(config_path="../conf", job_name="sweep"):
                hydra_cfg = compose(config_name="config")
                wandb_cfg = OmegaConf.create({"params": dict(config)})  # Convert W&B config to OmegaConf
                cfg = OmegaConf.merge(hydra_cfg, wandb_cfg)  # Merge Hydra config with W&B config

            # Call the train function with the merged config
            train(cfg)
    # Run the sweep agent
    wandb.agent(sweep_id, function=sweep_train)

if __name__ == "__main__":
    # Uncomment the function you want to run
    # main()
    run_sweep()