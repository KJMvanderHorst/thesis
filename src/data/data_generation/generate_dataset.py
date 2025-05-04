import os
import json
import numpy as np
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from src.data.data_generation.composite_signal_generator import SyntheticSignalGenerator

"""
This module provides functionality to generate and store datasets of composite signals 
using a synthetic signal generator. The generated dataset is saved as a compressed NPZ 
file, and the parameters used for generation are stored in a JSON file.

Classes:
    SyntheticSignalGenerator: A class responsible for generating synthetic signals 
    with specified characteristics.

Functions:
    generate_and_store_dataset(generator, num_signals, num_components):
        Generate a dataset of composite signals and store it as an NPZ file along 
        with the parameters used.

Usage:
    To use this module, create an instance of the `SyntheticSignalGenerator` class 
    with the desired parameters, and call the `generate_and_store_dataset` function 
    with the generator instance, the number of signals to generate, and the number 
    of frequency segments per signal.
"""

def generate_and_store_dataset(generator, num_signals, num_components, output_folder):
    """
    Generate a dataset of composite signals and store it as an NPZ file along with the parameters used.

    Args:
        generator (SyntheticSignalGenerator): An instance of the SyntheticSignalGenerator class.
        num_signals (int): Number of composite signals to generate.
        num_components (int): Number of frequency segments for each signal.
        output_folder (str): Path to the folder where the dataset will be stored.

    Returns:
        None
    """
    # Resolve the output folder to an absolute path
    output_folder = to_absolute_path(output_folder)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique identifier (timestamp)
    unique_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Initialize lists to store the dataset
    composite_signals = []
    all_components = []
    all_signals = []

    for _ in range(num_signals):
        # Generate a composite signal and its components
        composite_signal, components, signals = generator.generate_signal(
            f0=generator.fmin,
            bandwidth=generator.fmax - generator.fmin,
            k=num_components
        )
        composite_signals.append(composite_signal)
        all_components.append(components)
        all_signals.append(signals)

    # Convert the dataset to NumPy arrays
    composite_signals = np.array(composite_signals)
    all_components = np.array(all_components)
    all_signals = np.array(all_signals)

    # Save the dataset as an NPZ file with the unique identifier
    dataset_filename = f"composite_signals_{unique_id}.npz"
    dataset_path = os.path.join(output_folder, dataset_filename)
    np.savez_compressed(dataset_path, composite_signals=composite_signals, components=all_components, signals=all_signals)

    # Save the parameters used for the entire dataset as a JSON file with the same unique identifier
    from omegaconf import OmegaConf

    # Convert the params to a standard Python dictionary
    params = {
        "num_signals": num_signals,
        "fmin": generator.fmin,
        "fmax": generator.fmax,
        "duration": generator.duration,
        "signal_types": OmegaConf.to_container(generator.signal_types, resolve=True),  # Convert ListConfig to list
        "intermittence": generator.intermittence,
        "overlap_factor": generator.overlap_factor,
        "overlap_std": generator.overlap_std,
        "num_segments": num_components,
        "dataset_file": dataset_filename  # Link the dataset file in the JSON
    }

    # Save the parameters as a JSON file
    params_filename = f"parameters_{unique_id}.json"
    params_path = os.path.join(output_folder, params_filename)
    with open(params_path, "w") as params_file:
        json.dump(params, params_file, indent=4)

    print(f"Dataset saved to {dataset_path}")
    print(f"Parameters saved to {params_path}")

# Resolve the relative path for the configuration directory
CONFIG_PATH = os.path.relpath(os.path.join(os.path.dirname(__file__), "../conf"))
@hydra.main(version_base="1.1", config_path="../../conf", config_name="data_generation")
def main(cfg: DictConfig):
    """
    Main function to generate and store a dataset using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.

    Returns:
        None
    """

    # Create an instance of the signal generator with parameters from the config
    generator = SyntheticSignalGenerator(cfg)

    # Call the dataset generation function
    generate_and_store_dataset(
        generator=generator,
        num_signals=cfg.dataset.num_signals,
        num_components=cfg.dataset.num_components,
        output_folder=cfg.output_folder
    )

if __name__ == "__main__":
    main()