import os
import json
import numpy as np
from datetime import datetime

"""
This module provides functionality to generate and store datasets of composite signals 
using a synthetic signal generator. The generated dataset is saved as a compressed NPZ 
file, and the parameters used for generation are stored in a JSON file.

Classes:
    SyntheticSignalGenerator: A class responsible for generating synthetic signals 
    with specified characteristics.

Functions:
    generate_and_store_dataset(generator, num_signals, k):
        Generate a dataset of composite signals and store it as an NPZ file along 
        with the parameters used.

Usage:
    To use this module, create an instance of the `SyntheticSignalGenerator` class 
    with the desired parameters, and call the `generate_and_store_dataset` function 
    with the generator instance, the number of signals to generate, and the number 
    of frequency segments per signal.
"""

def generate_and_store_dataset(generator, num_signals, num_components):
    """
    Generate a dataset of composite signals and store it as an NPZ file along with the parameters used.

    Args:
        generator (SyntheticSignalGenerator): An instance of the SyntheticSignalGenerator class.
        num_signals (int): Number of composite signals to generate.
        k (int): Number of frequency segments for each signal.

    Returns:
        None
    """
    
    # Ensure the output folder exists
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique identifier (timestamp)
    unique_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Initialize lists to store the dataset
    composite_signals = []
    all_components = []

    for _ in range(num_signals):
        # Generate a composite signal and its components
        composite_signal, components = generator.generate_signal(f0=generator.fmin, bandwidth= generator.fmax - generator.fmin , k=num_components)
        composite_signals.append(composite_signal)
        all_components.append(components)

    # Convert the dataset to NumPy arrays
    composite_signals = np.array(composite_signals)
    all_components = np.array(all_components)

    # Save the dataset as an NPZ file with the unique identifier
    dataset_filename = f"composite_signals_{unique_id}.npz"
    dataset_path = os.path.join(output_folder, "data_storage", dataset_filename)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    np.savez_compressed(dataset_path, composite_signals=composite_signals, components=all_components)

    # Save the parameters used for the entire dataset as a JSON file with the same unique identifier
    params = {
        "num_signals": num_signals,
        "fmin": generator.fmin,
        "fmax": generator.fmax,
        "duration": generator.duration,
        "signal_types": generator.signal_types,
        "intermittence": generator.intermittence,
        "overlap_factor": generator.overlap_factor,
        "overlap_std": generator.overlap_std,
        "num_segments": num_components,
        "dataset_file": dataset_filename  # Link the dataset file in the JSON
    }
    params_filename = f"parameters_{unique_id}.json"
    params_path = os.path.join(output_folder, params_filename)
    with open(params_path, "w") as params_file:
        json.dump(params, params_file, indent=4)

    print(f"Dataset saved to {dataset_path}")
    print(f"Parameters saved to {params_path}")