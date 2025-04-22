import os
from generate_dataset import generate_and_store_dataset
from data_config import MIN_FREQUENCY, MAX_FREQUENCY
from composite_signal_generator import SyntheticSignalGenerator


"""
This script is used to repeatedly call the function responsible for generating datasets.

The purpose of this file is to automate the process of dataset generation by invoking
the dataset generation function multiple times, potentially with different parameters
or configurations. This can be useful for creating multiple datasets for testing,
training machine learning models, or other data processing tasks.

Make sure to configure the dataset generation function appropriately before running
this script to ensure the desired datasets are created.
"""

if __name__ == "__main__":
    # Create an instance of the SyntheticSignalGenerator
    generator = SyntheticSignalGenerator(
        fmin= MIN_FREQUENCY,
        fmax= MAX_FREQUENCY,
        duration=1.0,
        signal_types=['sine'],
        intermittence= 0.1,
        overlap_factor= 0.2,
        overlap_std= 0.1
    )

    # Generate and store the dataset
    generate_and_store_dataset(generator, num_signals=1000, num_components=3)