import os
from generate_dataset import generate_and_store_dataset
from generate_dataset import SyntheticSignalGenerator


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
        fmin=100,
        fmax=500,
        duration=1.0,
        signal_types=['sine'],
        intermittence= 0.1
    )

    # Generate and store the dataset
    generate_and_store_dataset(generator, num_signals=100, k=3)