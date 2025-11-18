"""Download the ShipsNet dataset and save it to the dataset folder."""

import os
import shutil

import kagglehub


def download_shipsnet_dataset():
    """Downloads the ShipsNet dataset and saves shipsnet.json to the current folder.
    Returns the path to the saved file, or None if not found.
    """
    # Download the latest version of the dataset
    path = kagglehub.dataset_download("rhammell/ships-in-satellite-imagery")
    print("Path to dataset files:", path)

    # Define the path to the dataset file
    shipsnet_path = os.path.join(path, "shipsnet.json")

    # Define the destination folder and file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    destination_folder = os.path.join(current_dir)
    destination_file = os.path.join(destination_folder, "shipsnet.json")

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Copy the dataset file to the destination folder
    if os.path.exists(shipsnet_path):
        shutil.copy(shipsnet_path, destination_file)
        print(f"Dataset saved to: {destination_file}")
        return destination_file
    else:
        print("Error: shipsnet.json not found in the downloaded dataset.")
        return None


if __name__ == "__main__":
    download_shipsnet_dataset()
