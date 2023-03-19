import os
import hashlib
import pickle
from pathlib import Path


def hash_dataset(dataset_path: Path) -> str:
    """
    Calculates the hash of the current state of the dataset by walking through the directory,
    reading each file in chunks, and updating an SHA-256 hash object. The resulting hash value
    is returned as a hexadecimal string.
    """
    # Initialize an SHA-256 hash object
    sha256 = hashlib.sha256()
    # Recursively traverse the dataset directory
    for path in dataset_path.glob('**/*'):
        # Check if the path represents a file
        if path.is_file():
            # Read file in chunks and update the hash object
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
    # Return the hash value as a hexadecimal string
    return sha256.hexdigest()


def dataset_has_changed(dataset_pkl: str, dataset_path: Path) -> bool:
    """
    Checks if the dataset has changed since the last training by comparing the hash of the
    current dataset with the hash saved in a pickle file. If the hash values differ, the
    function returns True, indicating that the dataset has changed.
    """
    # Check if the pickle file exists
    if not os.path.exists(dataset_pkl):
        return True
    # Load the previous hash value from the pickle file
    with open(dataset_pkl, "rb") as f:
        prev_hash = pickle.load(f)
    # Check if the hash of the current dataset is different from the previous hash
    return prev_hash != hash_dataset(dataset_path)


def remove_pkl_files(*file_paths: str) -> None:
    """
    Removes the pickle files specified by file paths to trigger re-training of the model.
    """
    # Loop over each file path argument
    for file_path in file_paths:
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            # Print a message to indicate that the file was deleted
            print(f"Removed {file_path} to trigger re-training.")


def dataset_check(dataset_path: Path) -> None:
    """
    Checks for changes in the dataset directory. If the dataset has changed, removes the pickle
    files to trigger re-training of the VGG-Face model.
    """
    # Define the paths to the pickle files
    dataset_pkl = dataset_path / "dataset-hashes.pkl"
    representations_pkl = dataset_path / "representations_vgg_face.pkl"
    # Check if the pickle file exists and if the dataset has changed
    if dataset_pkl.is_file() and dataset_has_changed(dataset_pkl,
                                                     dataset_path):
        # If the dataset has changed, remove the pickle files
        remove_pkl_files(dataset_pkl, representations_pkl)
        # Print a message to indicate that the pickle files were removed
        print("Removed pickle files to trigger re-training.")
    # Update the hash pickle file with the current dataset hash
    with dataset_pkl.open("wb") as f:
        pickle.dump(hash_dataset(dataset_path), f)
        # Print a message to indicate that the pickle file was updated
        print("Updated dataset hashes pickle file.")
