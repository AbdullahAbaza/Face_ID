# Import necessary libraries
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sqlite3

from deepface import DeepFace


# Define a function to find faces in an input image and return the identity of the closest match
def find_faces(input_img: Path, dataset_dir: Path) -> str:
    try:
        # Use DeepFace library to find the closest match in the dataset directory
        status = DeepFace.find(input_img.as_posix(),
                               dataset_dir.as_posix(),
                               model_name="Facenet512",
                               distance_metric="euclidean_l2",
                               enforce_detection=True,
                               detector_backend="ssd",
                               silent=False)
        # Return the identity of the closest match
        return status[0].iloc[0]['identity']

    except ValueError as e:
        logging.exception(e)
        # If no match is found, return an empty string
        return ""


# Define the main function
def main():
    # Configure logging to output only error messages
    logging.basicConfig(level=logging.ERROR)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('input_img', type=Path, help='path to the image')
    parser.add_argument('dataset_dir',
                        type=Path,
                        help='path to the dataset directory')
    args = parser.parse_args()
    input_img = args.input_img
    dataset_dir = args.dataset_dir

    # Check if the dataset directory exists
    if not dataset_dir.exists():
        logging.error(f"Dataset directory {dataset_dir} does not exist")
        return

    # Check if the input image file exists
    if not input_img.is_file():
        logging.error(f"Image path {input_img} is not a file")
        return

    # Call the find_faces function to get the identity of the closest match
    dataset_img = find_faces(input_img, dataset_dir)

    # Set the verified flag to indicate whether the input image was verified in the dataset
    if len(dataset_img) == 0:
        verified = 0  # False or Not Verified
    else:
        verified = 1  # True Verified in the dataset and saved the path in dataset_img

    # Create or connect to the database file
    db_filename = 'face_recognition_results.db'
    conn = sqlite3.connect(db_filename)

    # Create a table to store the results if it doesn't exist
    conn.execute('''CREATE TABLE IF NOT EXISTS recognition_results (
                    datetime text,
                    input_img text,
                    dataset_img text,
                    verified int
                    )''')

    # Insert the result into the table
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(f"INSERT INTO recognition_results VALUES (?, ?, ?, ?)",
                 (now, input_img.as_posix(), dataset_img, verified))
    conn.commit()

    # Close the database connection
    conn.close()

    # Print the verified flag
    print(verified)


# Call the main function if this script is being run directly
if __name__ == "__main__":
    main()
