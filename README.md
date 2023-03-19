# Face_ID

This project is a command-line tool for face recognition using the `DeepFace` library. https://github.com/serengil/deepface 
The tool takes an input image and a directory containing a dataset of reference images, and returns whether the input image is a verified match in the dataset.

# Installation
To use this tool, you need to install the following dependencies:

DeepFace
argparse
logging
pathlib
datetime
sqlite3

You can install these dependencies using pip:
```shell
$ pip install deepface argparse logging pathlib datetime sqlite3
```

# Usage
To use the tool, run the face_recognition.py script with the following command-line arguments:

input_img: the path to the input image
dataset_dir: the path to the directory containing the dataset of reference images

```shell
$ python Face_ID.py input_img dataset_dir
```

The tool will output a verified flag indicating whether the input image is a verified match in the dataset:

	- 1 indicates a verified match
	- 0 indicates no verified match

The tool also saves the result to a SQLite database file `face_recognition_results.db`. The database file contains a single table recognition_results with the following columns:

	- datetime: the date and time of the recognition result
	- input_img: the path to the input image
	- dataset_img: the path to the closest match in the dataset (if any)
	- verified: the verified flag (0 or 1)

You can use the SQLite command-line tool or a graphical tool such as `DB Browser for SQLite` to view the database file. https://sqlitebrowser.org/

Even though face recognition is based on one-shot learning, you can use multiple face pictures of a person as well. You should rearrange your directory structure as illustrated below.
```
user
├── database
│   ├── Alice
│   │   ├── Alice1.jpg
│   │   ├── Alice2.jpg
│   ├── Bob
│   │   ├── Bob.jpg
```


WARNING:
Representations for images are saved in the dataset folder after training under the name representations_{model_name}.pkl. If you added new instances after the creation, then please delete this file and run the script again. It will create it again.
