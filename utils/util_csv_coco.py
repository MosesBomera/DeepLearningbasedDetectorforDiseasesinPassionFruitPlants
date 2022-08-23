# util.py
"""

This script converts a labels.csv file to a COCO format JSON file.
The labels.csv file is expected to be in the following column format:

| 'file_name', 'xmin', 'ymin', 'width', 'height' |

The output file is named instances_(train|val).json depending on the flag
given.

The scripts expects the following file format:
project_name
    - train         (Contains the training images.)
    - val           (Contains the validation images )
    - annotations   (The folder in which the annotations file created is saved)

"""

import os
import json
import pandas as pd

from PIL import Image
from sys import exit, argv
from tqdm import tqdm

if len(argv) != 3:
    print("Usage: python util_csv_coco.py -train|test|val labels.csv")
    exit(1)

if argv[1] != '-train' and argv[1] != '-val' and argv[1] != '-test':
    print("Please use (-train) or (-val) or (-test) flag.")
    exit(1)

FLAG = str(argv[1][1:])

# DIRECTORIES
PROJECT_DIR = os.getcwd() #  project_name: brownspot
TRAIN_DIR = os.path.join(PROJECT_DIR, 'train')
# VAL_DIR = os.path.join(PROJECT_DIR, 'val')
# TEST_DIR =os.path.join(PROJECT_DIR, 'test')
ANNOTATIONS_DIR = os.path.join(PROJECT_DIR, 'annotations')

ANNOTATION_FILE = argv[2]
ANNOTATION_PATH = os.path.join(PROJECT_DIR, ANNOTATION_FILE)
CUR_DIR = os.path.join(PROJECT_DIR, FLAG)

def main():
    # Make directories and if necessary move the images in the respective folders.
    # dirs = [TRAIN_DIR, VAL_DIR, ANNOTATIONS_DIR]
    # make_dirs(dirs)

    filenames = os.listdir(TRAIN_DIR)

    # COCO format file
    output_annotation_file = os.path.join("instances_" + FLAG + ".json")
    labels = pd.read_csv(ANNOTATION_PATH)

    # Get the category list
    # categories = input("Enter categories separated by space: ")
    # categories = categories.rstrip('\n').lower().split(" ")

    categories = ['brownspot']
    # make and create coco file
    coco_data = make_coco_file(labels, categories, filenames)
    create_file(coco_data, output_annotation_file)

def make_dirs(dirs):
    """Creates directories that don't already exist."""
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)

def make_coco_file(labels, categories, filenames):
    """Creates a COCO format data structure."""

    classes = sorted(categories)
    category_list = []

    # COCO ANNOTATION FORMAT
    for i, category in enumerate(classes):
        foo = {}
        foo["supercategory"] = "None"
        foo["id"] = i + 1 # COCO is one-indexed
        foo["name"] = category
        category_list.append(foo)

    # COCO VARIABLE
    COCO_DATA = {}
    COCO_DATA["type"] = "instances"
    COCO_DATA["images"] = []
    COCO_DATA["annotations"] = []
    COCO_DATA["categories"] = category_list

    image_id = 0 # Image id
    annotation_id = 0 # Annotation id

    label = "brownspot" # Different implementation for more than one label

    # Iterate through the filenames
    for file_name in tqdm(filenames, desc="Creating COCO: "):

        # Add to COCO images
        temp = {}
        temp["file_name"] = file_name
        temp["height"] = 400 # Could be dynamic
        temp["width"]  = 400
        temp["id"]  = image_id
        COCO_DATA["images"].append(temp)

        # Bboxes
        image_bboxes = labels[labels.file_name == file_name]

        for _, row in image_bboxes.iterrows():
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            width = float(row["width"])
            height = float(row["height"])

            temp = {}
            temp["id"] = annotation_id
            annotation_id += 1
            temp["image_id"] = image_id
            temp["segmentation"] = []
            temp["ignore"] = 0
            temp["area"] = width * height
            temp["iscrowd"] = 0
            temp["bbox"] = [xmin, ymin, width, height]
            temp["category_id"] = classes.index(label) + 1 # There is only one class
            COCO_DATA["annotations"].append(temp)
        # Update image id
        image_id += 1

    # Return coco data format
    return COCO_DATA

def create_file(coco_data, output_file):
    """Create JSON file of the COCO_DATA."""
    f = open(output_file, 'w')
    json_str = json.dumps(coco_data, indent=4)
    f.write(json_str)
    f.close()

if __name__ == '__main__':
    main()
