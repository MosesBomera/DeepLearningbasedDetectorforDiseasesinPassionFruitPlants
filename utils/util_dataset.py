# Util.py
"""
This script acts within the directory that it is in and
returns three COCO format JSON files and three folders
of test, train and val.
"""

import json
import os

import pandas as pd
import numpy as np

from shutil import copyfile
from tqdm import tqdm

# Existing directories
CWD = os.getcwd()
DATASET_DIR = os.path.join(CWD, 'dataset')

# Directories to make.
TRAIN_DIR = os.path.join(CWD, 'train')
TEST_DIR = os.path.join(CWD, 'test')
VAL_DIR = os.path.join(CWD, 'val')
ANNOTATIONS_DIR = os.path.join(CWD, 'annotations')
JSONS_DIR = os.path.join(CWD, 'labels')

def main():
    """Yes, let's do it."""

    # 1.0 Prepare folders.
    dirs =  [TRAIN_DIR, VAL_DIR, TEST_DIR, ANNOTATIONS_DIR]
    make_dirs(dirs)

    # 2.0 Get all the *.json
    jsons = get_json(JSONS_DIR)
    df = process_json(jsons)

    # 3.0 Split the images into train, test, val
    file_names = os.listdir(DATASET_DIR)
    train, test, val = split_files(file_names,train=0.85, test=0.05, validate=0.1)

    # 4.0 Create the COCO format *.json
    categories = ['brownspot']

    TRAIN_COCO = make_coco_file(df, categories, train)
    output_annotation_file = os.path.join(ANNOTATIONS_DIR, "instances_train.json")
    create_file(TRAIN_COCO, output_annotation_file)

    TEST_COCO = make_coco_file(df, categories, test)
    output_annotation_file = os.path.join(ANNOTATIONS_DIR, "instances_test.json")
    create_file(TEST_COCO, output_annotation_file)

    VAL_COCO = make_coco_file(df, categories, val)
    output_annotation_file = os.path.join(ANNOTATIONS_DIR, "instances_val.json")
    create_file(VAL_COCO, output_annotation_file)

    # 5.0 Move files into the respective folders
    move_files(train, DATASET_DIR, TRAIN_DIR)
    move_files(test, DATASET_DIR, TEST_DIR)
    move_files(val, DATASET_DIR, VAL_DIR)

    # Should run like magic.

def make_dirs(dirs):
    """Make directories that don't exist."""
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)

def get_json(dir):
    """Get *.json files in dir."""
    jsons = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            jsons.append(os.path.join(dir, file))
    return jsons

def process_json(jsons):
    """EXtracts the image file_name and it's bounding box to a dataframe."""

    df = pd.DataFrame(columns=['file_name',
                                'xmin',
                                'ymin',
                                'width',
                                'height'])

    # Extract
    for file in tqdm(jsons, desc="Processing *.json: "):
        f = open(file, 'r')
        data = json.load(f)
        f.close()

        # Files
        files = data["images"]
        annotations = data["annotations"]

        for file in files:
            # Get file_name
            image_id = file["id"]

            for annotation in annotations:
                if image_id == annotation["image_id"]:
                    bbox = annotation["bbox"]
                    df = df.append({"file_name": file["file_name"],
                                    "xmin": bbox[0],
                                    "ymin": bbox[1],
                                    "width": bbox[2],
                                    "height": bbox[3]},
                                    ignore_index=True)
    return df

def split_indices(x, train=0.8, test=0.0, validate=0.2, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

def split_files(file_names,train=0.8, test=0.2, validate=0.0):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_names))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_names, train=train, test=test, validate=validate)
    train = []
    test = []
    val = []
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            for ix in item:
                if key == 'train':
                    train.append(file_names[ix])
                if key == 'test':
                    test.append(file_names[ix])
                if key == 'val':
                    val.append(file_names[ix])

    return train, test, val

# create coco file
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


def move_files(files, source, dest):
  """Move files from the source directory to the destination directory."""
  for filename in files:
      copyfile(os.path.join(source, filename),
                 os.path.join(dest, filename))

# Poof
if __name__ == "__main__":
    main()
