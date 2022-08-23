'''
Utility to convert from Labelbox JSON to YOLO format. Also handles test, train split of the images

- change: OUTPUT_DIR, IMAGE_ABS_DIR and LABELBOX_ABS_DIR
- The script is desigened to be run within the convert folder.
- NOTE: The labels are saved separately so depending on your YOLO setup you may have to combine them with the images
'''
from sys import argv, exit

import json
import os
from pathlib import Path
import numpy as np
import shutil
import csv

#Extra packages
#pip/conda install the following (numpy, pandas, pillow, tqdm)
import pandas as pd
from PIL import Image
from tqdm import tqdm

#GLOBAL DIRECTORY DEFINITIONS (change as you see fit)
OUTPUT_DIR = "../yolo/out/" # Folder into with the labels will be pt
IMAGE_ABS_DIR = r"/home/bomera/Shared/brownspot/brownspot"
#Folder in which the labelbox json files are placed (names are not important)
LABELBOX_ABS_DIR = r"labels"

CLASS_ID = 1

def get_json_files_in_dir(json_dir):

  label_files = []
  for label_file in os.listdir(json_dir):
      if label_file.endswith(".json"):
          label_files.append(os.path.join(json_dir, label_file))
  return label_files


def make_folders(path='../yolo/out'):
    # Create folders

    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    # os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path

def split_indices(x, train=0.8, test=0.2, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

def split_files(out_path, file_names,train=0.8, test=0.2, validate=0.0):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_names))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_names, train=train, test=test, validate=validate)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(os.path.join(out_path,  key + '.txt'), 'w') as file:
                for ix in item:
                    file.write('%s\n' % (file_names[ix]))

    print("Split: train: {} , test: {}, validation : {} (images)".format(len(i), len(j), len(k)))

# Convert Labelbox JSON file into YOLO-format labels ---------------------------
def convert_labelbox_json(name, file):
    # Create folders
    path = make_folders(OUTPUT_DIR)

    image_labels_json_file_names = get_json_files_in_dir(LABELBOX_ABS_DIR)

    data_list = []

    for image_labels_json_file_name in image_labels_json_file_names:
        with open(image_labels_json_file_name) as x:
            data_list.extend(json.load(x))

   #filenames
    file_names =  []

    for item in tqdm(data_list, desc='Images'):
        # print ('Working')
        item_label = item['Label']

        # check if the item['Label'] is empty. If empty, continue to the next item.
        if bool(item_label)==False:
            continue
        # check if the item['Label']['objects'] is empty, if so, proceed to the next item.
        elif bool(item_label['objects']==False):
            continue

        # All  is well, execute the loop below for all the bounding boxes.
        #Image file_name
        file_name=item['External ID']

        file_names.append(os.path.join(IMAGE_ABS_DIR, file_name))
        #name of file to save the bounding box items
        label_name = str(Path(file_name).stem + '.txt')

        #Get Image width and height
        current_img = Image.open(os.path.join(IMAGE_ABS_DIR, file_name))
        image_width, image_height = current_img.size

        #create label file for current image
        with open(os.path.join(OUTPUT_DIR, 'labels',  label_name), 'w') as file:

            #Extract the bounding boxes
            for bounding_box in item_label['objects']:
                # print(dictionary['bbox'])

                # class_name = bounding_box['title'].replace(" ", "")

                top = bounding_box['bbox']['top']
                left = bounding_box['bbox']['left']

                height = bounding_box['bbox']['height']
                width = bounding_box['bbox']['width']

                #note that the axis refers to the top left corner of the image
                xmin = left
                xmax = left + width
                ymin = top # actually
                ymax = top + height

                x_center = 0.5*(xmin + xmax)
                y_center = 0.5*(ymin + ymax)

                #Normalize with image dimensions
                x_center_norm =  x_center/image_width
                width_norm = width/image_width

                y_center_norm = y_center/image_height
                height_norm = height/image_height
                # <object-class> <x> <y> <width> <height> Note <> - normalized
                file.write('%g %.6f %.6f %.6f %.6f\n' % (CLASS_ID - 1, x_center_norm, y_center_norm,
                 width_norm, height_norm))



    # # Split data into train, test, and validate files
    split_files(OUTPUT_DIR, file_names, train=0.7, test=0.2, validate=0.1)
    print('Done. Output saved to %s' % (OUTPUT_DIR,))

def xml_to_csv_and_pbtxt(folder_path, filenames):
  #parsing a large json fail to python
  image_labels_json_file_names = get_json_files_in_dir(LABELBOX_ABS_DIR)

  data_list = []
  class_name = None

  for image_labels_json_file_name in image_labels_json_file_names:
    with open(image_labels_json_file_name) as x:
        data_list.extend(json.load(x))

      # Generating the csv in the root of the dataset
  parent_dir = Path(folder_path).parent
  csv_name = 'labels.csv'

  prefix = Path(folder_path).name.lower()
  csv_file_name = os.path.join(str(parent_dir), csv_name)

  # classes = []
  with open(csv_file_name, 'w') as csv_label_file:
    print('Working-csv')
    f = csv.writer(csv_label_file)
    # f.writerow(["Class", "fileName", "top","left","height","width"])
    f.writerow(['file_name', 'xmin', 'ymin', 'width', 'height'])

    if data_list:
        for item in tqdm(data_list, desc="Images"):
            # print ('Working')
            item_label = item['Label']
            # print(item_label)
            #image file_name
            file_name=item['External ID']

            # check if the item['Label'] is empty. If empty, continue to the next item.
            if bool(item_label)==False:
                continue
            # check if the item['Label']['objects'] is empty, if so, proceed to the next item.
            elif bool(item_label['objects']==False):
                continue
            else:
                pass

            if file_name in filenames:
                for bounding_box in item_label['objects']:
                    # print(dictionary['bbox'])

                    class_name = bounding_box['title'].replace(" ", "")

                    top = bounding_box['bbox']['top']
                    left = bounding_box['bbox']['left']
                    height = bounding_box['bbox']['height']
                    width = bounding_box['bbox']['width']

                    #note that the axis refers to the top left corner of the image
                    # xmin = left
                    # xmax = left + width
                    # ymin = top # actually
                    # ymax = top + height

                    f.writerow([file_name,
                                left,
                                top,
                                width,
                                height
                                ])

    else:
        print("Data List is empty.")


if __name__ == '__main__':
    # source = 'labelbox'
    #
    # if source is 'labelbox':  # Labelbox https://labelbox.com/
    #  convert_labelbox_json(name='brownspot_phase_I',
    #                         file='../labelbox/labels_cosmas.json')

    #
    directory = os.path.join(os.getcwd(), 'train')
    filenames = os.listdir(directory)

    # with open('/home/bomera/Shared/brownspot/train.txt', 'r') as train:
    #     lines = train.readlines()
    #     for line in lines:
    #         line = line.rstrip('\n')
    #         contents = line.split('/')
    #         filenames.append(contents[-1])

    xml_to_csv_and_pbtxt(directory, filenames)



    # zip results
    # os.system('zip -r ../coco.zip ../coco')
