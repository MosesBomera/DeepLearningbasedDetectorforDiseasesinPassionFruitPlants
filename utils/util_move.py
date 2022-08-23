# This script moves files from a specified directory to another directory.
import os
from sys import argv, exit

# Get current directory
directory = os.getcwd()
# print(os.listdir('.'))

if len(argv) != 3:
    print("Usage: python util.py file.txt IN_DIR OUT_DIR \
          \nIN_DIR     the directory with the files \
          \nOUT_DIR    the directory to which the files are going.")
    exit(1)

IN_DIR = argv[2]            #
OUT_DIR = argv[3]

# os.rename(os.path.join(directory, 'dataset', 'train.txt'), os.path.join(directory, 'train.txt'))
file = os.path.join(directory, argv[1])

with open(file, 'r')  as train:
    lines = train.readlines()

    for line in lines:
        # Get details
        contents = line.split('/')

        # the exact name of the file
        filename = contents[-1].strip('\n')

        type = argv[1].split('.')[0]
        # new
        new_dir = os.path.join(directory, 'dataset', type, filename)

        # print(type, new_dir)
        os.rename(line.rstrip('\n'), new_dir)
