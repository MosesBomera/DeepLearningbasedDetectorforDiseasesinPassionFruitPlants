# Script implements the katunda Graphical User Interface (GUI)
# Under the process explained below:
#
# 1.
# 2.
# 3.
# 4.
# 5.
# 6.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import cPickle as pickle
import errno
import fnmatch
import io
import os
import os.path
import picamera
import pygame
import stat
import threading
import time
# skimage ❭ color ❭ yuv2rgb
from pygame.locals import *
from subprocess import call
from time import sleep
from threading import Timer
import argparse
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Main function
def main():
    take_picture()

# Utility functions
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def take_picture():
    camera = picamera.PiCamera()
    camera.start_preview()
    sleep(5)
    camera.capture('/home/pi/Desktop/image.jpg')
    camera.stop_preview()

# Programme
if __name__ == '__main__':
    main()
