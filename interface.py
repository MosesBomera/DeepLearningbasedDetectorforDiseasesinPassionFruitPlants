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
import pickle
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

# Kind of settings
storePath = r'/home/pi/katundaphotos'


# Main function
def main():
    # 1. Take picture
    camera = picamera.PiCamera()
    camera.start_preview()
    sleep(5)
    camera.capture('/home/pi/Desktop/image.jpg')
    camera.stop_preview()

    # 2. Display saved picture using pygame.
    pygame.init()
    fpsClock = pygame.time.Clock()
    surface = pygame.display.set_mode((320, 240))
    black = (0, 0, 0)
    image = pygame.image.load('/home/pi/Desktop/image.jpg')

    while True:
        surface.fill(black)
        surface.blit(image, (0, 0))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        fpsClock.tick(30)

class Utils:
    # Utility functions
    def load_labels(path):
      with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

    def take_picture():
        camera = picamera.PiCamera()
        camera.start_preview()
        sleep(5)
        camera.capture('/home/pi/Desktop/image.jpg')
        camera.stop_preview()

class Classifier:
    def __init__(self, model, labels):
        self.labels = labels
        self.model = Interpreter(model)

    def set_input_tensor(self, image):
      tensor_index = self.model.get_input_details()[0]['index']
      input_tensor = self.model.tensor(tensor_index)()[0]
      input_tensor[:, :] = image

    def classify_image(self, image, top_k=1):
      """Returns a sorted array of classification results."""
      set_input_tensor(self.model, image)
      self.model.invoke()
      output_details = self.model.get_output_details()[0]
      output = np.squeeze(self.model.get_tensor(output_details['index']))

      # If the model is quantized (uint8 data), then dequantize the results
      if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

      ordered = np.argpartition(-output, top_k)
      return [(i, output[i]) for i in ordered[:top_k]]

# class Button:

# Programme
if __name__ == '__main__':
    main()
