# Python
import time
import logging
import argparse
import pygame
import os
import sys
import numpy as np
import subprocess

from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
from threading import Thread

class PiCameraStream(object):
    """
      Continuously capture video frames, and optionally render with an overlay

      Arguments
      resolution - tuple (x, y) size
      framerate - int
      vflip - reflect capture on x-axis
      hflip - reflect capture on y-axis
    """

    def __init__(self, *, resolution=(320, 240), framerate=24, vflip=True, hflip=True, rotation=0, preview=True):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.camera.rotation = rotation
        self.data_container = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(
            self.data_container, format="bgr", use_video_port=True)
        self.frame = None
        self.stopped = False

        if preview:
            print('starting camera preview')
            self.camera.start_preview()

    def render_overlay(self):
        pass

    def start(self):
        """Begin handling frame stream in a separate thread"""
        Thread(target=self.flush, args=()).start()
        return self

    def flush(self):
        # looping until self.stopped flag is flipped
        # for now, grab the first frame in buffer, then empty buffer
        for f in self.stream:
            self.frame = f.array
            self.data_container.truncate(0)

            if self.stopped:
                self.stream.close()
                self.data_container.close()
                self.camera.close()
                return

    def read(self):
        return self.frame[0:224, 48:272, :]  # crop the frame

    def stop(self):
        self.stopped = True


class Utils:
    # Utility functions
    def load_labels(path):
      with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


class Classifier:
    def __init__(self, model, labels):
        self.labels = labels
        self.model = Interpreter(model)
        self.model.allocate_tensors()

    def get_wh(self):
        _, height, width, _ = self.model.get_input_details()[0]['shape']
        return (height, width)

    def set_input_tensor(self, image):
      tensor_index = self.model.get_input_details()[0]['index']
      input_tensor = self.model.tensor(tensor_index)()[0]
      input_tensor[:, :] = image

    def classify(self, image, top_k=1):
      """Returns a sorted array of classification results."""
      self.set_input_tensor(self.model, image)
      self.model.invoke()
      output_details = self.model.get_output_details()[0]
      output = np.squeeze(self.model.get_tensor(output_details['index']))

      # If the model is quantized (uint8 data), then dequantize the results
      if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

      ordered = np.argpartition(-output, top_k)
      return [(i, output[i]) for i in ordered[:top_k]]

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

os.environ['SDL_FBDEV'] = "/dev/fb1"
os.environ['SDL_VIDEODRIVER'] = "fbcon"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
capture_manager = PiCameraStream(resolution=(max(320, screen.get_width()), max(240, screen.get_height())), rotation=180, preview=False)

last_seen = [None] * 10
last_spoken = None

def main():
    global last_spoken

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()

    # Labels
    labels = Utils.load_labels(args.labels)

    # Initialize the model
    model = Classifier(args.model, labels)
    width, height = model.get_wh()

    pygame.mouse.set_visible(False)
    screen.fill((0,0,0))
    try:
        splash = pygame.image.load(os.path.dirname(sys.argv[0])+'/icons/passion-fruit.jpg')
        screen.blit(splash, ((screen.get_width() / 2) - (splash.get_width() / 2),
                    (screen.get_height() / 2) - (splash.get_height() / 2)))
    except pygame.error:
        pass
    pygame.display.update()

    # use the default font
    smallfont = pygame.font.Font(None, 24)
    medfont = pygame.font.Font(None, 36)
    bigfont = pygame.font.Font(None, 48)

    capture_manager.start()

    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue
        frame = capture_manager.read()
        # get the raw data frame & swap red & blue channels
        previewframe = np.ascontiguousarray(np.flip(np.array(capture_manager.frame), 2))
        # make it an image
        img = pygame.image.frombuffer(previewframe, capture_manager.camera.resolution, 'RGB')
        # draw it!
        screen.blit(img, (0, 0))
        image = frame.resize((width, height), Image.ANTIALIAS)

        timestamp = time.monotonic()
        # Prediction
        results = model.classify(image)
        logging.info(results)

        delta = time.monotonic() - timestamp
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if args.tflite else "TF", delta * 1000, 1 / delta))
        print(last_seen)

        # add FPS on top corner of image
        fpstext = "%0.1f FPS" % (1/delta,)
        fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        fpstext_position = (screen.get_width()-10, 10) # near the top right corner
        screen.blit(fpstext_surface, fpstext_surface.get_rect(topright=fpstext_position))

        label_id, conf = results[0]
        name = labels[label_id]
        if conf > CONFIDENCE_THRESHOLD:
            print("Detected", name)

            persistant_obj = False  # assume the object is not persistant
            last_seen.append(name)
            last_seen.pop(0)

            inferred_times = last_seen.count(name)
            if inferred_times / len(last_seen) > PERSISTANCE_THRESHOLD:  # over quarter time
                persistant_obj = True

            detecttext = name.replace("_", " ")
            detecttextfont = None
            for f in (bigfont, medfont, smallfont):
                detectsize = f.size(detecttext)
                if detectsize[0] < screen.get_width(): # it'll fit!
                    detecttextfont = f
                    break
            else:
                detecttextfont = smallfont # well, we'll do our best
            detecttext_color = (0, 255, 0) if persistant_obj else (255, 255, 255)
            detecttext_surface = detecttextfont.render(detecttext, True, detecttext_color)
            detecttext_position = (screen.get_width()//2,
                                   screen.get_height() - detecttextfont.size(detecttext)[1])
            screen.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))

            if persistant_obj and last_spoken != detecttext:
                os.system('echo %s | festival --tts & ' % detecttext)
                last_spoken = detecttext
            break
    else:
        last_seen.append(None)
        last_seen.pop(0)
        if last_seen.count(None) == len(last_seen):
            last_spoken = None


        pygame.display.update()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        capture_manager.stop()
