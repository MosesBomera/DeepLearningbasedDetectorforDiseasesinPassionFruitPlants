# Python
import time
import logging
import argparse
import pygame
import os
import sys
import numpy as np
import subprocess
from capture import PiCameraStream
from PIL import Image
from tflite_runtime.interpreter import Interpreter

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

os.environ['SDL_FBDEV'] = "/dev/fb1"
os.environ['SDL_VIDEODRIVER'] = "fbcon"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

capture_manager = PiCameraStream(resolution=(max(512, screen.get_width()), max(512, screen.get_height())), rotation=180, preview=False)

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=5):
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

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()
    return args

last_seen = [None] * 10
last_spoken = None

def main(args):
    global last_spoken

    pygame.mouse.set_visible(False)
    screen.fill((0,0,0))
    # try:
    #     splash = pygame.image.load(os.path.dirname(sys.argv[0])+'/icons/passion-fruit.bmp')
    #     screen.blit(splash, ((screen.get_width() / 2) - (splash.get_width() / 2),
    #                 (screen.get_height() / 2) - (splash.get_height() / 2)))
    # except pygame.error:
    #     pass
    # pygame.display.update()

    # use the default font
    smallfont = pygame.font.Font(None, 24)
    medfont = pygame.font.Font(None, 36)
    bigfont = pygame.font.Font(None, 48)

    labels = load_labels(args.labels)
    logging.info(labels)

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

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

        timestamp = time.monotonic()
        prediction = classify_image(interpreter, frame)
        logging.info(f"Predictions: {prediction}")
        delta = time.monotonic() - timestamp
        logging.info("Inference took %d ms, %0.1f FPS" % (delta * 1000, 1 / delta))
        print(last_seen)

        # add FPS on top corner of image
        fpstext = "%0.1f FPS" % (1/delta,)
        fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        fpstext_position = (screen.get_width()-10, 10) # near the top right corner
        screen.blit(fpstext_surface, fpstext_surface.get_rect(topright=fpstext_position))

        for p in prediction:
            label_id, conf = p
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
                detecttext_color = (0, 0, 255) if persistant_obj else (255, 255, 255)
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
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
