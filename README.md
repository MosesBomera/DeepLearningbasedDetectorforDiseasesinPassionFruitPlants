# A Deep Learning based Detector for Diseases in Passion Fruit Plants
Pests and diseases pose a key challenge to passion fruit farmers across Uganda and East Africa in general. They lead to loss of investment as yields reduce and losses increases. As the majority of the farmers, including passion fruit farmers, in the country are smallholder farmers from low-income households, they do not have the sufficient information and means to combat these challenges. While, passion fruits have the potential to improve the well-being of these farmers as they have a short maturity period and high market value , without the required knowledge about the health of their crops, farmers cannot intervene promptly to turn the situation around.

For this work, we have partnered with the Uganda National Crop Research Institute (NaCRRI) to develop a dataset of expertly labelled passion fruit plant leaves and fruits, both diseased and healthy. We have made use of their extension service to collect images from 5 districts in Uganda,
With the dataset in place, we are employing state-of-the-art techniques in machine learning, and specifically deep learning, techniques at scale for object detection and classification to correctly determine the health status of passion fruit plants and provide an accurate diagnosis for positive detections.This work focuses on two major diseases woodiness (viral) and brown spot (fungal) diseases.

## Folder Structure
### [pi_interface](./pi_interface)
Contains the implementation code for running inference using the trained models on a Raspberry Pi platform.

### [utils](./utils)
Contains utility files used to preprocess the image dataset including resizing, cropping, annotation json creation scripts etc.

### [object_detection_tutorial](./object_detection_tutorial)
Contains a Tensorflow object detection tutorial in three parts from data environment setup, data parsing and visualizing and finally end to end model training.

### [paper](./paper)
Results from this project on the brown spot disease detection were presented at [Data Science Africa](http://www.datascienceafrica.org/) 2020, Kampala. The slides can be found in the paper folder. The paper can be accessed on arXiv [here](https://arxiv.org/abs/2007.14103v2).
