from __future__ import print_function

import os, sys, random
from typing import List

from sklearn.metrics import classification_report
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist
from keras.datasets import cifar100
from random import sample, randint
from keras.layers import Input, Cropping2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from pathlib import *
from src.ml.utils import get_all_files, extract_tar, check_md5, download_file
import cv2

# Set some initial parameters.


batch_size = 100

img_sizeX = 500
img_sizeY = 500
EXTRACT = False

input_shape = (img_sizeY, img_sizeX, 3)
input_img = Input(shape=input_shape)

# Import the keras backend for loss functions.
import keras.backend as K


# The mean squared error. maybe remove.
def mean_squared_error(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


if download_file("https://dataset.erwankessler.com/faces.tar"):
    print("File downloaded")
else:
    print("File already exists")

if check_md5("ca9606f7193adfaab2da0c7926c35764"):
    print("MD5 checked")
else:
    print("MD5 error")
if EXTRACT:
    main_path = extract_tar()
else:
    main_path = "../../data/dataset_train"
print("Archive extracted")

laplacian_one_path = "../../data/dataset_images/LaplacianImages/one"
laplacian_two_path = "../../data/dataset_images/LaplacianImages/two"
laplacian_three_path = "../../data/dataset_images/LaplacianImages/three"

all_images_path = get_all_files(main_path)
random.shuffle(all_images_path)

number_of_images = len(all_images_path)
