# Inspired by Deep Learning for Computer Vision with Python [Rosebrock]
# Support classes for the demos

import tensorflow as tf
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_path_list, verbose=-1):
        data_list = []
        label_list = []

        # loop over the input images
        for (i, image_path) in enumerate(image_path_list):
            # load the image and extract the class label
            # assume that the path has the format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            # loop over the image preprocessors and apply each to the image
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat processed image as a feature vector
            # update the data list and the label list
            data_list.append(image)
            label_list.append(label)

            # occasionally show an update
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f'processed {i + 1}/{len(image_path_list)}')

        # return a tuple of the data and labels
        return (np.array(data_list), np.array(label_list))

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width   # target image width
        self.height = height # target image height
        self.inter = inter   # interpolation method to use when resizing

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        self.data_format = data_format # image data format

    def preprocess(self, image):
        # apply the utility function that correctly rearranges the dimensions of the image
        return tf.keras.preprocessing.image.img_to_array(image, data_format=self.data_format)
