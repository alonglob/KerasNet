# personal tfrecords reader.
# parameters: height, width, raw pixel matrix, label

import tensorflow as tf
import skimage.io as io
from collections import Iterator, Generator
import numpy as np


class DataSet:
    def __init__(self, tfrecords_path, num_classes, training=True, size=1):

        if training:
            datasetName = 'dataset_training_shuffled.tfrecords'
        else:
            datasetName = 'dataset_validation_shuffled.tfrecords'

        self.record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_path+datasetName)
        self.size = size
        self.num_classes = num_classes
        self.training = training

    def __next__(self):

        labels = np.ndarray([self.size, self.num_classes])

        for i in range(self.size):
            string_record = next(self.record_iterator)

            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])

            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            img_string = (example.features.feature['image_raw']
                          .bytes_list
                          .value[0])

            label = (example.features.feature['label_raw']
                     .int64_list
                     .value[0])

            # after the first run, the size can be determined:
            if i == 0:
                images = np.ndarray([self.size, height, width, 3])

            img_1d = np.fromstring(img_string, dtype=np.uint8)
            img_1d = np.divide(img_1d.astype(dtype=np.float32), 255)
            reconstructed_img = img_1d.reshape((height, width, -1))

            targets = np.array([label]).reshape(-1)
            one_hot_labels = np.eye(self.num_classes)[targets]

            images[i] = reconstructed_img
            labels[i] = one_hot_labels

        return images,  labels

    def __iter__(self):
        return self
