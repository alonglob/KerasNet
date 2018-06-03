from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image, ImageOps
import numpy as np
from PIL import Image


# create a filter from image

# Create a DataGenerator with Predefined Image Processing functions

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    samplewise_std_normalization=True,
    data_format="channels_last")

# From the train dataGenerator create an identical test dataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)


def initiate_generators(batchSize=1, targetSize=(256, 256), dataset_path='/home/alon/Documents/DataSet/'):

    # Training_Generator of DataGenerator Type
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=targetSize,
        batch_size=batchSize,
        class_mode='categorical')

    # Validation_Generator of DataGenerator Type
    validation_generator = test_datagen.flow_from_directory(
        dataset_path,
        target_size=targetSize,
        batch_size=batchSize,
        class_mode='categorical')

    return train_generator, validation_generator
