from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
from PIL import Image

# create a filter from image
filter_img = Image.open('filter.png')
filter_array = np.array(filter_img.getdata())

def PreProcessing_blacken(numpy_input):
    numpy_processed = np.array(numpy_input)
    print(np.shape(numpy_processed))
    numpy_output = np.dot(numpy_input,filter_array)

    return numpy_output




# Create a DataGenerator with Predefined Image Processing functions

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    samplewise_std_normalization=True,
    preprocessing_function=PreProcessing_blacken,
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
