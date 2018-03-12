from keras.preprocessing.image import ImageDataGenerator

# Create a DataGenerator with Predefined Image Processing functions
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last")

# From the train dataGenerator create an identical test dataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Training_Generator of DataGenerator Type
train_generator = train_datagen.flow_from_directory(
        '/home/alon/Documents/DataSet/',
        target_size=(256, 256),
        batch_size=8,
        class_mode='categorical')

# Validation_Generator of DataGenerator Type
validation_generator = test_datagen.flow_from_directory(
        '/home/alon/Documents/DataSet/',
        target_size=(256, 256),
        batch_size=8,
        class_mode='categorical')

