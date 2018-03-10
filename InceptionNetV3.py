from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras import applications

from modules.read_tfrecord import DataSet

import numpy as np

# Create the Model, pretrained.
#model = ResNet50(include_top=False, weights='imagenet',input_shape=(251,251,3),classes=2, pooling='max')
#model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3), pooling='avg')
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(64, activation='relu')(x)
# and a logistic layer -- let's say we have 3 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
   #print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# importing the training data
dataset = iter(DataSet('/home/alon/Documents/tf_records/', 3, batch_size=2))
try:
    for i in range(20000):
        batch = next(dataset)

        model.train_on_batch(batch[0], batch[1])
        if i % 2000 == 0:
            classes = model.predict(batch[0])

            print(classes)
            print(batch[1])

        if i % 100 == 0:
            print('epoch: ' + str(i))

except StopIteration:
    print('iterator has stopped, probably finished training/eval')
except:
    print('this is not an iterator issue, issue is being raised:')
    raise