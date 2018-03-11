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
# model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(251,251,3))
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(64, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
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
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
#for layer in model.layers[:249]:
   #layer.trainable = False
#for layer in model.layers[249:]:
   #layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
print('Model loaded.')

x = model.output
x = Flatten(input_shape=(model.output_shape))(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation='softmax',name='predictions')(x)


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
# importing the training data

dataset = iter(DataSet('/home/alon/Documents/tf_records/', 2, batch_size=10))
try:
    for i in range(8000):
        batch = next(dataset)

        model.train_on_batch(batch[0], batch[1])
        if i % 100 == 0:
            classes = model.predict(batch[0])

            output_label = np.unravel_index(np.argmax(classes, axis=None), classes.shape)
            input_label = np.unravel_index(np.argmax(batch[1], axis=None), batch[1].shape)
            print('label: ' + np.str(input_label)+', output: ' + np.str(output_label))
            print(classes)

except StopIteration:
    print('iterator has stopped, probably finished training/eval')
except:
    print('this is not an iterator issue, issue is being raised:')
    raise
