from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Input, Flatten, Dropout
from keras.optimizers import SGD
from keras import applications

from modules.read_tfrecord import DataSet

import numpy as np

# Create the Model, pretrained.
#model = ResNet50(include_top=False, weights='imagenet',input_shape=(251,251,3),classes=2, pooling='max')
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(251,251,3))
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
