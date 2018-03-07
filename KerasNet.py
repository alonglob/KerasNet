from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Input, Flatten, Dropout
from keras.optimizers import SGD
from modules.read_tfrecord import DataSet

import numpy as np

# Create the Model
model = Sequential()

model.add(Conv2D(64, (8, 8), activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# importing the training data

dataset = iter(DataSet('/home/alon/Documents/tf_records/', 2, size=10))
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
