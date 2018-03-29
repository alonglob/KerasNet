from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD

from old_files.read_tfrecord import DataSet

# Create the Model
model = Sequential()

model.add(Conv2D(128, (128, 128), activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(128, (64, 64), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# importing the training data

dataset = iter(DataSet('/home/alon/Documents/tf_records/', 3, batch_size=10))
try:
    for i in range(20000):
        batch = next(dataset)

        model.train_on_batch(batch[0], batch[1])
        if i % 2000 == 0:
            classes = model.predict(batch[0])

            print(classes)
            print(batch[1])

        if i % 500 == 0:
            print('epoch: ' + str(i))

except StopIteration:
    print('iterator has stopped, probably finished training/eval')
except:
    print('this is not an iterator issue, issue is being raised:')
    raise
