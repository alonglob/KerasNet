# 2nd run,
# after creating train/valid directories with numbered labels,
# these files will be converted to a tensorflow records file.


from PIL import Image
import numpy as np
import tensorflow as tf
import os, sys
import threading


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_writer(image_path, label, writer, channels=3):
    # Saving the image as a Serialized String with the writer.
    img = np.array(Image.open(image_path))
    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]

    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'label_raw': _int64_feature(label),
        'channels': _int64_feature(channels)}))

    writer.write(example.SerializeToString())


def create_tfrecord(dir_path, label_num, num_of_files, writer, training=True):
    # create a tfrecord file from the images in dir_path
    directories = ['/train_directory/label_', '/validation_directory/label_']
    if training:
        directory = directories[0]
        print('creating a training tfrecords file for label ' + str(label_num))
    else:
        directory = directories[1]
        print('creating a validation tfrecords file for label ' + str(label_num))

    i = 0
    for _ in os.listdir(dir_path + directory + str(label_num) + '/'):

        if i > num_of_files:
            break

        try:
            tf_writer(dir_path + directory + str(label_num) + '/' + _, label_num, writer)
        except Exception:
            print(dir_path + directory + str(label_num) + '/' + _ + ' failed')
            pass

        if i % 1000 == 0:
            sys.stdout.write("\r" + str(i) + '/' + str(num_of_files) + ' for label_' + str(label_num))
            sys.stdout.flush()

        i = i + 1
    print('')
    print('writing for label_' + str(label_num) + ' complete.')


def shuffle_tfrecord(dir_path, filename, num_of_data, training=True):
    datasetNames = ['dataset_training_shuffled.tfrecords', 'dataset_validation_shuffled.tfrecords']
    if training:
        dataName = datasetNames[0]
    else:
        dataName = datasetNames[1]

    print(dir_path + '/tf_records/' + filename)
    record_iterator = tf.python_io.tf_record_iterator(path=(dir_path + '/tf_records/' + filename))
    connected = []
    height = 0
    width = 0

    print('uploading tfrecords to memory....')

    for i in range(num_of_data):
        string_record = next(record_iterator)

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

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        connected.append([reconstructed_img, label])

        if i % 100 == 0:
            sys.stdout.write("\r" + 'uploading status: ' + str(i) + '/' + str(num_of_data))
            sys.stdout.flush()

    np.random.shuffle(connected)
    writer = tf.python_io.TFRecordWriter(dir_path + '/tf_records/' + dataName)
    print(': successfully loaded and shuffled data, saving process is beginning: ')

    for i in range(num_of_data):

        img_raw = connected[i][0].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'label_raw': _int64_feature(connected[i][1]),
            'channels': _int64_feature(3)}))

        writer.write(example.SerializeToString())

        if i % 100 == 0:
            sys.stdout.write("\r" + 'saving status: ' + str(i) + '/' + str(num_of_data))
            sys.stdout.flush()

    writer.close()

    print(': successfully shuffled and saved data.')


class TfThread(threading.Thread):
    def __init__(self,name, dir_path, label_num, num_of_files, writer, training=True):
        threading.Thread.__init__(self)
        self.dir_path = dir_path
        self.label_num = label_num
        self.num_of_files = num_of_files
        self.writer = writer
        self.training = training
        self.name = name


    def run(self):
        print("Starting " + self.name)
        create_tfrecord(self.dir_path, self.label_num, self.num_of_files, self.writer, self.training)
        print("Exiting " + self.name)


tfrecords_filenames = ['dataset_training.tfrecords', 'dataset_validation.tfrecords']
doc_path = '/home/alon/Documents'
# Write a training tfrecord dataset
writer1 = tf.python_io.TFRecordWriter(doc_path + '/tf_records/' + tfrecords_filenames[0])
thread1 = TfThread('thread 1', doc_path, 0, 20000, writer1, training=True)
thread2 = TfThread('thread 2', doc_path, 1, 13000, writer1, training=True)


# Write a validation tfrecord dataset
writer2 = tf.python_io.TFRecordWriter(doc_path + '/tf_records/' + tfrecords_filenames[1])
thread3 = TfThread('thread 3', doc_path, 0, 20000, writer2, training=False)
thread4 = TfThread('thread 4', doc_path, 1, 13000, writer2, training=False)


thread1.start()
thread2.start()
thread3.start()
thread4.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()

writer1.close()
writer2.close()

# Shuffle the tfrecords for the final format
shuffle_tfrecord(doc_path, tfrecords_filenames[0], 33000, training=True)
shuffle_tfrecord(doc_path, tfrecords_filenames[1], 33000, training=False)
print('writer successfully closed.')
