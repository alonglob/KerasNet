# PreProcessing.py combines the magnifier.py and the image_to_tfrecord.py.
# this should run carefully

from PIL import Image, ImageOps
import image_slicer as slicer
import numpy as np
import tensorflow as tf
import os, sys
import threading

from modules.progress import progress


# check this for actual functionality! seems to work?
class FramerThread(threading.Thread):
    def __init__(self, path, name, label_num, training=True):
        threading.Thread.__init__(self)
        self.path = path
        self.name = name
        self.label_num = label_num
        self.training = training

    def run(self):
        print(" Starting " + self.name)
        framer(self.path, self.name, self.label_num, self.training)
        print(" Exiting " + self.name)


def mainFraming(main_path, num_classes):
    # create a training dataset
    thread1 = FramerThread(main_path, 'Blue_Dream', 0, training=True)
    thread2 = FramerThread(main_path, 'Lemon_Haze', 1, training=True)
    thread3 = FramerThread(main_path, 'Green_Crack', 2, training=True)

    # create a validation dataset
    thread4 = FramerThread(main_path, 'Blue_Dream', 0, training=False)
    thread5 = FramerThread(main_path, 'Lemon_Haze', 1, training=False)
    thread6 = FramerThread(main_path, 'Green_Crack', 2, training=False)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()

    print("finished main processes")


def framer(main_path, name, label, training=True):
    if training:
        directory = '/home/alon/Documents/train_directory/label_'
        print('Processing training images.')
    else:
        directory = '/home/alon/Documents/validation_directory/label_'
        print('Processing validation images.')

    i = 0
    x = 10  # i actually don't know why this is necessary
    cd = main_path + name
    for _ in os.listdir(cd):
        if _ != 'ready':
            try:
                img = Image.open(cd + '/' + _)

                old_size = img.size
                new_size = (512, 512 + x)
                img = img.crop((0, 0, old_size[0], old_size[1] - x))

                deltaw = int(new_size[0] - old_size[0])
                deltah = int(new_size[1] - old_size[1])
                ltrb_border = (int(deltaw / 2), int(deltah / 2), int(deltaw / 2), int(deltah / 2))
                img_with_border = ImageOps.expand(img, border=ltrb_border, fill='black')

                # img_with_border.save('Blueberry/ready/' + str(i) + '.png'
                img_with_border = img_with_border.resize(size=(512, 512))
                for theta in range(0, 360, 45):
                    img_with_border.rotate(theta).save(directory + str(label) + '/' + str(i) + '.png')
                    slicer.slice(directory + str(label) + '/' + str(i) + '.png', 4)
                    os.remove(directory + str(label) + '/' + str(i) + '.png')
                    i = i + 1
                if i < 100:
                    im = Image.open(directory + '0/0_01_01.png')
                    im_size = im.size

                sys.stdout.write("\r" + str(i) + ' ' + name + ' images processed.')
                sys.stdout.flush()

            except Exception:
                print('theres an issue')
                raise

    sys.stdout.write("\r" + 'A total of ' + str(i) + ' ' + name + ' images have been processed.')
    sys.stdout.write('image size: ' + np.str(im_size))
    sys.stdout.flush()


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

        progress(i, num_of_files, 'label_' + str(label_num))

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

        progress(i, num_of_data, 'upload status')

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

        progress(i, num_of_data, 'shuffle\saving status')

    writer.close()

    print(': successfully shuffled and saved data.')


class TfThread(threading.Thread):
    def __init__(self, name, dir_path, label_num, num_of_files, writer, training=True):
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


def mainrecording(tfrecords_path, num_classes):
    tfrecords_filenames = ['dataset_training.tfrecords', 'dataset_validation.tfrecords']
    path = tfrecords_path

    # Write a training tfrecord dataset
    writer1 = tf.python_io.TFRecordWriter(path + '/tf_records/' + tfrecords_filenames[0])

    thread1 = TfThread('thread 1', path, 0, 13000, writer1, training=True)
    thread2 = TfThread('thread 2', path, 1, 13000, writer1, training=True)
    thread3 = TfThread('thread 3', path, 2, 13000, writer1, training=True)

    # Write a validation tfrecord dataset
    writer2 = tf.python_io.TFRecordWriter(path + '/tf_records/' + tfrecords_filenames[1])
    thread4 = TfThread('thread 3', path, 0, 13000, writer2, training=False)
    thread5 = TfThread('thread 4', path, 1, 13000, writer2, training=False)
    thread6 = TfThread('thread 4', path, 2, 13000, writer2, training=False)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()

    writer1.close()
    writer2.close()

    # Shuffle the tfrecords for the final format
    shuffle_tfrecord(path, tfrecords_filenames[0], 33000, training=True)
    shuffle_tfrecord(path, tfrecords_filenames[1], 33000, training=False)
    print('writer successfully closed.')


if __name__ == '__main__':
    # set labels
    num_classes = 3

    # framer
    dataset_path = '/home/alon/Documents/DataSet/'
    mainFraming(dataset_path, num_classes)

    # tfrecords creation
    tfrecords_path = '/home/alon/Documents'
    mainrecording(tfrecords_path, num_classes)
