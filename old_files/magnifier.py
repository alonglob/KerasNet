# 1st run,
# magnifier.py

from PIL import Image, ImageOps
import os,sys
import image_slicer as slicer
import threading

# check this for actual functionality!
class FramerThread(threading.Thread):
    def __init__(self,path, name, label_num, training=True):
        threading.Thread.__init__(self)
        self.path = path
        self.name = name
        self.label_num = label_num
        self.training = training

    def run(self):
        print("Starting " + self.name)
        framer(self.path, self.name, self.label_num, self.training)
        print("Exiting " + self.name)

def main(main_path):

    # create a training dataset
    FramerThread(main_path, 'Blue_Dream', 0, training=True)
    FramerThread(main_path, 'Lemon_Haze', 1, training=True)

    # create a validation dataset
    FramerThread(main_path, 'Blue_Dream', 0, training=False)
    FramerThread(main_path, 'Lemon_Haze', 1, training=False)

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
                new_size = (128, 128 + x)
                img = img.crop((0, 0, old_size[0], old_size[1] - x))

                deltaw = int(new_size[0] - old_size[0])
                deltah = int(new_size[1] - old_size[1])
                ltrb_border = (int(deltaw / 2), int(deltah / 2), int(deltaw / 2), int(deltah / 2))
                img_with_border = ImageOps.expand(img, border=ltrb_border, fill='black')

                # img_with_border.save('Blueberry/ready/' + str(i) + '.png'
                img_with_border = img_with_border.resize(size=(128, 128))
                for theta in range(0, 360, 45):
                    img_with_border.rotate(theta).save(directory + str(label) + '/' + str(i) + '.png')
                    slicer.slice(directory + str(label) + '/' + str(i) + '.png', 4)
                    os.remove(directory + str(label) + '/' + str(i) + '.png')
                    i = i + 1
                if i > 3:
                    im = Image.open(directory + '0/0_01_01.png')
                    sys.stdout.write("\r" + str(i) + ' ' + name + ' images processed.' + str(im.size))

                sys.stdout.write("\r" + str(i) + ' ' + name + ' images processed.')
                sys.stdout.flush()

            except Exception:
                print('theres an issue')
                raise

    sys.stdout.write("\r" + 'A total of ' + str(i) + ' ' + name + ' images have been processed.')
    sys.stdout.flush()

if __name__ == '__main__':
    dataset_path = '/home/alon/Documents/DataSet/'
    main(dataset_path)
