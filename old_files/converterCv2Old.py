import os, sys
import random
import numpy as np
from PIL import Image, ImageOps
import skimage.io as io
import cv2


def main(main_path):
    # image_generator(dir_path=(main_path + 'Blue_Dream/ready'))
    # image_generator(dir_path=(main_path + 'Blue_Dream/ready'))
    Concatenate('/home/alon/PycharmProjects/NeuralNetwork/Converter/npy/')
    print("finished main process")


def image_generator(dir_path, num_images=500):
    i = 0
    final_array = np.ndarray(shape=(num_images, 25, 25, 3))
    # label_array = np.ndarray(shape=(500, 2))
    for _ in os.listdir(dir_path):
        if _ != 'ready' and i < num_images:
            try:
                img = cv2.imread(dir_path + '/' + _, 1)
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    print("failed image " + _)
                    pass
                final_array[i] = img
                # label_array[i] = [1, 0]
                i = i + 1
                print("image processing successful " + str(i) + '/' + str(num_images))
            except ValueError:
                print("failed image " + _)
                pass

    try:
        print("saving .npy file....")
        np.save(dir_path + '/Blue_Dream', final_array)
        print("successfully saved Lemon_Haze.npy file")
    except ValueError:
        print("failed to save the .npy file.")

def Concatenate(dir_path, limit = 5000):
    i = 0
    images = []

    for _ in os.listdir(dir_path):
        if _ != 'ready' and i < limit:
            try:
                images.append(np.load(dir_path + _))
                print(_ + ' has been successfully loaded')
            except Exception:
                print("failed loading " + _)
                pass

    if len(images) > 0:
        images = np.concatenate(images)

        try:
            print("saving .npy file....")
            np.save(dir_path + '/RGB', images)
            print("successfully saved RGB.npy file")
        except ValueError:
            print("failed to save the .npy file.")
    else:
        print ('error, not enough arrays to concatenate')



if __name__ == '__main__':
    main_path = '/home/alon/Documents/DataSet/'
    main(main_path)
