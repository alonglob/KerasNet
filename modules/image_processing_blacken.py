from PIL import Image
import os
from multiprocessing import Process

paths_0 = ["/home/alon/Documents/DataSet/Blue_Dream/",
            "/home/alon/Documents/DataSet/Blue_Dream_Processed/"]

paths_1 =  ["/home/alon/Documents/DataSet/Lemon_Haze/",
            "/home/alon/Documents/DataSet/Lemon_Haze_Processed/"]

def processing(im):
    width, height = im.size

    rect_size = (300, 150)
    rect_pos = (0, height - 150)

    rect = Image.new("RGBA", rect_size, (0, 0, 0, 0))
    im.paste(rect, rect_pos)
    return im

def blacken(path):
    dirs = os.listdir(path[0])
    for item in dirs:
        if os.path.isfile(path[0] + item):
            im = Image.open(path[0] + item).convert("RGBA")
            processed_image = processing(im)

            processed_image.save(path[1] + item + '.png', 'PNG')

p1 = Process(target=blacken, args=(paths_0,))
p2 = Process(target=blacken, args=(paths_1,))


p1.start()
p2.start()
p1.join()
p2.join()
