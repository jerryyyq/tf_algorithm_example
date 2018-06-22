import os
from PIL import Image

img = Image.open("data_source/olivettifaces.gif")

label = 0
index = 0

os.mkdir("data_source/olivettifaces/00")

for row in range(20):
    for column in range(20):
        region = (column * 47, row * 57, (column+1)*47, (row+1)*57)
        cropImg = img.crop(region)
        file = "data_source/olivettifaces/{:0>2}/{}.gif".format(label, index)
        cropImg.save(file)

        if 9 == column or 19 == column:
            index = 0
            label += 1
            os.mkdir("data_source/olivettifaces/{:0>2}".format(label))

        index += 1

