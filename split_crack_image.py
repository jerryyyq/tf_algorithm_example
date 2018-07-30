import os, glob
from PIL import Image

# 源图片尺寸：2560 * 3072，只取中间的 3 列
WIDTH = 430
HEIGHT = 350

IMAGE_DIR = '/home/yangyuqi/work/crack_sample/is'       # 'is' and 'not'
files = glob.glob( os.path.join( IMAGE_DIR, '*.BMP' ) )

for image in files:
    file_name = os.path.basename(image)
    name_fields = file_name.split('.')

    index = 0
    img = Image.open(image)
    for row in range(8):
        for column in range(1, 4):
            region = (column * WIDTH, row * HEIGHT, (column+1)*WIDTH, (row+1)*HEIGHT)
            cropImg = img.crop(region)
            file = IMAGE_DIR + "/{}-{}-{}.gif".format(name_fields[0], row, column)
            cropImg.save(file)

            index += 1

    # 处理最后一行
    for column in range(1, 4):
        region = (column * WIDTH, img.height - HEIGHT, (column+1)*WIDTH, img.height)
        cropImg = img.crop(region)
        file = IMAGE_DIR + "/{}-{}-{}.gif".format(name_fields[0], 8, column)
        cropImg.save(file)