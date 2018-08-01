# 轮廓和边界计算的一些算法展示：https://blog.csdn.net/liqiancao/article/details/55670749

import os, glob
from PIL import Image

# 源图片尺寸：1280 * 1536，只取中间的 1 列    # 源图片尺寸：2560 * 3072，只取中间的 3 列
WIDTH = 215     # 430
HEIGHT = 50     # 一般从第 8 个（index 为 7）开始就是杆部分

IMAGE_DIR = '/home/yangyuqi/work/crack_sample_small/is'       # 'is' and 'not'
files = glob.glob( os.path.join( IMAGE_DIR, '*.BMP' ) )

for image in files:
    file_name = os.path.basename(image)
    name_fields = file_name.split('.')

    index = 0
    img = Image.open(image)
    for row in range( img.height // HEIGHT ):
        for column in range(2, 3):    # 只取第三列
            region = (column * WIDTH, row * HEIGHT, (column+1)*WIDTH, (row+1)*HEIGHT)
            cropImg = img.crop(region)
            file = IMAGE_DIR + "/{}-{}-{}.gif".format(name_fields[0], row, column)
            cropImg.save(file)

            index += 1

    # 处理最后一行
    for column in range(2, 3):
        region = (column * WIDTH, img.height - HEIGHT, (column+1)*WIDTH, img.height)
        cropImg = img.crop(region)
        file = IMAGE_DIR + "/{}-{}-{}.gif".format(name_fields[0], row, column)
        cropImg.save(file)


'''
# 做 record 生成
from img_to_tf_record import Img2TFRecord
one_Set = Img2TFRecord('/home/yangyuqi/work/crack_sample_small/class', '/home/yangyuqi/work/crack_sample_small/tf_record', 'gif')
one_Set.generate_tf_record_files( (215, 50) )

'''