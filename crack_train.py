import sys, os
sys.path.append('../tf_algorithm_example')
from DL_CNN import DL_CNN

WIDTH = 215     # 430
HEIGHT = 50     # 一般从第 8 个（index 为 7）开始就是杆部分

#np.set_printoptions(threshold=np.inf)

one_cnn = DL_CNN('/home/yangyuqi/work/crack_sample_small/tf_record_two', HEIGHT, WIDTH, 1, 2, 'model/crack/crack_best.ckpt')
one_cnn.train(1000, 96, 4)

one_cnn.verify(24, 4)

# one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_small/class_two/6-crack/29-2-9-2.bmp')  # is
# one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_small/class_two/5-no-crack/29-2-14-2.bmp')  # not
# one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_small/class_two/5-no-crack/2-2-13-2.bmp')  # not

# one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_small/is/24-2-19-2.bmp')  # not

# ----------------------------------------------------------------------------------------
"""
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

def detect_one_region(cropImg):
    '''
    # cropImg = cropImg.convert('L')
    img_ndarray = np.asarray(cropImg, dtype='float32') / 255

    batch_one_image = np.empty((1, HEIGHT, WIDTH, 1))
    batch_one_image[0] = img_ndarray.reshape(HEIGHT, WIDTH, 1)

    tensor_one_image = tf.convert_to_tensor(batch_one_image)
    tensor_one_image = tf.cast(tensor_one_image, tf.float32)
    '''

    batch_one_image = np.empty((1, HEIGHT, WIDTH))
    batch_one_image[0] = np.asarray(cropImg)

    tensor_one_image = tf.reshape( batch_one_image, (1, HEIGHT, WIDTH, 1) )
    tensor_one_image = tf.cast(tensor_one_image, tf.float32) * (1./255)

    ret = one_cnn.recognition(tensor_one_image, False)
    print('recognition ret = ', ret)
    return ret[0]



# img = Image.open('/home/yangyuqi/work/crack_sample_small/not/26-2.BMP')
img = Image.open('/home/yangyuqi/work/crack_check_python/crack_sample_small/is/49-2.BMP')
# resize_img = img.resize((img.width // 2, img.height // 2), Image.ANTIALIAS)
print('img', img)

for row in range( 7, img.height // HEIGHT ):
    for column in range(2, 3):    # 只取第三列
        region = (column * WIDTH, row * HEIGHT, (column+1)*WIDTH, (row+1)*HEIGHT)
        print('region = ', region)
        cropImg = img.crop(region)
        print('cropImg = ', cropImg)

        temp_file = os.path.join('/tmp', 'region{}-{}.gif'.format(row, column))
        if os.path.exists(temp_file):
            os.remove(temp_file)

        cropImg.save(temp_file)
        ret = one_cnn.recognition_one_image(temp_file)

        '''
        cropImg = cropImg.convert('L')
        ret = detect_one_region(cropImg)
        '''
"""