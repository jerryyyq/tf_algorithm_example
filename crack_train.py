import sys, os
from DL_CNN import DL_CNN
#np.set_printoptions(threshold=np.inf)

'''
WIDTH = 215     # mini: 107
HEIGHT = 50     # mini: 25          一般从第 8 个（index 为 7）开始就是杆部分

one_cnn = DL_CNN('/home/yangyuqi/work/crack_sample_small/tf_record_two', HEIGHT, WIDTH, 1, 2, 'model/crack/crack_best.ckpt')
one_cnn.train(1000, 96, 4)
'''

WIDTH = 107
HEIGHT = 25

one_cnn = DL_CNN('/home/yangyuqi/work/crack_sample_mini/tf_record_two', HEIGHT, WIDTH, 1, 2, 'model/crack_mini/crack_best.ckpt')
# one_cnn.train(500, 84, 3, True, 0.01, 21)
one_cnn.train(500, 89, 5, False, 0.01)

#one_cnn = DL_CNN('/home/yangyuqi/work/crack_sample_mini/tf_record_two', HEIGHT, WIDTH, 1, 2, 'model/crack_mini/crack_best.ckpt-0')
# one_cnn.verify(21, 3)

# is
print('---------------- is ----------------')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/class_two/6-crack/24-2-9-2.bmp') 
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/class_two/6-crack/29-2-9-2.bmp')

# not
print('---------------- not ----------------')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/not/2-2-10-2.bmp')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/not/2-2-14-2.bmp')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/not/23-2-6-2.bmp')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/not/23-2-11-2.bmp')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/not/27-2-7-2.bmp')
one_cnn.recognition_one_image('/home/yangyuqi/work/crack_sample_mini/not/27-2-22-2.bmp')


# ----------------------------------------------------------------------------------------
print('---------------- image ----------------')
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
img = Image.open('/home/yangyuqi/work/crack_sample_mini/is/49-2.BMP')
#resize_img = img.resize((img.width // 2, img.height // 2), Image.NEAREST)   # for mini
resize_img = img
print('resize_img = ', resize_img)

for row in range( 7, resize_img.height // HEIGHT ):
    for column in range(2, 3):    # 只取第三列
        region = (column * WIDTH, row * HEIGHT, (column+1)*WIDTH, (row+1)*HEIGHT)
        cropImg = resize_img.crop(region)

        temp_file = os.path.join('/tmp', 'region{}-{}.bmp'.format(row, column))
        if os.path.exists(temp_file):
            os.remove(temp_file)

        
        cropImg.save(temp_file)
        print('temp_file is ', temp_file)
        ret = one_cnn.recognition_one_image(temp_file)
        '''

        cropImg = cropImg.convert('L')
        ret = detect_one_region(cropImg)
        '''