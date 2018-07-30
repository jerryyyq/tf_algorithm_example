# Copyright 2018 YangYuQi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    in_dir include breed and all images, it must like below:
    --------------------------------------------------------
    in_dir/breed1/xxx.jpg
    in_dir/breed1/yyy.jpg
    ...
    in_dir/breed1/zzz.jpg

    in_dir/breed2/xxx.jpg
    in_dir/breed2/yyy.jpg
    ...
    in_dir/breed2/zzz.jpg

    ......

    in_dir/breedn/xxx.jpg
    in_dir/breedn/yyy.jpg
    ...
    in_dir/breedn/zzz.jpg
    --------------------------------------------------------


    out_dir will save output tf_redores files, it will include 
    training files and test files and label file, 
    the test images accounts for 20% of the total accounts.
    ????_record.inf include the image's amount of every record file.
    The directory like below:
    --------------------------------------------------------
    out_dir/train_1.tfr 
    out_dir/train_2.tfr 
    ...
    out_dir/train_n.tfr 

    out_dir/test_1.tfr 
    out_dir/test_2.tfr 
    ...
    out_dir/test_n.tfr 

    out_dir/label.txt
    out_dir/train_record.inf
    out_dir/test_record.inf
    --------------------------------------------------------
"""

import glob, os
import datetime
import tensorflow as tf
from PIL import Image


# because my tensorflow is CPU version, it running will report 'warning', write below line for disable this 'warning'.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Img2TFRecord(object):

    def __init__(self, in_dir, out_dir, image_type = 'jpg'):
        """Initializes function and write labels to out_dir/label.txt.

        Args:
            in_dir:     a directory containing all breed images, it must is a absolute path and must exist.
            out_dir:    a directory for save tf_records files for output, it must is a absolute path.
            image_type: the image's type of input.

        Returns:
            none.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.__in_dir = in_dir
        self.__out_dir = out_dir
        self.__image_type = image_type



    def generate_tf_record_files(self, resize, channel = 1, one_record_max_imgaes = 1024):
        """generate tf_record files to out_dir.

        Args:
            resize:                 (width, height). example: (250, 151)
            channel:                record images channel of output -> 1 = Gray Scale Image; 3 = RGB; 4 = RGBA.
            one_record_max_imgaes:  the max image amount of one tf_record file.

        Returns:
            none.
        """
        self.__generate_label_file_list()

        self.__generate_tf_record_files('train', self.__train_image_with_breed, resize, channel, one_record_max_imgaes)
        self.__generate_tf_record_files('test', self.__test_image_with_breed, resize, channel, one_record_max_imgaes)

        print( "generate_tf_record_files file finish.\n" )


    def read_train_images_from_tf_records( self, batch_size, reshape, label_size = 0 ):
        """batch read train images from tf_record files. And convert image data to 0.0 ~ 1.0 
        Args:
            batch_size:     how many images in one batch, example: 10
            reshape:        [height, width, channel], must be equal image size. example: [250, 151, 1]
            label_size:     how many class of total label, example: 40

        Returns:
            image_batch(float32), label_batch(float32)
            if label_size > 0 then: label_batch is one hot type, example: [batch_size][0] == [0., 0., 1., 0., 0.], [batch_size][1] == [0., 0., 0., 1., 0.]
            else: label_batch only label number, example: [batch_size][0] == 2., [batch_size][1] == 3.
        """

        records_path = os.path.join(self.__out_dir, 'train_*.tfr')
        return self.read_images_from_tf_records(records_path, batch_size, reshape, label_size)

    def read_test_images_from_tf_records( self, batch_size, reshape, label_size = 0 ):
        """batch read test images from tf_record files. And convert image data to 0.0 ~ 1.0 
        Args:
            batch_size:     how many images in one batch, example: 10
            reshape:        [height, width, channel], must be equal image size. example: [250, 151, 1]
            label_size:     how many class of total label, example: 40

        Returns:
            image_batch(float32), label_batch(float32)
            if label_size > 0 then: label_batch is one hot type, example: [batch_size][0] == [0., 0., 1., 0., 0.], [batch_size][1] == [0., 0., 0., 1., 0.]
            else: label_batch only label number, example: [batch_size][0] == 2., [batch_size][1] == 3.
        """

        records_path = os.path.join(self.__out_dir, 'test_*.tfr')
        return self.read_images_from_tf_records(records_path, batch_size, reshape, label_size)
    

    @staticmethod
    def read_images_from_tf_records( records_path, batch_size, reshape, label_size = 0 ):
        """batch read images from tf_record files. And convert image data to 0.0 ~ 1.0 

        Args:
            records_path:   example: '/tmp/tf_out/tmp1/train_*.tfr'
            batch_size:     how many images in one batch, example: 10
            reshape:        [height, width, channel], must be equal image size. example: [250, 151, 1]
            label_size:     how many class of total label, example: 40

        Returns:
            image_batch, label_batch
        """

        file_path = tf.train.match_filenames_once( records_path )
        file_queue = tf.train.string_input_producer( file_path )
        reader = tf.TFRecordReader()
        _, serialized = reader.read( file_queue )

        feature = {
            'label': tf.FixedLenFeature([], tf.int64),   # tf.string
            'image': tf.FixedLenFeature([], tf.string)
        }

        features = tf.parse_single_example( serialized, features = feature )

        record_image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast( features['label'], tf.int64 ) # tf.string

        image = tf.reshape( record_image, reshape )  # [250, 151, 1]  # reshape
        image = tf.cast(image, tf.float32) * (1./255)

        if 0 < label_size:
            label = tf.one_hot(label, label_size, 1, 0)

        label = tf.cast( label, tf.float32 )

        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue)
        return image_batch, label_batch
        

    def __generate_tf_record_files(self, prefix, image_with_breed, resize, channel = 1, one_record_max_imgaes = 1024):
        # delete info file
        info_file = os.path.join(self.__out_dir, prefix + '_record.inf')
        if os.path.exists(info_file):
            os.remove(info_file)

        # main code
        writer = None
        current_index = 0
        image_amount = 0
        for i, label_file in enumerate(image_with_breed):
            if 0 == i % one_record_max_imgaes:
                if writer:
                    writer.close()
                    writer = None
                    self.__save_record_info_to_file(prefix, current_index - 1, image_amount)

                record_file = os.path.join(self.__out_dir, prefix + '_' + str(current_index) + '.tfr')
                writer = tf.python_io.TFRecordWriter(record_file)
                current_index += 1
                image_amount = 0

            img = None
            try:
                img = Image.open( label_file[1] )
                if 1 == channel:
                    img = img.convert('L')
                elif 3 == channel:
                    img = img.convert('RGB')
                elif 4 == channel:
                    img = img.convert('RGBA')

                img = img.resize( resize )
            
            except Exception as e:
                print('image decode fail: ', label_file[1], ' err: ', e)
                continue

            image_bytes = img.tobytes()
            image_label = label_file[0]

            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value = [image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_bytes]))
            }
            
            example = tf.train.Example( features=tf.train.Features(feature = feature) )

            writer.write(example.SerializeToString())
            image_amount += 1

        if writer:
            writer.close()
            writer = None
            self.__save_record_info_to_file(prefix, current_index - 1, image_amount)

        print( "generate {} record file finish, total {} files.\n".format(prefix, current_index) )


    def __save_record_info_to_file(self, prefix, record_index, image_amount):
        info_file = os.path.join(self.__out_dir, prefix + '_record.inf')
        with open(info_file, 'a') as f:
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            f.write(nowTime + '\t' + prefix + '_' + str(record_index) + '.tfr\t' + str(image_amount) + '\n')


    def __generate_label_file_list(self):
        """save out_dir/label.txt and add (label, file_path) to self.__train_image_with_breed and self.__test_image_with_breed
            self.__????_image_with_breed: [(label, img1 absolute path), (label, img2 absolute path), ... (label, imgn absolute path)]
        """

        self.__breed_label = {}
        self.__train_image_with_breed = []
        self.__test_image_with_breed = []

        # get all jpg files absolute path list
        image_file_names = glob.glob( self.__in_dir + '/*/*.' + self.__image_type )
        image_file_names.sort()

        index = 0
        for i, file_path in enumerate(image_file_names):
            breed = file_path.split('/')[-2]
            label = self.__breed_label.get(breed)
            if label is None:
                label = index
                self.__breed_label[breed] = label
                index += 1
                print("breed = {}, label = {}\n".format(breed, self.__breed_label[breed]))

            if 0 == i % 5:  # 20% image put in test set.
                self.__test_image_with_breed.append((label, file_path))
            else:
                self.__train_image_with_breed.append((label, file_path))

        self.__save_breed_label_to_file()


    def __save_breed_label_to_file(self):
        label_breed = sorted(self.__breed_label.items(), key = lambda item:item[1])
        with open(self.__out_dir + '/label.txt', 'w') as f:
            for item in label_breed:
                f.write(str(item[1]) + '\t' + item[0] + '\n')


        
if __name__ == '__main__':
    one_Set = Img2TFRecord('/home/yangyuqi/Downloads/Images', '/tmp/tf_out/tmp1')

    # test generate function
    #one_Set.generate_tf_record_files( (250, 151) )

    # test read function
    image_batch, label_batch = one_Set.read_train_images_from_tf_records([250, 151, 1], 5)
    init = tf.initialize_all_variables()   # tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        for i in range(3):
            img, lab = sess.run([image_batch, label_batch])
            print(img, lab)

        #关闭线程  
        coord.request_stop()  
        coord.join(threads)
