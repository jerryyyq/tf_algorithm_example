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
    training files and test files and lable file, 
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
            out_dir:    a directory for save tf_records files for output, it must is a absolute path and must exist.
            image_type: the image's type of input.

        Returns:
            none.
        """

        self.__in_dir = in_dir
        self.__out_dir = out_dir
        self.__image_type = image_type

        self.__generate_label_file_list()


    def generate_tf_record_files(self, resize, channel = 1, one_record_max_imgaes = 1024):
        """generate tf_record files to out_dir.

        Args:
            resize:                 example: (250, 151)
            channel:                record images channel of output -> 1 = Gray Scale Image; 3 = RGB; 4 = RGBA.
            one_record_max_imgaes:  the max image amount of one tf_record file.

        Returns:
            none.
        """
        self.__generate_tf_record_files('train', self.__train_image_with_breed, resize, channel, one_record_max_imgaes)
        self.__generate_tf_record_files('test', self.__test_image_with_breed, resize, channel, one_record_max_imgaes)



    def __generate_tf_record_files(self, prefix, image_with_breed, resize, channel = 1, one_record_max_imgaes = 1024):
        # delete info file
        info_file = os.path.join(self.__out_dir, prefix + '_record.inf')
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

        writer.close()
        self.__save_record_info_to_file(prefix, current_index - 1, image_amount)


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
        print("breed_label: {}".format(self.__breed_label))
        label_breed = sorted(self.__breed_label.items(), key = lambda item:item[1])
        with open(self.__out_dir + '/label.txt', 'w') as f:
            for item in label_breed:
                f.write(str(item[1]) + '\t' + item[0] + '\n')


        
if __name__ == '__main__':
    one_Set = Img2TFRecord('/home/yangyuqi/Downloads/Images', '/tmp/tf_out/tmp1')
    one_Set.generate_tf_record_files( (250, 151) )
