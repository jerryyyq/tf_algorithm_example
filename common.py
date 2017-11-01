import os
import numpy
import tensorflow as tf


# 打印一个 变量的值
def echo_tensor(sess, tensor, prefix = ''):
    # 注意： print() 显示时会把元素之间的逗号去掉
    if( isinstance(tensor, tf.Tensor) or isinstance(tensor, tf.Variable) ):
        print( '{0} tensor.shape = {1}, tensor = {2}{3}'.format(prefix, sess.run(tf.shape(tensor)), sess.run(tensor), os.linesep) )
    else:
        print( '{0} not_tensor = {1}{2}'.format(prefix, tensor, os.linesep) )
        
        

# 从 csv 文件读取数据，加载解析，创建批次读取张量多行数据
# 调用举例: read_csv(4, ['1.csv', '2.csv'], [[0], [0.], ['']])
# 
# 如果 csv 文件中数据为：
# 1,2,3,4
# 11,12,13,14
# 21,22,23,24
# 31,32,33,34
# 41,42,43,44
# 51,52,53,54
# 
# 那么输出是：
# [[21, 51, 1, 61], [22, 52, 2, 62], [23, 53, 3, 63], [24, 54, 4, 64]]
# 
def read_csv(batch_size, file_name, record_defaults):
    file_path = list( map(lambda name: os.path.join(os.getcwd(), name), file_name) )
    print('file_path = ', file_path, '\r\n')
    file_queue = tf.train.string_input_producer(file_path) 

    reader = tf.TextLineReader(skip_header_lines = 1)

    key, value = reader.read(file_queue)
    print(key, value)

    decoded = tf.decode_csv(value, record_defaults, ',')
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size) # 读取文件，加载张量batch_size行

    
# 从二进制字节流中读取 4 个字节转为 int32 返回
def read_int32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return int(numpy.frombuffer(bytestream.read(4), dtype=dt)[0])

# 将：[0, 1, 2, 3] 转化为：[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.astype(numpy.int32)
    return labels_one_hot


# dataset 是个二维数组，第一维度为所有的品种，第二个维度为该品种下的所有的图片的文件名
# record_location 为生成的 tfrecords 保存的路径（需要可写），会按照 one_record_count 个图片生成一个 tfrecords 文件。生成的文件会自动在名字段添加 -1、2、3... 的索引
# resize 是需要对图片统一调整的大小：[长，宽]
# channels 是 1 表示要转化为灰度图
def imageset_to_records_files(dataset, record_location, resize, \
                              channels = 1, image_type = 'jpg', one_record_count = 100):
    writer = None
    current_index = 0
    image_number = 0
    sess = tf.Session()
    
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if image_number % one_record_count == 0:
                if writer:
                    writer.close()
                    
                record_filename = "{record_location}-{current_index}.tfrecords". \
                format(record_location=record_location, current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
                current_index += 1
                    
            image_number += 1

            image_file = tf.read_file(image_filename)

            try:
                if( 'jpg' == image_type ):
                    image = tf.image.decode_jpeg(image_file, channels)
                elif( 'png' == image_type ):
                    image = tf.image.decode_png(image_file, channels)
            except:
                print('image decode fail: ', image_filename)
                continue

            resized_image = tf.image.resize_images( image, resize )

            #image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_loaded = sess.run(tf.cast(resized_image, tf.uint8))
            image_bytes = image_loaded.tobytes()
            image_label = breed.encode("utf-8")

            feature = {
                'label':tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_label])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_bytes]))
            }
            
            example = tf.train.Example( features=tf.train.Features(feature = feature) )

            writer.write(example.SerializeToString())

    writer.close()                

# pil_imageset_to_records_files 的速度是 imageset_to_records_file 的好几倍
def pil_imageset_to_records_files(dataset, record_location, resize, \
                              channels = 1, image_type = 'jpg', one_record_count = 100):
    from PIL import Image
    writer = None
    current_index = 0
    image_number = 0
    
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if image_number % one_record_count == 0:
                if writer:
                    writer.close()
                    
                record_filename = "{record_location}-{current_index}.tfrecords". \
                format(record_location=record_location, current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
                current_index += 1
                    
            image_number += 1

            img = None
            try:
                img = Image.open( image_filename )
                if( 1 == channels ):
                    img = img.convert('L')
                img = img.resize( resize )
            
            except:
                print('image decode fail: ', image_filename)
                continue

            image_bytes = img.tobytes()
            image_label = breed.encode("utf-8")

            feature = {
                'label':tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_label])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_bytes]))
            }
            
            example = tf.train.Example( features=tf.train.Features(feature = feature) )

            writer.write(example.SerializeToString())

    writer.close()


# records_path: './record_files/train-image/*.tfrecords'
# image_shape: [250, 151, 1]
# batch_size: 3
def read_images_from_tfrecords(records_path, image_shape, batch_size):
    file_path = tf.train.match_filenames_once( records_path )
    file_queue = tf.train.string_input_producer( file_path )
    reader = tf.TFRecordReader()
    _, serialized = reader.read( file_queue )

    feature = {
        'label':tf.FixedLenFeature([], tf.string),
        'image':tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_single_example( serialized, features = feature )

    record_image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.cast( features['label'], tf.string )

    image = tf.reshape(record_image, [250, 151, 1])

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue)
    return image_batch, label_batch
    






