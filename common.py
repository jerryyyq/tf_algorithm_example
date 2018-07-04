import os
import numpy
import collections
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

'''
高维数组变为一维数组

例如：
a = [[['a','b'], ['c', 'd', 'e', 'f']], [[1, 2, 3], [4, 5, 6, 7, 8]]]
return: ['a', 'b', 'c', 'd', 'e', 'f', 1, 2, 3, 4, 5, 6, 7, 8]
'''
def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    
    return b


'''
将多维内容列表里的所有字符串展开成为一个一维字符列表
例如：
content_list = [[['a','d'], ['cdd', 'ef']], [['12', '1'], ['4567', '4']]]
return: ['a', 'd', 'c', 'd', 'd', 'e', 'f', '1', '2', '1', '4', '5', '6', '7', '4']
'''
def content_list_to_all_words(content_list):
    all_words = []
    contents = flatten(content_list)
    for item in contents:
        all_words += [word for word in item]

    return all_words


'''
所有的不重复的字，按出现次数从高到低排序后，将他们映射为： 0 ～ (字数 - 1) 的映射字典
返回不重复的字的元组和映射的字典
例如：
all_words = ['b', 'c', 'a', 'f', 'd', 'c', 'a', 'e']
return: (('c', 'a', 'b', 'f', 'd', 'e'), {'c': 0, 'a': 1, 'b': 2, 'f': 3, 'd': 4, 'e': 5})
'''
def all_words_to_word_num_map( all_words ):
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 每个字映射为一个数字ID
    return words, dict(zip(words, range(len(words))))


'''
将句子列表中的所有字符，替换为 map 中的向量值
例如： 
sentence_list = ['bbaa', 'ceff', 'ade']
word2vec_map = {'a': 1, 'b': 5, 'c': 0, 'd': 4, 'e': 3, 'f': 2}
return: [[5, 5, 1, 1], [0, 3, 2, 2], [1, 4, 3]]
'''
def words_to_vectors(sentence_list, word2vec_map):
    return [ list(map(lambda word: word2vec_map[word], sentence)) for sentence in sentence_list ]


'''
将句子列表按 batch_size 分批整理为 RNN 需要的数据序列。每个的元素的长度按每分批里的最长元素长度，不同的分批长度不同
sentence_vec_list：原始数据，一般为已转为向量的所有输入内容
batch_size：一批数据有多少个。在 NLP 中一般值的是一批处理多少个句子
fill_value：对长度不足的元素，用什么值来填充

例如：
sentence_vec_list = [[5, 5, 1, 1], [0, 3, 2], [1, 4, 3], [2, 5, 6, 7, 8, 9, 3]]
batch_size = 2
fill_value = 999

return x_batches: [[[5, 5, 1, 1], [0, 3, 2, 999]], [[1, 4, 3, 999, 999, 999, 999], [2, 5, 6, 7, 8, 9, 3]]]
       y_batches: [[[5, 1, 1, 1], [3, 2, 999, 999]], [[4, 3, 999, 999, 999, 999, 999], [5, 6, 7, 8, 9, 3, 3]]]
'''
def sentence_list_to_x_y_batches(sentence_vec_list, batch_size, fill_value):
    n_chunk = len(sentence_vec_list) // batch_size  # 一共有多少个块
    x_batches = []  # shape: [n_chunk, batch_size, length]
    y_batches = []

    for i in range(n_chunk):  
        start_index = i * batch_size  
        end_index = start_index + batch_size  
        
        batches = sentence_vec_list[start_index : end_index]  
        length = max(map(len, batches))                        # 本 batche 中最长的句子的长度

        xdata = [[fill_value] * length for row in range(batch_size)]
        for row in range(batch_size):  
            xdata[row][:len(batches[row])] = batches[row]

        ydata = [[*xdata[i][1:], xdata[i][-1]] for i in range(len(xdata))]

        x_batches.append(xdata)                               # 所有的句子，但其中每 batche 一组，每一组的长度是一样的
        y_batches.append(ydata)

    return x_batches, y_batches 



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
    






