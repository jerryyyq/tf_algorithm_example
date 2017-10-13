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

    
    
def read_int32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return int(numpy.frombuffer(bytestream.read(4), dtype=dt)[0])


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.astype(numpy.int32)
    return labels_one_hot
