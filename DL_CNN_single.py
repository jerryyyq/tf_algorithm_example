######################## 机器学习 CNN 类 ##############################
# Author: 杨玉奇
# email: yangyuqi@sina.com
# url: https://github.com/jerryyyq/tf_algorithm_example
# copyright yangyuqi
# 著作权归作者 杨玉奇 所有。商业转载请联系作者获得授权，非商业转载请注明出处。
# date: 2018-03-28
###################################################################

# 代码来自：https://www.jianshu.com/p/3e5ddc44aa56

# 一个 C++ 实现的神经网络参考代码：https://blog.csdn.net/xingchenbingbuyu/article/details/53736154

import os
import tensorflow as tf

from img_to_tf_record import Img2TFRecord

# numpy and PIL only for test function do_recognition_one_image, not must for CNN
import numpy as np
from PIL import Image


MODEL_FILE_NAME = 'best.ckpt'


def convolutional_layer(layer_index, data, kernel_size, bias_size, pooling_size):
    kernel = tf.get_variable("conv_{}".format(layer_index), kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable("bias_{}".format(layer_index), bias_size, initializer=tf.random_normal_initializer())

    # 卷积
    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')

    # 线性修正，将所有的元素中的负数置为零
    linear_output = tf.nn.relu(tf.add(conv, bias))

    # 池化
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
    return pooling


def linear_layer(linear_index, data, weights_size, biases_size):
    weights = tf.get_variable("weigths_{}".format(linear_index), weights_size, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases_{}".format(linear_index), biases_size, initializer=tf.random_normal_initializer())
    return tf.add(tf.matmul(data, weights), biases)


''''' 
创建卷积层 
@:param data           输入图像 
@:param class_size     总共有多少个类别 
@:param channel        图像的通道数是多少
@:return Tensor        计算出的 label(one hot 格式)
''' 
def convolutional_neural_network(data, class_size, channel):
    '''
    使用 tf 创建 2 层 cnn，5 * 5 的 filter，输入为灰度，所以：

    第一层的 channel 是 1，图像高为 57，宽为 47, 输出 32 个 filter，maxpooling 是缩放一倍
    第二层的输入为 32 个channel，高是 29，宽为 24, 输出为 64 个 filter，maxpooling 是缩放一倍

    全连接层的输入为 64 个channel，高是 15，宽为 12, 将输入的 15 * 12 * 64 的多维度，
    进行 flatten，映射到 1024 个数据上，
    输出层的输入为 1024 个数据，映射到 onehot 类别上，最后输出的向量 shape 为：[batch_size, class_size]
    '''

    # 输入样本为：[batch_size, 57, 47, 1]
    # 经过第一层卷积神经网络后，得到的张量 shape 为：[batch_size, 29, 24, 32]
    layer1_output = convolutional_layer(
        layer_index = 1, 
        data = data,
        kernel_size = [5, 5, channel, 32],
        bias_size = [32],
        pooling_size = [1, 2, 2, 1]   # 用 2x2 模板做池化
    )

    # 经过第二层卷积神经网络后，得到的张量 shape 为：[batch_size, 15, 12, 64]
    layer2_output = convolutional_layer(
        layer_index = 2,
        data = layer1_output,
        kernel_size = [5, 5, 32, 64],
        bias_size = [64],
        pooling_size = [1, 2, 2, 1]
    )

    # 全连接层。将卷积层张量数据拉成 2-D 张量只有一列的列向量，得到的张量 shape 为：[batch_size, 1024]
    layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)  # 得到的张量 shape 为：[batch_size, 15 * 12 * 64]
    layer3_output = tf.nn.relu(
        linear_layer(
            linear_index = 1,
            data = layer2_output_flatten,
            weights_size = [15 * 12 * 64, 1024],
            biases_size = [1024]
        )
    )
    # 减少过拟合，随机让某些权重不更新
    # layer3_output = tf.nn.dropout(layer3_output, 0.8)
    
    # 输出层，得到的张量 shape 为：[batch_size, class_size]
    output = linear_layer(
        linear_index = 2,
        data = layer3_output,
        weights_size = [1024, class_size],      # 根据类别个数定义最后输出层的神经元
        biases_size = [class_size]
    )

    return output


def loss(feature, label):
    softmax_out = tf.nn.softmax_cross_entropy_with_logits(logits=feature, labels=label)
    return tf.reduce_mean(softmax_out)

# 返回计算出的 class(非 one hot 格式)
def do_recognition(image_set, class_size, channel):
    neural_out = convolutional_neural_network(image_set, class_size, channel)
    return tf.argmax(neural_out, 1)


def do_train(model_dir):    # 'model/olivettifaces'
    if not os.path.exists( model_dir ):
        os.makedirs(model_dir)
        print("create the directory: %s" % model_dir)

    one_set = Img2TFRecord('data_source/olivettifaces', 'tf_record/olivettifaces', 'gif')

    batch_size = 5
    image_set, label_set = one_set.read_train_images_from_tf_records(batch_size, [57, 47, 1], 40)  # height, width, channel
    verify_image_set, verify_label_set = one_set.read_test_images_from_tf_records(batch_size, [57, 47, 1], 40)  # 为了方法二

    neural_out = convolutional_neural_network(image_set, 40, 1)
    loss_out = loss(neural_out, label_set)
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss_out)
    # 将 loss 与 optimizer 保存以供 tensorboard 使用
    tf.scalar_summary('loss', loss_out)            # 高版本 -> tf.summary.scalar('loss', loss_out)
    # tf.scalar_summary('optimizer', optimizer)      # 高版本 -> tf.summary.scalar('accuracy', accuracy)

    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    tf.get_variable_scope().reuse_variables()  # 为了方法二对准确率进行计算，复用当前变量
    calculate_label = do_recognition(verify_image_set, 40, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(calculate_label, tf.argmax(verify_label_set, 1)), tf.float32))
    # 将 accuracy 保存以供 tensorboard 使用
    tf.scalar_summary('accuracy', accuracy)        # 高版本 -> tf.summary.scalar('accuracy', accuracy)

    merged_summary_op = tf.merge_all_summaries()   # 高版本 -> tf.summary.merge_all()

    # 用于保存训练结果的对象
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()   # 高版本 -> tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.train.SummaryWriter('my_graph/olivettifaces', sess.graph) # 高版本 -> tf.summary.FileWriter('my_graph/olivettifaces', graph = tf.get_default_graph())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        best_loss = float('Inf')
        epoch_loss = 0

        train_sample_number = 320
        for i in range(int(train_sample_number/batch_size) * 500):   # 训练 500 轮
            _, cost, summary = sess.run([optimizer, loss_out, merged_summary_op])
            summary_writer.add_summary(summary, i)
            epoch_loss += cost

            '''
            # 方法一，每一轮比较一次
            # 共有学习样本 320 个，所以每一轮应该是 320 / batch_size 次
            if i % (320/batch_size) == 0:
                print(i, ' : ', epoch_loss)
                if best_loss > epoch_loss:
                    best_loss = epoch_loss

                    save_path = saver.save( sess, os.path.join(model_dir, MODEL_FILE_NAME) )
                    print( "Model saved in file: {}, epoch_loss = {}" . format(save_path, epoch_loss) )
                    if 0.0 == epoch_loss:
                        print('epoch_loss == 0.0, exited!')
                        break

                epoch_loss = 0
            '''

            
            # 方法二，按准确率结束学习
            # 获取测试数据的准确率
            # 每一轮测试一下
            if i % int(train_sample_number/batch_size) != 0:
                continue

            acc = accuracy.eval()
            print(i, "acc:", acc, "  loss:", epoch_loss)
            epoch_loss = 0
            # 准确率大于0.98时保存并退出
            if acc > 0.98 and i > 2:
                saver.save(sess, os.path.join(model_dir, MODEL_FILE_NAME), global_step = i)
                print('accuracy less 0.98, exited!')
                break  # sys.exit(0)
            

        #关闭线程  
        coord.request_stop()  
        coord.join(threads)


def do_verify(model_dir):    # 'model/olivettifaces'
    one_set = Img2TFRecord('data_source/olivettifaces', 'tf_record/olivettifaces', 'gif')

    batch_size = 5
    verify_image_set, verify_label_set = one_set.read_test_images_from_tf_records(batch_size, [57, 47, 1], 40)  # height, width, channel

    calculate_label = do_recognition(verify_image_set, 40, 1)
    correct = tf.equal( calculate_label, tf.argmax(verify_label_set, 1) )
    accuracy = tf.reduce_mean( tf.cast(correct, 'float') )

    # 用于保存训练结果的对象
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 恢复数据并校验和测试
        saver.restore(sess, os.path.join(model_dir, MODEL_FILE_NAME))

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        print('valid set accuracy: ', accuracy.eval())


        '''
        test_pred = tf.argmax(neural_out, 1).eval({X: test_set_x})
        test_true = np.argmax(test_set_y, 1)
        test_correct = correct.eval({X: test_set_x, Y: test_set_y})
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i'
                %(test_true[i], test_pred[i]))
        plot_errordata(incorrect_index, "olivettifaces.gif")
        '''

        #关闭线程  
        coord.request_stop()  
        coord.join(threads)


# 返回计算出的 label(非 one hot 格式)
def do_recognition_one_image(model_dir, image_file_path, image_height, image_width, channel, class_size):
    # 加载图像并将数据转换为 tf 向量
    img = Image.open(image_file_path)
    img_ndarray = np.asarray(img, dtype='float32') / 256
    print('img_ndarray: ', img_ndarray)

    batch_one_image = np.empty((1, image_height, image_width, channel))
    batch_one_image[0] = img_ndarray.reshape(image_height, image_width, channel)
    print('batch_one_image: ', batch_one_image)

    tensor_one_image = tf.convert_to_tensor(batch_one_image)
    tensor_one_image = tf.cast(tensor_one_image, tf.float32)

    # 进行卷积运算
    neural_out = convolutional_neural_network(tensor_one_image, class_size, channel)

    # 计算各个类别的概率
    probability = tf.nn.softmax(neural_out)

    # 获得预测类别
    class_index = tf.argmax(neural_out, 1)


    test_softmax = tf.nn.softmax([[0., -0.5, 0.5, 2, 1, 0.2, 0.2, -0.1]])

    # 用于保存训练结果的对象
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 恢复数据并校验和测试
        saver.restore(sess, os.path.join(model_dir, MODEL_FILE_NAME))
        print(sess.run([neural_out, probability, class_index]))
        print(sess.run(test_softmax))




if __name__ == '__main__':
    # do_train('model/olivettifaces')
    # do_verify('model/olivettifaces')
    do_recognition_one_image('model/olivettifaces', 'data_source/olivettifaces/03/2.gif', 57, 47, 1, 40)