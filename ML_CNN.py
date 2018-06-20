######################## 机器学习 CNN 类 ##############################
# Author: 杨玉奇
# email: yangyuqi@sina.com
# url: https://github.com/jerryyyq/tf_algorithm_example
# copyright yangyuqi
# 著作权归作者 杨玉奇 所有。商业转载请联系作者获得授权，非商业转载请注明出处。
# date: 2018-03-28
###################################################################

import tensorflow as tf

from common import *
from ML_Model import ML_Model
from img_to_tf_record import Img2TFRecord


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
@:param lable_size     总共有多少个标签 
@:param channel        图像的通道数是多少
@:return Tensor        做过卷积后的图像 
''' 
def convolutional_neural_network(data, lable_size, channel):
    '''
    使用tf创建3层cnn，5 * 5 的 filter，输入为灰度，所以：

    第一层的 channel 是 1，图像宽高为 57，输出 32 个 filter，maxpooling 是缩放一倍
    第二层的输入为 32 个channel，宽高是 32，输出为 64 个 filter，maxpooling 是缩放一倍
    第三层的输入为 64 个channel，宽高是 16，输出为 64 个 filter，maxpooling 是缩放一倍

    所以最后输入的图像是 8 * 8 * 64，卷积层和全连接层都设置了 dropout 参数

    将输入的 8 * 8 * 64 的多维度，进行 flatten，映射到512个数据上，然后进行 softmax，输出到 onehot 类别上，类别的输入根据采集的人员的个数来确定。
    '''  

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch_size, 29, 24, 32]
    layer1_output = convolutional_layer(
        layer_index=1, 
        data=data,
        kernel_size=[5, 5, channel, 32],
        bias_size=[32],
        pooling_size=[1, 2, 2, 1]   # 用 2x2 模板做池化
    )

    # 经过第二层卷积神经网络后，得到的张量shape为：[batch_size, 15, 12, 64]
    layer2_output = convolutional_layer(
        layer_index=2,
        data=layer1_output,
        kernel_size=[5, 5, 32, 64],
        bias_size=[64],
        pooling_size=[1, 2, 2, 1]
    )

    # 全连接层。将卷积层张量数据拉成 2-D 张量只有一列的列向量
    layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
    layer3_output = tf.nn.relu(
        linear_layer(
            linear_index=1,
            data=layer2_output_flatten,
            weights_size=[15 * 12 * 64, 1024],
            biases_size=[1024]
        )
    )
    # layer3_output = tf.nn.dropout(layer3_output, 0.8)
    
    # 输出层
    output = linear_layer(
        data=layer3_output,
        weights_size=[1024, lable_size],      # 根据类别个数定义最后输出层的神经元
        biases_size=[lable_size]
    )

    return output


def loss(feature, label):
    softmax_out = tf.nn.softmax_cross_entropy_with_logits(logits=feature, labels=label)
    return tf.reduce_mean(softmax_out)


def do_train():
    one_set = Img2TFRecord('data_source/olivettifaces', 'tf_record/olivettifaces', 'gif')

    batch_size = 5
    image_batch, label_batch = one_Set.read_train_images_from_tf_records([57, 47, 1], batch_size)  # height, width, channel

    neural_out = convolutional_neural_network(image_batch, 40, 1)
    loss_out = loss(neural_out, label_batch)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss_out)

    # 用于保存训练结果的对象
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()   # tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        for i in range(3):
            img, lab = sess.run([image_batch, label_batch])
            print(img, lab)





        # 若不存在模型数据，需要训练模型参数
        if not os.path.exists(model_path + ".index"):
            # session.run(tf.global_variables_initializer())
            session.run(tf.initialize_all_variables() )
            
            best_loss = float('Inf')
            for epoch in range(20):
                epoch_loss = 0
                for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                    x = train_set_x[i * batch_size: (i + 1) * batch_size]
                    y = train_set_y[i * batch_size: (i + 1) * batch_size]
                    _, cost = session.run([train_step, loss_out], feed_dict={X: x, Y: y})
                    epoch_loss += cost

                print(epoch, ' : ', epoch_loss)
                if best_loss > epoch_loss:
                    best_loss = epoch_loss
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                        print("create the directory: %s" % model_dir)
                    save_path = saver.save(session, model_path)
                    print("Model saved in file: %s" % save_path)

        # 恢复数据并校验和测试
        saver.restore(session, model_path)
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))

        test_pred = tf.argmax(predict, 1).eval({X: test_set_x})
        test_true = np.argmax(test_set_y, 1)
        test_correct = correct.eval({X: test_set_x, Y: test_set_y})
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i'
                %(test_true[i], test_pred[i]))
        plot_errordata(incorrect_index, "olivettifaces.gif")
























        #关闭线程  
        coord.request_stop()  
        coord.join(threads)

