######################## 机器学习 CNN 类 ##############################
# Author: 杨玉奇
# email: yangyuqi@sina.com
# url: https://github.com/jerryyyq/tf_algorithm_example
# copyright yangyuqi
# 著作权归作者 杨玉奇 所有。商业转载请联系作者获得授权，非商业转载请注明出处。
# date: 2018-06-22
###################################################################

import os
import tensorflow as tf

from img_to_tf_record import Img2TFRecord

# CNN 图像识别基类，派生类可以通过重写 _convolutional_neural_network 函数来构建自己的 CNN 模型
class DL_CNN:
    ####################### 构造与析构函数 #######################
    def __init__(self, tf_record_dir, image_height, image_width, image_channel, class_size, model_file = '', graph_dir = '' ):
        self._tf_record_dir = tf_record_dir
        self._image_height = image_height
        self._image_width = image_width
        self._image_channel = image_channel
        self._class_size = class_size

        if model_file:
            self._model_file = model_file
        else:
            self._model_file = os.path.join( 'model', self.__class__.__name__ + '/best.ckpt' )

        # 创建保存训练模型需要的目录
        model_dir = os.path.dirname( os.path.realpath(self._model_file) )
        if not os.path.exists( model_dir ):
            os.makedirs( model_dir )


        if graph_dir:
            self._graph_dir = graph_dir
        else:
            self._graph_dir = os.path.join( 'graph', self.__class__.__name__ )

        # 创建保存 graph 的目录
        if not os.path.exists( self._graph_dir ):
            os.makedirs( self._graph_dir )


   

    ''''' 
    训练 
    @:param train_wheels           计划对样本训练多少轮 
    @:param train_sample_number    用于训练的样本的总数 
    @:param batch_size             每一次取多少个样本进行训练 
    @:return none
    ''' 
    def train(self, train_wheels, train_sample_number, batch_size = 5):
        print(train_wheels, train_sample_number, batch_size)
        one_set = Img2TFRecord('', self._tf_record_dir)

        reshape = [self._image_height, self._image_width, self._image_channel]
        image_set, label_set = one_set.read_train_images_from_tf_records(batch_size, reshape, self._class_size)  # height, width, channel

        neural_out = self._convolutional_neural_network(image_set, self._class_size, self._image_channel)
        loss_out = self._loss(neural_out, label_set)
        optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss_out)     #tf.train.GradientDescentOptimizer(0.01).minimize(loss_out)
        # 将 loss 与 optimizer 保存以供 tensorboard 使用
        if tf.__version__ < '1':
            tf.scalar_summary('loss', loss_out)
            # tf.scalar_summary('optimizer', optimizer)
        else:
            tf.summary.scalar('loss', loss_out)
            # tf.summary.scalar('optimizer', optimizer)

        ''' 方法二
        verify_image_set, verify_label_set = one_set.read_test_images_from_tf_records(batch_size, reshape, self._class_size)

        # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
        tf.get_variable_scope().reuse_variables()  # 为了方法二对准确率进行计算，复用当前变量
        calculate_label = self._recognition(verify_image_set, self._class_size, self._image_channel)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(calculate_label, tf.argmax(verify_label_set, 1)), tf.float32))
        # 将 accuracy 保存以供 tensorboard 使用
        tf.scalar_summary('accuracy', accuracy)        # 高版本 -> tf.summary.scalar('accuracy', accuracy)
        '''
        if tf.__version__ < '1':
            merged_summary_op = tf.merge_all_summaries()
        else:
            merged_summary_op = tf.summary.merge_all()

        # 用于保存训练结果的对象
        saver = tf.train.Saver()

        if tf.__version__ < '1':
            init = tf.initialize_all_variables()
        else:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init)
            if tf.__version__ < '1':
                summary_writer = tf.train.SummaryWriter(self._graph_dir, sess.graph)
            else:
                summary_writer = tf.summary.FileWriter(self._graph_dir, graph = tf.get_default_graph())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners( sess = sess, coord = coord )

            best_loss = float('Inf')
            epoch_loss = 0

            for i in range(int(train_sample_number/batch_size) * train_wheels):
                _, cost, summary = sess.run([optimizer, loss_out, merged_summary_op])
                summary_writer.add_summary(summary, i)
                epoch_loss += cost

                #'''
                # 方法一，每一轮比较一次
                # 共有学习样本 320 个，所以每一轮应该是 320 / batch_size 次
                if i % (train_sample_number/batch_size) == 0:
                    print(i, ' : ', epoch_loss)
                    if best_loss > epoch_loss:
                        best_loss = epoch_loss

                        save_path = saver.save( sess, self._model_file )
                        print( "Model saved in file: {}, epoch_loss = {}" . format(save_path, epoch_loss) )
                        if 0.0 == epoch_loss:
                            print('epoch_loss == 0.0, exited!')
                            break

                    epoch_loss = 0
                #'''

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
                    saver.save(sess, self._model_file, global_step = i)
                    print('accuracy less 0.98, exited!')
                    break  # sys.exit(0)
                '''

            #关闭线程  
            coord.request_stop()
            coord.join(threads)


    ''''' 
    校验。打印值：1 为准确，0 为失败
    @:param test_sample_number     用于验证的样本的总数 
    @:param batch_size             每一次取多少个样本进行校验 
    @:return none
    ''' 
    def verify(self, test_sample_number, batch_size = 5):
        print('_tf_record_dir = ', self._tf_record_dir)
        one_set = Img2TFRecord('', self._tf_record_dir)

        reshape = [self._image_height, self._image_width, self._image_channel]
        verify_image_set, verify_label_set = one_set.read_test_images_from_tf_records(batch_size, reshape, self._class_size)
        # verify_label_set = tf.Print(verify_label_set, [verify_label_set], 'verify_label_set = ', summarize=100)

        calculate_label = self._recognition(verify_image_set, self._class_size, self._image_channel)
        # calculate_label = tf.Print(calculate_label, [calculate_label], 'calculate_label = ', summarize=100)

        correct = tf.equal( calculate_label, tf.argmax(verify_label_set, 1) )
        accuracy = tf.reduce_mean( tf.cast(correct, 'float') )

        # 用于保存训练结果的对象
        saver = tf.train.Saver()

        if tf.__version__ < '1':
            init = tf.initialize_all_variables()
        else:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init)              # 不加这行会导致 OutOfRangeError (see above for traceback): RandomShuffleQueue 的错误发生
            # 恢复数据并校验和测试
            saver.restore(sess, self._model_file)

            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners( sess = sess, coord = coord )

            for i in range( int(test_sample_number/batch_size) ):
                print( 'verify i = {}, valid set accuracy = {}'.format(i, accuracy.eval()) )



            #关闭线程  
            coord.request_stop()  
            coord.join(threads)


    ''''' 
    识别，计算属于各个类别的概率 
    @:param image_set          tensor 对象，待识别图像集合，图像 shape 需要与训练时的相同
    @:param out_probability    tensor 对象，待识别图像集合，图像 shape 需要与训练时的相同
    @:return                   tensor 对象，如果 out_probability 为 True, 那么返回计算出的各类别的概率值( one hot 格式)
                                           如果 out_probability 为 False，那么返回识别出的类别（非 one hot 格式）
    ''' 
    def recognition(self, image_set, out_probability = True):
        neural_out = self._convolutional_neural_network(image_set, self._class_size, self._image_channel)
        neural_out = tf.Print(neural_out, [neural_out], 'neural_out = ', summarize=100)        
        # 计算各个类别的概率
        probability = tf.nn.softmax(neural_out)

        # 计算类别
        class_index = tf.argmax(neural_out, 1)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 恢复数据并校验和测试
            saver.restore(sess, self._model_file)

            if out_probability:
                ret = sess.run( probability )
            else:
                ret = sess.run( class_index )
            
            print( ret )
            return ret



    @staticmethod
    def _convolutional_layer(layer_index, data, kernel_size, bias_size, pooling_size):
        kernel = tf.get_variable("conv_{}".format(layer_index), kernel_size, initializer=tf.random_normal_initializer())
        bias = tf.get_variable("bias_{}".format(layer_index), bias_size, initializer=tf.random_normal_initializer())

        # 卷积
        conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')

        # 线性修正，将所有的元素中的负数置为零
        linear_output = tf.nn.relu(tf.add(conv, bias))

        # 池化
        pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
        return pooling

    @staticmethod
    def _linear_layer(linear_index, data, weights_size, biases_size):
        weights = tf.get_variable("weigths_{}".format(linear_index), weights_size, initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases_{}".format(linear_index), biases_size, initializer=tf.random_normal_initializer())
        return tf.add(tf.matmul(data, weights), biases)


    ''''' 
    创建卷积层。派生类主要可以重写这个函数来实现自己的 DL 模型 
    @:param data           输入图像 
    @:param class_size     总共有多少个类别 
    @:param image_channel  图像的通道数是多少
    @:return Tensor        计算出的 label(one hot 格式)
    ''' 
    @staticmethod
    def _convolutional_neural_network(data, class_size, image_channel):
        # 经过第一层卷积神经网络后，得到的张量shape为：[batch_size, 29, 24, 32]
        layer1_output = DL_CNN._convolutional_layer(
            layer_index = 1, 
            data = data,
            kernel_size = [5, 5, image_channel, 32],
            bias_size = [32],
            pooling_size = [1, 2, 2, 1]   # 用 2x2 模板做池化
        )

        # 经过第二层卷积神经网络后，得到的张量shape为：[batch_size, 15, 12, 64]
        layer2_output = DL_CNN._convolutional_layer(
            layer_index = 2,
            data = layer1_output,
            kernel_size = [5, 5, 32, 64],
            bias_size = [64],
            pooling_size = [1, 2, 2, 1]
        )

        # 再加一层试试，得到的张量shape为：[batch_size, , , 128]
        layer_last_output = DL_CNN._convolutional_layer(
            layer_index = 3,
            data = layer2_output,
            kernel_size = [5, 5, 64, 128],
            bias_size = [128],
            pooling_size = [1, 2, 2, 1]
        )

        # 全连接层。将卷积层张量数据拉成 2-D 张量只有一列的列向量
        layer_last_output_flatten = tf.contrib.layers.flatten(layer_last_output)
        layer_all_link = tf.nn.relu(
            DL_CNN._linear_layer(
                linear_index = 1,
                data = layer_last_output_flatten,
                weights_size = [layer_last_output.shape[1] * layer_last_output.shape[2] * layer_last_output.shape[3], 1024],    # layer2_output, weights_size = [15 * 12 * 64, 1024]
                biases_size = [1024]
            )
        )
        # 减少过拟合，随机让某些权重不更新
        # layer3_output = tf.nn.dropout(layer3_output, 0.8)
        
        # 输出层
        output = DL_CNN._linear_layer(
            linear_index = 2,
            data = layer_all_link,
            weights_size = [1024, class_size],      # 根据类别个数定义最后输出层的神经元
            biases_size = [class_size]
        )

        return output

    @staticmethod
    def _loss(feature, label):
        if tf.__version__ < '1':
            softmax_out = tf.nn.softmax_cross_entropy_with_logits(logits = feature, labels = label)
        else:
            softmax_out = tf.nn.softmax_cross_entropy_with_logits_v2(logits = feature, labels = label)

        return tf.reduce_mean(softmax_out)

    # 返回计算出的 class(非 one hot 格式)
    @staticmethod
    def _recognition(image_set, class_size, image_channel):
        neural_out = DL_CNN._convolutional_neural_network(image_set, class_size, image_channel)
        return tf.argmax(neural_out, 1)



if __name__ == '__main__':
    image_height = 57
    image_width = 47
    image_channel = 1
    one_cnn = DL_CNN('tf_record/olivettifaces', image_height, image_width, image_channel, 40, 'model/olive/olive_best.ckpt')

    # 下面这三个场景不能同时运行，每次只能运行一个场景
    # 场景一：使用训练数据进行模型训练
    # one_cnn.train(500, 320)

    # 场景二：使用验证数据来校验训练好的模型的准确率
    # one_cnn.verify(80, 10)

    # 场景三：使用训练好的模型来识别一个图片的类型
    
    import numpy as np
    from PIL import Image

    img = Image.open('data_source/olivettifaces/00/2.gif')
    img_ndarray = np.asarray(img, dtype='float32') / 255

    batch_one_image = np.empty((1, image_height, image_width, image_channel))
    batch_one_image[0] = img_ndarray.reshape(image_height, image_width, image_channel)

    tensor_one_image = tf.convert_to_tensor(batch_one_image)
    tensor_one_image = tf.cast(tensor_one_image, tf.float32)

    one_cnn.recognition(tensor_one_image)
    
