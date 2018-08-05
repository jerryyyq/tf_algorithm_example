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
# tensorboard --logdir=graph_dir，如果没有设置，本类的默认为：'graph/DL_CNN'
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
    @:param continue_train         是否继续上一次训练。如果是，会进行 saver.restore
    @:param learning_rate          学习率
    @:param test_sample_number     用于验证的样本的总数。应用于使用测试准确率判断是否训练完成的情况

    @:return none
    ''' 
    def train(self, train_wheels, train_sample_number, batch_size = 5, continue_train = False, learning_rate = 0.01, test_sample_number = -1):
        print('开始训练。最大轮次：{}, 样本总数：{}, 每批样本数：{}' . format(train_wheels, train_sample_number, batch_size) )
        one_set = Img2TFRecord('', self._tf_record_dir)

        reshape = [self._image_height, self._image_width, self._image_channel]
        # 获取训练集数据
        image_set, label_set = one_set.read_train_images_from_tf_records(batch_size, reshape, self._class_size)  # height, width, channel
        # 调用模型
        neural_out = self._convolutional_neural_network(image_set, self._class_size, self._image_channel)
        loss_out = self._loss(neural_out, label_set)
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(neural_out, 1), tf.argmax(label_set, 1)), tf.float32))

        # tf.train.GradientDescentOptimizer(0.01).minimize(loss_out)    # 梯度下降算法不需要下面的 with
        with tf.variable_scope(name_or_scope = '', reuse = tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer( learning_rate ).minimize(loss_out)

        # 将 loss 与 optimizer 保存以供 tensorboard 使用
        if tf.__version__ < '1':
            tf.scalar_summary('loss', loss_out)
            tf.scalar_summary('train_accuracy', train_accuracy)
            # tf.scalar_summary('optimizer', optimizer)
        else:
            tf.summary.scalar('loss', loss_out)
            tf.summary.scalar('train_accuracy', train_accuracy)
            # tf.summary.scalar('optimizer', optimizer)

        # 方法二：按准确率来比较。注意：计算准确率时使用的是：测试集！测试集！测试集！
        verify_image_set, verify_label_set = one_set.read_test_images_from_tf_records(batch_size, reshape, self._class_size)

        # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
        # tf.get_variable_scope().reuse_variables()  # 为了方法二对准确率进行计算，复用当前变量
        predict_labell = self._recognition(verify_image_set, self._class_size, self._image_channel)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_labell, tf.argmax(verify_label_set, 1)), tf.float32))
        # 将 accuracy 保存以供 tensorboard 使用
        if tf.__version__ < '1':
            tf.scalar_summary('accuracy', accuracy)
        else: 
            tf.summary.scalar('accuracy', accuracy)
        
        # -------------------------------------------------------------------------
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
            if continue_train:
                saver.restore(sess, self._model_file)

            if tf.__version__ < '1':
                summary_writer = tf.train.SummaryWriter(self._graph_dir, sess.graph)
            else:
                summary_writer = tf.summary.FileWriter(self._graph_dir, graph = tf.get_default_graph())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners( sess = sess, coord = coord )

            best_loss = float('Inf')
            for i in range(train_wheels):
                epoch_loss = 0.
                total_accuracy = 0.
                for j in range( train_sample_number // batch_size ):
                    _, cost, taccuracy, summary = sess.run([optimizer, loss_out, train_accuracy, merged_summary_op])
                    summary_writer.add_summary(summary, i * j)

                    # print('j = {}, cost = {}, train_accuracy = {}' . format(j, cost, taccuracy))
                    epoch_loss += cost
                    total_accuracy = total_accuracy + taccuracy

                '''
                # 方法一，每一轮比较一次
                # 共有学习样本 320 个，所以每一轮应该是 320 / batch_size 次
                print(i, ', epoch_loss is : ', epoch_loss, ', train_accuracy is : ', total_accuracy / j)
                if best_loss > epoch_loss:
                    best_loss = epoch_loss

                    save_path = saver.save( sess, self._model_file )
                    print( "Model saved in file: {}, epoch_loss = {}" . format(save_path, epoch_loss) )
                    if 0.0 == epoch_loss:
                        print('epoch_loss == 0.0, exited!')
                        break
                '''

                #'''                
                # 方法二，按准确率结束学习
                # 获取测试数据的准确率
                if 0 > test_sample_number:
                    acc = accuracy.eval()
                else:
                    acc = 0
                    for k in range(test_sample_number // batch_size):
                        acc = acc + accuracy.eval()

                    acc = acc / k
                
                print(i, ', epoch_loss is : ', epoch_loss, ', train_accuracy is : ', total_accuracy / j, ', verify accuracy is : ', acc)
                # 准确率大于0.98时保存并退出
                if i > 2:
                    if best_loss > epoch_loss:
                        best_loss = epoch_loss
                        save_path = saver.save( sess, self._model_file, global_step = 0 )
                        print( "Model saved in file: {}, epoch_loss = {}, global_step = 0" . format(save_path, epoch_loss) )           
                
                    if total_accuracy / j > 0.97:
                        save_path = saver.save( sess, self._model_file, global_step = i )
                        print( "train_accuracy > 0.97, Model saved in file: {}, epoch_loss = {}, global_step = {}" . format(save_path, epoch_loss, i) )                    
                    
                        if acc > 0.98:
                            saver.save(sess, self._model_file)
                            print('verify accuracy > 0.98, finish!')
                            break  # sys.exit(0)
                #'''

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

        predict_labell = self._recognition(verify_image_set, self._class_size, self._image_channel)
        # predict_labell = tf.Print(predict_labell, [predict_labell], 'predict_labell = ', summarize=100)

        correct = tf.equal( predict_labell, tf.argmax(verify_label_set, 1) )
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
    识别一批图像数据，计算属于各个类别的概率 
    @:param image_set           tensor 对象，待识别图像集合，图像 shape 需要与训练时的相同
    @:param out_probability     返回的是各类的概率值还是直接返回类别，默认返回的是各类的概率值
    @:return                    如果 out_probability 为 True, 那么返回计算出的各类别的概率值( one hot 格式)
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

    ''''' 
    识别一个图像文件，计算属于各个类别的概率 
    @:param image_file_path     待识别图像文件
    @:param out_probability     返回的是各类的概率值还是直接返回类别，默认返回的是各类的概率值
    @:return                    如果 out_probability 为 True, 那么返回计算出的各类别的概率值( one hot 格式)
                                如果 out_probability 为 False，那么返回识别出的类别（非 one hot 格式）
    ''' 
    def recognition_one_image(self, image_file_path, out_probability = True):
        import numpy as np
        from PIL import Image
        img = Image.open(image_file_path)
        img = img.convert('L')

        batch_one_image = np.empty((1, self._image_height, self._image_width))
        batch_one_image[0] = np.asarray(img)

        tensor_one_image = tf.reshape( batch_one_image, (1, self._image_height, self._image_width, self._image_channel) )
        tensor_one_image = tf.cast(tensor_one_image, tf.float32) * (1./255)
        return self.recognition(tensor_one_image, out_probability)


    @staticmethod
    def _convolutional_layer(layer_index, data, kernel_size, bias_size, pooling_size):
        # regularizer = tf.contrib.layers.l1_regularizer(0.1)
        # with tf.variable_scope(name_or_scope = 'conv_layer' + str(layer_index), reuse = tf.AUTO_REUSE, regularizer = regularizer):
        with tf.variable_scope(name_or_scope = 'conv_layer' + str(layer_index), reuse = tf.AUTO_REUSE):
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
        with tf.variable_scope(name_or_scope = 'linear_layer' + str(linear_index), reuse = tf.AUTO_REUSE):
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
        layer_last_output = DL_CNN._convolutional_layer(    # layer2_output = 
            layer_index = 2,
            data = layer1_output,
            kernel_size = [5, 5, 32, 64],
            bias_size = [64],
            pooling_size = [1, 2, 2, 1]
        )

        '''
        # 再加一层试试，得到的张量shape为：[batch_size, , , 128]
        layer_last_output = DL_CNN._convolutional_layer(
            layer_index = 3,
            data = layer2_output,
            kernel_size = [5, 5, 64, 128],
            bias_size = [128],
            pooling_size = [1, 2, 2, 1]
        )
        '''

        # 全连接层。将卷积层张量数据拉成 2-D 张量只有一列的列向量
        all_link_n_number = 256    # 1024
        layer_last_output_flatten = tf.contrib.layers.flatten(layer_last_output)
        layer_all_link = tf.nn.relu(
            DL_CNN._linear_layer(
                linear_index = 1,
                data = layer_last_output_flatten,
                weights_size = [layer_last_output.shape[1] * layer_last_output.shape[2] * layer_last_output.shape[3], all_link_n_number],    # layer2_output, weights_size = [15 * 12 * 64, 1024]
                biases_size = [all_link_n_number]
            )
        )
        
        # 减少过拟合，随机让某些权重不更新
        # layer_all_link = tf.nn.dropout(layer_all_link, 0.8)
        
        # 输出层
        output = DL_CNN._linear_layer(
            linear_index = 2,
            data = layer_all_link,
            weights_size = [all_link_n_number, class_size],      # 根据类别个数定义最后输出层的神经元
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
        # return tf.reduce_mean(softmax_out) + 0.01 * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

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

    img = Image.open('data_source/olivettifaces/20/2.gif')
    img = img.convert('L')
    img_ndarray = np.asarray(img, dtype='float32') / 255

    batch_one_image = np.empty((1, image_height, image_width, image_channel))
    batch_one_image[0] = img_ndarray.reshape(image_height, image_width, image_channel)

    tensor_one_image = tf.convert_to_tensor(batch_one_image)
    tensor_one_image = tf.cast(tensor_one_image, tf.float32)

    one_cnn.recognition(tensor_one_image, False)
    
