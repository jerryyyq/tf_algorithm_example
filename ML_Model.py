######################## 机器学习基类 ##############################
# Author: 杨玉奇
# email: yangyuqi@sina.com
# url: https://github.com/jerryyyq/tf_algorithm_example
# copyright yangyuqi
# 著作权归作者 杨玉奇 所有。商业转载请联系作者获得授权，非商业转载请注明出处。
# date: 2017-09-12
###################################################################

import tensorflow as tf
import os

class ML_Model:
    ####################### 构造与析构函数 #######################
    def __init__(self):
        self.__sess = tf.Session()
        self.__x = tf.Variable(0., name = "x")
    
    
    def __del__(self):
        self.__sess.close()
    
    
    ####################### 启动执行函数 #######################
    # training_steps    实际训练迭代次数 
    def do_train(self, training_steps = 1000):
        print( 'do_train: start' )
        
        self.__sess.run( tf.initialize_all_variables() )
    
        features, label = self.inputs()
    
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners( sess = self.__sess, coord = coord ) 

        try:
            total_loss = self.loss( features, label ) 
            train_op = self.train( total_loss ) 

            for step in range(training_steps):    # 实际训练闭环 
                self.__sess.run(train_op)

                # 查看训练过程损失递减
                if step % 10 == 0:
                    # print( str(step) + " loss: ", sess.run([total_loss]) )
                    self._echo_tensor( total_loss, 'step_' + str(step) + ' loss: ' )

            #print( str(training_steps) + " final loss: ", sess.run([total_loss]) )
            self._echo_tensor( total_loss, 'step_' + str(step) + ' final loss: ' )

            # 模型评估
            self.evaluate( features, label ) 
    
        except tf.errors.OutOfRangeError:
            print( 'Done training -- epoch limit reached' )

        finally:
            coord.request_stop()        
            coord.join( threads )
            
        print( 'do_train: finish' )
        
    
    ####################### 主流程函数 #######################
    # 计算返回推断模型输出
    def inference(self, features):
        return features + self.__x
    
    # 计算损失(训练数据 features 及 label) 
    def loss(self, features, label):
        label_predicted = self.inference( features )
        return label - label_predicted
    
    # 读取或生成训练数据
    def inputs(self): 
        features = tf.constant(0.)
        label = tf.constant(0.)
        return features, label

    # 训练
    def train(self, loss):
        # loss also is cost
        learning_rate = 0.0000001
        # return tf.train.Optimizer(False, name = 'sample').minimize( loss )  # 会产生： NotImplementedError 异常
        return tf.train.GradientDescentOptimizer( learning_rate ).minimize( loss )

    # 完成学习后，进行效果评估
    def evaluate(self, features, label):
        print( self.__sess.run(self.inference(features)) )
    
    
    
    ####################### 辅助函数 #######################
    def _echo_tensor(self, tensor, prefix = ''):
        # 注意： print() 显示时会把元素之间的逗号去掉
        if( isinstance(tensor, tf.Tensor) ):
            print( '{0} tensor.shape = {1}, tensor = {2}\r\n'.format(prefix, self.__sess.run(tf.shape(tensor)), self.__sess.run(tensor)) )
            # print( '{0} tensor = {1}\r\n'.format(prefix, self.__sess.run(tensor)) )
        else:
            print( '{0} not_tensor = {1}\r\n'.format(prefix, tensor) )
            
            
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
    def _read_csv(self, batch_size, file_name, record_defaults):
        file_path = list( map(lambda name: os.path.join(os.getcwd(), name), file_name) )
        print('file_path = ', file_path, '\r\n')
        file_queue = tf.train.string_input_producer(file_path) 

        reader = tf.TextLineReader(skip_header_lines = 1)

        key, value = reader.read(file_queue)
        print(key, value)

        decoded = tf.decode_csv(value, record_defaults, ',')
        return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size) # 读取文件，加载张量batch_size行
        
        
        
if __name__ == '__main__':
    one_ml = ML_Model()
    one_ml.do_train( 10 )

