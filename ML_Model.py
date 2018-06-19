######################## 机器学习基类 ##############################
# Author: 杨玉奇
# email: yangyuqi@sina.com
# url: https://github.com/jerryyyq/tf_algorithm_example
# copyright yangyuqi
# 著作权归作者 杨玉奇 所有。商业转载请联系作者获得授权，非商业转载请注明出处。
# date: 2017-09-12
###################################################################

import os
import tensorflow as tf

from common import *

class ML_Model:
    ####################### 构造与析构函数 #######################
    def __init__(self):
        self._sess = None
        
        self.__x = tf.Variable(0., name = "x")
        self.__save_path = os.path.join( '/tmp', self.__class__.__name__ + '.vari' )
    
    def __del__(self):
        if(self._sess):
            self._sess.close()
    
    
    ####################### 学习与评价启动执行函数 #######################
    # training_steps    实际训练迭代次数
    # file_name: ['train_1.csv', 'train_2.csv']
    def do_train(self, training_steps = 1000, train_file_name = [], train_batch_size = 10):
        print( '-------------- do_train: start -----------------' )
        self._sess = tf.Session()
        self._sess.run( tf.initialize_all_variables() )
    
        feature_batch, label_batch = self.inputs( train_file_name, train_batch_size )
        train_op = self.train( feature_batch, label_batch )
        
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners( sess = self._sess, coord = coord ) 

        try:
            # while not coord.should_stop():
            for step in range(training_steps):    # 实际训练闭环 
                if coord.should_stop():
                    print('---- coord should stop ----')
                    break
                self._sess.run(train_op)

                # 查看训练过程损失递减
                if step % 10 == 0:
                    echo_tensor( self._sess, feature_batch, 'features_' + str(step) )
                    echo_tensor( self._sess, label_batch, 'label_' + str(step) )
                    echo_tensor( self._sess, self._total_loss, 'step_' + str(step) + ' loss: ' )

            #print( str(training_steps) + " final loss: ", sess.run([total_loss]) )
            echo_tensor( self._sess, self._total_loss, 'training end. step_' + str(step) + ' final loss: ' )

            saver = tf.train.Saver()
            save_path = saver.save( self._sess, self.__save_path )
            print('save_path is: ', save_path)

            # 模型评估
            evaluate_result = self.evaluate( feature_batch, label_batch ) 
            echo_tensor( self._sess, evaluate_result, 'evaluate_result' )
    
        except tf.errors.OutOfRangeError:
            print( 'Done training -- epoch limit reached' )

        finally:
            coord.request_stop()        
            coord.join( threads )
            
        self._sess.close()
  
        print( '----------------- do_train: finish -----------------' )
 

    # file_name: ['test_1.csv', 'test_2.csv']
    def do_evaluate(self, file_name = [], batch_size = 10):
        print( '-------------- do_evaluate: start -----------------\n' )
        self._sess = tf.Session()
        saver = tf.train.Saver()
        
        ckpt = tf.train.get_checkpoint_state( '/tmp' )
        if ckpt and ckpt.model_checkpoint_path:
            # saver.restore( self._sess, ckpt.model_checkpoint_path )
            saver.restore( self._sess, self.__save_path )
        else:
            print('not find ckpt file!!!!!')      
        
        test_features, test_label = self.inputs( file_name, batch_size )
        
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners( sess = self._sess, coord = coord ) 

        self._sess.run([test_features, test_label])
        
        accuracy_rate = self.evaluate( test_features, test_label )
        echo_tensor( self._sess, accuracy_rate, 'accuracy_rate' )
        
        coord.request_stop()        
        coord.join( threads )
        
        self._sess.close()
        print( '----------------- do_evaluate: finish -----------------\n' )
    
    
    ####################### 主流程函数 #######################
    # 计算返回推断模型输出
    def inference(self, features):
        return features + self.__x
    
    # 计算损失(训练数据 features 及 label) 
    def loss(self, features, label):
        label_predicted = self.inference( features )
        return label - label_predicted
    
    # 读取或生成训练数据
    # file_name: ['1.csv', '2.csv']
    def inputs(self, file_name = [], batch_size = 10): 
        features = tf.constant(0.)
        label = tf.constant(0.)
        return features, label

    # 训练
    def train(self, feature_batch, label_batch):
        self._total_loss = self.loss( feature_batch, label_batch ) 
        # loss also is cost
        learning_rate = 0.0000001
        # return tf.train.Optimizer(False, name = 'sample').minimize( loss )  # 会产生： NotImplementedError 异常
        return tf.train.GradientDescentOptimizer( learning_rate ).minimize( self._total_loss )

    # 完成学习后，进行效果评估
    def evaluate(self, test_features, test_label):
        echo_tensor( self._sess, self.__x, 'At evaluate, the __x' )

        label_predicted = tf.to_float( self.inference(test_features) )
        
        different = (label_predicted - test_label) / test_label
        
        echo_tensor( self._sess, test_label, 'test_label' )
        echo_tensor( self._sess, label_predicted, 'label_predicted' )
 
        return tf.reduce_mean(different)

      
        
        
if __name__ == '__main__':
    one_ml = ML_Model()
    one_ml.do_train( 10 )

