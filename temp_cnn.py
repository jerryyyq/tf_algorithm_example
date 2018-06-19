import numpy as np

import sklearn.metrics as metrics
import tensorflow as tf
from PIL import Image
from tensorflow.contrib import learn
from tensorflow.contrib.learn import SKCompat
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.ops import init_ops

IMAGE_SIZE = 28
LOG_DIR = './ops_logs'

mnist = learn.datasets.load_dataset('mnist')

def inference(x, num_class):
  with tf.variable_scope('softmax'):
    dtype = x.dtype.base_dtype
    init_mean = 0.0
    init_stddev = 0.0
    weight = tf.get_variable('weights',
                                [x.get_shape()[1], num_class], initializer=init_ops.random_normal_initializer(init_mean, init_stddev, dtype=dtype), dtype=dtype)
    biases = tf.get_variable('bias', [num_class], initializer=init_ops.random_normal_initializer(init_mean, init_stddev, dtype=dtype), dtype=dtype)

    logits = tf.nn.xw_plus_b(x, weight, biases)
    return logits

def model(features, labels, mode):
    if mode != model_fn_lib.ModeKeys.INFER:
        labels = tf.one_hot(labels, 10, 1, 0)
    else:
        labels = None

    inputs = tf.reshape(features, (-1, IMAGE_SIZE, IMAGE_SIZE, 1))

    #conv1
    conv1 = tf.contrib.layers.conv2d(inputs, 4, [5, 5], scope='conv_layer1', activation_fn=tf.nn.tanh);
    pool1 = tf.contrib.layers.max_pool2d(conv1, [2, 2], padding='SAME')
    #conv2
    conv2 = tf.contrib.layers.conv2d(pool1, 6, [5, 5], scope='conv_layer2', activation_fn=tf.nn.tanh);
    pool2 = tf.contrib.layers.max_pool2d(conv2, [2, 2], padding='SAME')
    pool2_shape = pool2.get_shape()
    pool2_in_flat = tf.reshape(pool2, [pool2_shape[0].value or -1, np.prod(pool2_shape[1:]).value])
    #fc
    fc1 = tf.contrib.layers.fully_connected(pool2_in_flat, 1024, scope='fc_layer1', activation_fn=tf.nn.tanh)
    #dropout
    is_training = False
    if mode == model_fn_lib.ModeKeys.TRAIN: 
        is_training = True

    dropout = tf.contrib.layers.dropout(fc1, keep_prob=0.5, is_training=is_training, scope='dropout')

    logits = inference(dropout, 10)
    prediction = tf.nn.softmax(logits)
    if mode != model_fn_lib.ModeKeys.INFER:
        loss = tf.contrib.losses.softmax_cross_entropy(logits, labels)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
            learning_rate=0.1)
    else:
        train_op = None
        loss = None

    return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


classifier = SKCompat(learn.Estimator(model_fn=model, model_dir=LOG_DIR))

classifier.fit(mnist.train.images, mnist.train.labels, steps=1000, batch_size=300)

predictions = classifier.predict(mnist.test.images)
score = metrics.accuracy_score(mnist.test.labels, predictions['class'])
print('Accuracy: {0:f}'.format(score))