#coding=utf-8
# 参考文章：https://zhuanlan.zhihu.com/p/28196873
# 参考代码：https://github.com/hzy46/Char-RNN-TensorFlow

import tensorflow as tf
import numpy as np

batch_size = 2
input_size = 4

# ------------------------- RNN ----------------------------
'''
将一个 batch 送入模型计算，设输入数据的形状为(batch_size, input_size)，
那么计算时得到的隐层状态就是(batch_size, state_size)，
输出就是(batch_size, cell.output_size)。默认 cell.output_size == 神经元个数：num_units
代码：（代码中没有输出，只有隐层状态）
''' 

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=6) # state_size = 6
print(cell.state_size) # 6

inputs = tf.placeholder(np.float32, shape=(batch_size, input_size))   # (2, 4)
# inputs = np.empty( (batch_size, input_size), np.float32 )

h0 = cell.zero_state(batch_size, np.float32) # 通过 zero_state 得到一个全 0 的初始状态，形状为 (batch_size, state_size)
print( h0.get_shape() )      # (2, 6)

output, h1 = cell.__call__(inputs, h0) #调用call函数
# output, h1 = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0)

print( output.get_shape() )  # (2, 6)
print( h1.get_shape() )      # (2, 6)


# ------------------------- LSTM ----------------------------

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( num_units=7 )
print(lstm_cell.state_size)  # 14

linputs = tf.placeholder(np.float32, shape=(batch_size, input_size))   # (2, 4)
lh0 = lstm_cell.zero_state(batch_size, np.float32) # 通过zero_state得到一个全0的初始状态
print(lh0)      # shape=(2, 14)

loutput, lh1 = lstm_cell.__call__(linputs, lh0)
 
print(loutput)  # shape=(2, 7)
print(lh1)      # shape=(2, 14)


# ------------------------- dynamic_rnn time_major=False ----------------------------
'''
基础的RNNCell有一个很明显的问题：对于单个的 RNNCell，我们使用它的 __call__ 函数进行运算时，只是在序列时间上前进了一步。
比如使用 x1、h0 得到 h1，通过 x2、h1 得到 h2 等。这样的h话，如果我们的序列长度为 10，就要调用 10 次 __call__ 函数，比较麻烦。
对此，TensorFlow 提供了一个 tf.nn.dynamic_rnn 函数，使用该函数就相当于调用了 n 次 __call__ 函数。即通过 {h0, x1, x2, …, xn} 直接得到 {h1, h2…, hn}。

具体来说，设我们输入数据的格式为(batch_size, time_steps, input_size)，其中 time_steps 表示序列本身的长度，
如在Char RNN中，长度为 10 的句子对应的 time_steps 就等于 10。最后的 input_size 就表示输入数据单个序列单个时间维度上固有的长度。
'''

time_steps = 5
dlstm_cell = tf.nn.rnn_cell.BasicLSTMCell( num_units=7 )
print(dlstm_cell.state_size)  # 14

dinputs = tf.placeholder(np.float32, shape=(batch_size, time_steps, input_size))  # shape=(2, 5, 4)
dh0 = dlstm_cell.zero_state(batch_size, np.float32)
print(dh0)      # shape=(2, 14)

doutputs, fstate = tf.nn.dynamic_rnn(dlstm_cell, dinputs, initial_state=dh0)
print(doutputs)  # shape=(2, 5, 7)
print(fstate)    # shape=(2, 14)

# ------------------------- dynamic_rnn state_is_tuple=True, time_major=True ----------------------------
dlstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell( num_units=7, state_is_tuple=True )
print(dlstm_cell1.state_size)  # LSTMStateTuple(c=7, h=7)

dinputs1 = tf.placeholder(np.float32, shape=(time_steps, batch_size, input_size))  # shape=(5, 2, 4)
dh01 = dlstm_cell1.zero_state(batch_size, np.float32)
print(dh01)       # LSTMStateTuple(c=<tf.Tensor 'zeros:0' shape=(2, 7) dtype=float32>, h=<tf.Tensor 'zeros_1:0' shape=(2, 7) dtype=float32>)

doutputs1, fstate1 = tf.nn.dynamic_rnn(dlstm_cell1, dinputs1, initial_state=dh01, time_major=True)
print(doutputs1)  # shape=(5, 2, 7)
print(fstate1)    # LSTMStateTuple(c=<tf.Tensor 'RNN/while/Exit_2:0' shape=(2, 7) dtype=float32>, h=<tf.Tensor 'RNN/while/Exit_3:0' shape=(2, 7) dtype=float32>)


# ------------------------- MultiRNNCell ----------------------------
'''
很多时候，单层 RNN 的能力有限，我们需要多层的 RNN。将 x 输入第一层 RNN 的后得到隐层状态 h，这个隐层状态就相当于第二层 RNN 的输入，
第二层 RNN 的隐层状态又相当于第三层 RNN 的输入，以此类推。在 TensorFlow 中，可以使用 tf.nn.rnn_cell.MultiRNNCell 函数对 RNNCell 进行堆叠。
通过 MultiRNNCell 得到的 cell实际也是 RNNCell 的子类，因此也有 __call__ 方法、state_size 和 output_size 属性。
同样可以通过 tf.nn.dynamic_rnn 来一次运行多步。
'''

# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN，也可以写成：cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell()] * 3)

# 得到的 cell 实际也是 RNNCell 的子类
# 它的 state_size 是 384, 即：3 个 128
# 表示共有 3 个隐层状态，每个隐层状态的大小为 128
print(cell.state_size) # 384

# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32)                 # 通过 zero_state 得到一个全 0 的初始状态
output, h1 = cell.__call__(inputs, h0)
print(h1)                                            # shape=(32, 384)

