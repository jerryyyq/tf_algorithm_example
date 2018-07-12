# 参考文章：https://www.leiphone.com/news/201704/IlnwSvF6pGOZoHZq.html
# from __future__ import print_function, division    

import numpy as np    
import tensorflow as tf    
import matplotlib.pyplot as plt    

num_epochs = 100    
total_series_length = 50000    
truncated_backprop_length = 15    
state_size = 4    
num_classes = 2    
echo_step = 3    
batch_size = 5    
num_batches = total_series_length//batch_size//truncated_backprop_length    # 666 次，每次 15 个数

'''
return shape(batch_size, total_series_length / batch_size) = (5, 10000):
x = [[1 1 0 ... 1 1 1]
 [0 0 1 ... 0 1 1]
 [0 1 0 ... 0 0 0]
 [1 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 0]] 
y = [[0 0 0 ... 0 1 1]
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 1 0 0]] 
'''
def generateData():    
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))    
    y = np.roll(x, echo_step)        # 后 3 个放到前面。      例如：x = [1,2,3,4,5,6,7,8,9,10,11,12], y = [10,11,12,1,2,3,4,5,6,7,8,9] 
    # y[0:echo_step] = 0    
    x = x.reshape((batch_size, -1))  # 按 batch_size 数分段。例如：batch_size = 2, 那么 x = [[1,2,3,4,5,6], [7,8,9,10,11,12]]; y = [[10,11,12,1,2,3], [4,5,6,7,8,9]]
    y = y.reshape((batch_size, -1))
    y[:, :echo_step] = 0             # 前 3 个元素置为 0。    例如： y = [[0,0,0,1,2,3], [0,0,0,7,8,9]]
    return (x, y)    


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])    # (5, 15)
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])   
'''
batchX =  [[1 1 0 0 0 0 1 1 1 1 0 1 1 0 0]
 [0 0 1 0 1 0 1 1 1 1 1 0 1 1 1]
 [0 1 0 1 1 1 0 1 1 0 0 1 1 1 1]
 [1 0 0 1 1 1 0 1 0 1 1 1 0 0 0]
 [0 0 0 0 1 1 0 0 0 1 1 1 0 0 0]] 
batchY =  [[0 0 0 1 1 0 0 0 0 1 1 1 1 0 1]
 [0 0 0 0 0 1 0 1 0 1 1 1 1 1 0]
 [0 0 0 0 1 0 1 1 1 0 1 1 0 0 1]
 [0 0 0 1 0 0 1 1 1 0 1 0 1 1 1]
 [0 0 0 0 0 0 0 1 1 0 0 0 1 1 1]]
''' 

init_state = tf.placeholder(tf.float32, [batch_size, state_size])   # (5, 4)  初值 = [[0. 0. 0. 0.] [0. 0. 0. 0.] [0. 0. 0. 0.] [0. 0. 0. 0.] [0. 0. 0. 0.]]


W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)     # (5, 4)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)                      # (1, 4)


W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)        # (4, 2)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)                    # (1, 2)


# 把每个数拆成一个元素。Unpack columns
# inputs_series = tf.unpack(batchX_placeholder, axis=1)    # 按列解包   (15, 5) 。tf.unpack 在 1.x 已被 tf.unstack 替代
# labels_series = tf.unpack(batchY_placeholder, axis=1)
# inputs_series = tf.transpose(batchX_placeholder, perm=[1, 0])
# labels_series = tf.transpose(batchY_placeholder, perm=[1, 0])
# inputs_series = np.transpose(batchX_placeholder)
# labels_series = np.transpose(batchY_placeholder)
inputs_series = tf.unstack(batchX_placeholder, axis=1)         # [(5, 1), .... 共 15 个]  把每个数拆成一个元素
labels_series = tf.unstack(batchY_placeholder, axis=1)
'''
batchX =  [[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]]
inputs_series = [[1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 1, 1], [0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]
'''




print('inputs_series, info = ', inputs_series, '\n')
# inputs_series = tf.Print(inputs_series, [inputs_series, inputs_series.shape], 'inputs_series = ')


# Forward pass    
current_state = init_state    
states_series = []

index = 0
for current_input in inputs_series:    
    current_input = tf.Print(current_input, [current_input, current_input.shape, 'index = ' + str(index)], message='current_input0 = ', summarize=100)
    # [1, 0, 0, 1, 0]
    current_input = tf.reshape(current_input, [batch_size, 1])  # shape: (5, 1)。
    current_input = tf.Print(current_input, [current_input, current_input.shape, 'index = ' + str(index)], message='current_input1 = ', summarize=100)
    # [[1][0][0][1][0]]
    print('current, info = ', current_input, '\n')


    current_state = tf.Print(current_state, [current_state, current_state.shape, 'index = ' + str(index)], message='current_state = ', summarize=100)
    # [[0.788977802 -0.758756518 0.878002822 0.829297543][0.485389352 -0.58057 0.708371818 -0.412607044][0.551199 0.24081631 0.932751775 0.228438258][-0.816147685 -0.0264752097 -0.915767789 -0.879304767][-0.751862705 -0.0714274198 0.888504 0.431245238]][5 4]

    # 列后追加
    # input_and_state_concatenated = tf.concat(1, [current_input, current_state])  # Increasing number of columns    
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)    # shape (5, 5)
    input_and_state_concatenated = tf.Print(input_and_state_concatenated, [input_and_state_concatenated, input_and_state_concatenated.shape, 'index = ' + str(index)], message='input_and_state_concatenated = ', summarize=100)
    # [[1 -0.732565463 -0.0681669414 0.908388376 0.454913557][0 0.644927859 0.211452574 -0.751372695 -0.90508461][0 -0.737988293 0.234725475 -0.907563686 -0.87096405][1 0.661614358 -0.143374264 -0.873949289 -0.301904887][0 -0.71041739 -0.154791594 -0.74115169 -0.793806553]][5 5]

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition    
    next_state = tf.Print(next_state, [next_state, next_state.shape, 'index = ' + str(index)], message='next_state = ', summarize=100)
    # [[0.831956625 0.16452603 -0.891076565 -0.345829368][-0.718315482 -0.751014352 0.862468779 0.130369157][-0.562301695 0.132066295 -0.847182155 -0.863753378][-0.752997875 -0.124342352 0.488403261 0.874053717][-0.733458877 -0.0517550744 -0.960962713 -0.207119182]][5 4]

    states_series.append(next_state)    
    current_state = next_state  
    index += 1  

# loss
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition  
print('logits_series = ', logits_series)  
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]    
print('predictions_series = ', predictions_series)  


losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
print('losses = ', losses) 

total_loss = tf.reduce_mean(losses)    
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)    


def plot(loss_list, predictions_series, batchX, batchY):    
    plt.subplot(2, 3, 1)    
    plt.cla()    
    plt.plot(loss_list)    

    for batch_series_idx in range(5):    
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]    
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])    

        plt.subplot(2, 3, batch_series_idx + 2)    
        plt.cla()    
        plt.axis([0, truncated_backprop_length, 0, 2])    
        left_offset = range(truncated_backprop_length)    
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")    
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")    
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")    

    plt.draw()    
    plt.pause(0.0001)    



with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())    
    plt.ion()    
    plt.figure()    
    plt.show()    
    loss_list = []    

    for epoch_idx in range(1):    # range(num_epochs)   做 100 轮，每次都新生成 50000 个数据
        x, y = generateData()  
        _current_state = np.zeros((batch_size, state_size))    # (5, 4)
        print(" ++++++++ New data, epoch", epoch_idx)
        print("x = ", x, "\ny = ", y, "\n_current_state = ", _current_state) 

        for batch_idx in range(2):    # range(num_batches)
            start_idx = batch_idx * truncated_backprop_length    
            end_idx = start_idx + truncated_backprop_length   
            print("start_idx = ", start_idx, "\nend_idx = ", end_idx)


            batchX = x[:,start_idx:end_idx]    
            batchY = y[:,start_idx:end_idx]    
            print("batchX = ", batchX, "\nbatchY = ", batchY)


            _total_loss, _train_step, _current_state, _predictions_series = sess.run([total_loss, train_step, current_state, predictions_series],    
                feed_dict = {    
                    batchX_placeholder : batchX,    
                    batchY_placeholder : batchY,    
                    init_state : _current_state    
                })    

            loss_list.append(_total_loss)    
            

            if batch_idx % 100 == 0:    
                print("Step",batch_idx, "Loss", _total_loss)    
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()