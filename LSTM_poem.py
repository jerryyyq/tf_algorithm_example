#coding=utf-8

import collections
import time  
import numpy as np
import tensorflow as tf  

from common import *

# -------------------------------数据预处理--------------------------- # 

POETRY_FILE = 'data_source/poetry.txt'  
MODEL_FILE = 'model/poetry/best.ckpt'

FILL_WORD = ' '   # 用于填充的字符


'''
从诗集文件中读取出所有的内容列表（去掉标题，选择字数在 5 到 79 的行，并在每行左右添加 '[' 和 ']' 字符）
返回的列表已被排序，字数少的排在前面
例如：
poetry_file = 'data_source/poetry.txt'
return: [......, '[二月江南山水路，李花零落春无主。一个鱼儿无觅处，风和雨，玉龙生甲归天去。]', '[我有屋三椽，住在灵源。无遮四壁任萧然。万象森罗为斗拱，瓦盖青天。无漏得多年，结就因缘。修成功行满三千。降得火龙伏得虎，陆路神仙。]', '[天不高，地不大。惟有真心，物物俱含载。不用之时全体在。用即拈来，万象周沙界。虚无中，尘色内。尽是还丹，历历堪收采。这个鼎炉解不解。养就灵乌，飞出光明海。]', '[东与西，眼与眉。偃月炉中运坎离，灵砂且上飞。最幽微，是天机，你休痴，你不知。]', '[江南鼓，梭肚两头栾。钉著不知侵骨髓，打来只是没心肝。空腹被人谩。]', '[暂游大庾，白鹤飞来谁共语？岭畔人家，曾见寒梅几度花。春来春去，人在落花流水处。花满前蹊，藏尽神仙人不知。]']
'''
def get_contents_from_poetry_file(poetry_file):
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8',) as f:  
        for line in f:  
            try:  
                title, content = line.strip().split(':')

                # 去掉空格
                content = content.replace(' ', '')
                # 忽略掉含有特殊字符的内容
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:  
                    continue

                # 忽略掉超短或超长的内容
                if len(content) < 5 or len(content) > 79:
                    continue

                # 在每行内容两端添加 '[' 和 ']'
                content = '[' + content + ']'  
                poetrys.append(content)

            except Exception as e:   
                pass

    # 按诗的字数排序，字数少的在前
    poetrys = sorted(poetrys,key=lambda line: len(line))
    print('添加到集合中的总行数: ', len(poetrys))
    return poetrys


# x_batches、 x_batches shape: [n_chunk, batch_size, length]
def get_x_and_y_batche_from_poetry_file(poetry_file, batch_size):
    poetrys = get_contents_from_poetry_file(poetry_file)
    all_words = content_list_to_all_words(poetrys)

    # 添加用于补位的特殊字符
    all_words += FILL_WORD
    print('总字数: ', len(all_words))

    # words = 列表：所有的不重复的字，按出现次数从高到低排序，words = ('，', '。', ']', '[', '不', '人', '山', '风', '日', '无', '一', ...... '幺', FILL_WORD) 共 6110 个
    # word_num_map = {'庸': 2601, '纚': 4887, '惠': 1237, ..., '，': 0, ..., FILL_WORD: 6109, ..., '幺': 6108}
    words, word2vec_map = all_words_to_word_num_map(all_words)

    # poetrys_vector = [[3, 28, 544, 104, 720, 1, 2], [3, 649, 48, 9, 2147, 1, 2], [3, 424, 4867, 2127, 1100, 1, 2], [3, 345, 1146, 2615, 2671, 1, 2], [3, 822, 10, 1366, 332, 1, 2], ......]
    poetrys_vector = words_to_vectors(poetrys, word2vec_map)

    return (words, word2vec_map, *sentence_list_to_x_y_batches(poetrys_vector, batch_size, word2vec_map[FILL_WORD]) )

    
    
#---------------------------------------RNN--------------------------------------#
# 定义RNN  
def neural_network(input_data, batch_size, vector_number, model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell       # 有文章说 1.0 tf.nn.rnn_cell -> tf.contrib.rnn，但好像不需要
    elif model == 'gru':  
        cell_fun = tf.nn.rnn_cell.GRUCell 
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)    # shape: [64, 128]   值为 全 0.0
    # print('initial_state', initial_state)   # Tuple( LSTMStateTuple(c=shape=(64, 128), h=shape=(64, 128)), LSTMStateTuple(c=shape=(64, 128), h=shape=(64, 128)) )

    with tf.variable_scope('rnnlm'):  
        softmax_w = tf.get_variable("softmax_w", [rnn_size, vector_number])  # 此处应该不需要 +1, 猜测是原作者考虑到在 words 最后添加的 ' ', 但其实此时空格已经计算在内了
        softmax_b = tf.get_variable("softmax_b", [vector_number])  
        with tf.device("/cpu:0"):    # 此时，下面的 Tensor 是储存在内存里的，而非显存里。
            embedding = tf.get_variable("embedding", [vector_number, rnn_size])    # embedding shape: [vector_number, rnn_size] 个 -1 ~ 1 之间的随机唯一值
            inputs = tf.nn.embedding_lookup(embedding, input_data)  # input_data shape: [batch_size, length]；inputs shape: [batch_size, length, rnn_size]
            # inputs = tf.Print(inputs, [inputs], 'inputs')
    
    # with tf.control_dependencies([tf.Print(initial_state, [initial_state], "initial_state")]):  # [0, 0, 0, ......]
    #    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  # outputs shape=(64, ?, 128)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  # outputs shape=(64, ?, 128)


    # print('outpurs: ', outputs)
    # outputs = tf.Print(outputs, [outputs], 'outputs', 20, 7)
    output = tf.reshape(outputs,[-1, rnn_size])   # output shape=(?, 128)
    #print('output: ', output)
    #output = tf.Print(output, [output], 'output')
    
    logits = tf.matmul(output, softmax_w) + softmax_b  
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state


#训练
def train_neural_network(poetry_file, train_wheels, batch_size):
    words, word2vec_map, x_batches, y_batches = get_x_and_y_batche_from_poetry_file(poetry_file, batch_size)
    vector_number = len(words)
    print(vector_number, len(x_batches), len(y_batches))

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    logits, last_state, _, _, _ = neural_network(input_data, batch_size, vector_number)
    last_state = tf.convert_to_tensor(last_state)  # 不要放在循环中，类似的还有 tf.train.Saver()。其他 tf.zeros_like(), tf.ones_like() 都不要放循环里

    targets = tf.reshape(output_targets, [-1])
    if tf.__version__ < '1':
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], vector_number)
    else:
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], vector_number)

    cost = tf.reduce_mean(loss)

    learning_rate = tf.Variable(0.0, trainable=False)  
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)  
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        if tf.__version__ < '1':
            saver = tf.train.Saver( tf.all_variables() )
            sess.run(tf.initialize_all_variables())
        else:
            saver = tf.train.Saver( tf.global_variables() )
            sess.run(tf.global_variables_initializer())

        best_loss = float('Inf')
        epoch_loss = 0

        for epoch in range(train_wheels):     # 对全部数据进行 50 轮次重复训练
            print( sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch))) )
            for n in range(len(x_batches)):  # 按 batch_size 对全部数据分批进行一轮训练
                train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})  
                epoch_loss += train_loss
                print(epoch, n, train_loss)


            if best_loss > epoch_loss:
                best_loss = epoch_loss

                save_path = saver.save( sess, MODEL_FILE )    # saver.save(sess, MODEL_FILE, global_step=epoch)
                print( "Model saved in file: {}, epoch_loss = {}" . format(save_path, epoch_loss) )
                if 0.0 == epoch_loss:
                    print('epoch_loss == 0.0, exited!')
                    break

            epoch_loss = 0


#-------------------------------生成古诗---------------------------------#  
def weights_to_word(weights, words):  
    t = np.cumsum(weights)
    s = np.sum(weights)  
    sample = int(np.searchsorted(t, np.random.rand(1)*s))    # sample = int(np.searchsorted(t, np.random.rand(1)*s)) 
    return words[sample] 


# 使用训练完成的模型
def gen_poetry(poetry_file, begin_word = '['):
    batch_size = 1
    words, word2vec_map, x_batches, y_batches = get_x_and_y_batche_from_poetry_file(poetry_file, batch_size)
    vector_number = len(words)
    print(vector_number, len(x_batches), len(y_batches))

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    _, last_state, probs, cell, initial_state = neural_network(input_data, batch_size, vector_number)

    with tf.Session() as sess:
        if tf.__version__ < '1':
            saver = tf.train.Saver( tf.all_variables() )
            sess.run(tf.initialize_all_variables())
        else:
            saver = tf.train.Saver( tf.global_variables() )
            sess.run(tf.global_variables_initializer())
   
        saver.restore( sess, MODEL_FILE )

        state_ = sess.run(cell.zero_state(batch_size, tf.float32))

        x = [[word2vec_map['[']] for i in range(batch_size)]   #  x = np.array([list(map(word2vec_map.get, '['))])  
        [probs_, state_] = sess.run( [probs, last_state], feed_dict={input_data: x, initial_state: state_} )  
        word = weights_to_word(probs_, words)


        #word = words[np.argmax(probs_)]
        poem = ''  
        while word != ']':
            poem += word
            x = [[word2vec_map[word]] for i in range(batch_size)]    # 其实可以写为： x = [[word2vec_map[word]]]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
            word = weights_to_word(probs_, words)  
            #word = words[np.argmax(probs_)]  
        return poem  



def gen_poetry_with_head(poetry_file, head):  
    batch_size = 1
    words, word2vec_map, x_batches, y_batches = get_x_and_y_batche_from_poetry_file(poetry_file, batch_size)
    vector_number = len(words)
    print(vector_number, len(x_batches), len(y_batches))

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    _, last_state, probs, cell, initial_state = neural_network(input_data, batch_size, vector_number)

    with tf.Session() as sess:  
        if tf.__version__ < '1':
            saver = tf.train.Saver( tf.all_variables() )
            sess.run(tf.initialize_all_variables())
        else:
            saver = tf.train.Saver( tf.global_variables() )
            sess.run(tf.global_variables_initializer())

        saver.restore(sess, MODEL_FILE)  

        state_ = sess.run(cell.zero_state(1, tf.float32))
        poem = ''  
        i = 0  
        for word in head:  
            while word != '，' and word != '。':  
                poem += word  
                x = [[word2vec_map[word]] for i in range(batch_size)]     # x = np.array([list(map(word2vec_map.get, word))])  
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                word = weights_to_word(probs_, words)  
                time.sleep(1)  
            if i % 2 == 0:  
                poem += '，'  
            else:  
                poem += '。'  
            i += 1

        return poem  




if __name__ == '__main__':
    # 下面这三个场景不能同时运行，每次只能运行一个场景
    # 场景一：使用训练数据进行模型训练
    # train_neural_network(POETRY_FILE, 50, 64)

    # 场景二：使用验证数据来生成
    print(gen_poetry(POETRY_FILE, '我'))

    # 场景三：使用训练好的模型来生成藏头诗
    # print( gen_poetry_with_head(POETRY_FILE, '一二三四') )