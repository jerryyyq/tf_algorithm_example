#coding=utf-8

import collections
import time  
import numpy as np
import tensorflow as tf  
    
# -------------------------------数据预处理--------------------------- # 

poetry_file = 'data_source/poetry.txt'  
MODEL_FILE = 'model/poetry/best.ckpt'

# 诗集  
poetrys = []  
with open(poetry_file, "r", encoding='utf-8',) as f:  
    for line in f:  
        try:  
            title, content = line.strip().split(':')  
            content = content.replace(' ','')  
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:  
                continue  
            if len(content) < 5 or len(content) > 79:
                print(content)
                continue  
            content = '[' + content + ']'  
            poetrys.append(content)  
        except Exception as e:   
            pass  

# poetrys = [......, '[二月江南山水路，李花零落春无主。一个鱼儿无觅处，风和雨，玉龙生甲归天去。]', '[我有屋三椽，住在灵源。无遮四壁任萧然。万象森罗为斗拱，瓦盖青天。无漏得多年，结就因缘。修成功行满三千。降得火龙伏得虎，陆路神仙。]', '[天不高，地不大。惟有真心，物物俱含载。不用之时全体在。用即拈来，万象周沙界。虚无中，尘色内。尽是还丹，历历堪收采。这个鼎炉解不解。养就灵乌，飞出光明海。]', '[东与西，眼与眉。偃月炉中运坎离，灵砂且上飞。最幽微，是天机，你休痴，你不知。]', '[江南鼓，梭肚两头栾。钉著不知侵骨髓，打来只是没心肝。空腹被人谩。]', '[暂游大庾，白鹤飞来谁共语？岭畔人家，曾见寒梅几度花。春来春去，人在落花流水处。花满前蹊，藏尽神仙人不知。]']

    
# 按诗的字数排序，字数少的在前
poetrys = sorted(poetrys,key=lambda line: len(line))  
print('唐诗总数: ', len(poetrys))  
    
# 统计每个字出现次数  
all_words = []  
for poetry in poetrys:  
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)  
count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
words, _ = zip(*count_pairs)  # words = 列表：所有的不重复的字，按出现次数从高到低排序，words = ('，', '。', ']', '[', '不', '人', '山', '风', '日', '无', '一', ...... '幺') 共 6109 个

# 取前多少个常用字  
words = words[:len(words)] + (' ',)  # 在后面补一个 (......, '幺', ' ')  共 6110 个

# 每个字映射为一个数字ID  
word_num_map = dict(zip(words, range(len(words))))   # word_num_map = {'庸': 2601, '纚': 4887, '惠': 1237, ..., '，': 0, ..., ' ': 6109, ..., '幺': 6108}
# 把诗转换为向量形式，参考TensorFlow练习1  
to_num = lambda word: word_num_map.get(word, len(words))  
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]  
#[[3, 28, 544, 104, 720, 1, 2], [3, 649, 48, 9, 2147, 1, 2], [3, 424, 4867, 2127, 1100, 1, 2], [3, 345, 1146, 2615, 2671, 1, 2], [3, 822, 10, 1366, 332, 1, 2], ......]
  
    
# 每次取64首诗进行训练。生成时要设置为 1
batch_size = 1    # 训练时要设置为：64，生成时要设置为：1
n_chunk = len(poetrys_vector) // batch_size  # 一共有多少个块
x_batches = []  # shape: [n_chunk, batch_size, length]
y_batches = []  
for i in range(n_chunk):  
    start_index = i * batch_size  
    end_index = start_index + batch_size  
    
    batches = poetrys_vector[start_index:end_index]  
    length = max(map(len,batches))                        # 本 batche 中最长的句子的长度
    xdata = np.full((batch_size,length), word_num_map[' '], np.int32)  
    for row in range(batch_size):  
        xdata[row,:len(batches[row])] = batches[row]  
    ydata = np.copy(xdata)                   # ydata = [[*xdata[i][1:], xdata[i][-1]] for i in range(len(xdata))]
    ydata[:,:-1] = xdata[:,1:]
    """ 
    xdata             ydata 
    [6,2,4,6,9]       [2,4,6,9,9] 
    [1,4,2,8,5]       [4,2,8,5,5] 
    """  
    x_batches.append(xdata)                               # 所有的句子，但其中每 batche 一组，每一组的长度是一样的
    y_batches.append(ydata)  
    
    
#---------------------------------------RNN--------------------------------------#

input_data = tf.placeholder(tf.int32, [batch_size, None])  
output_targets = tf.placeholder(tf.int32, [batch_size, None])  
# 定义RNN  
def neural_network(model='lstm', rnn_size=128, num_layers=2):  
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
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])  # 此处应该不需要 +1, 猜测是原作者考虑到在 words 最后添加的 ' ', 但其实此时空格已经计算在内了
        softmax_b = tf.get_variable("softmax_b", [len(words)+1])  
        with tf.device("/cpu:0"):    # 此时，下面的 Tensor 是储存在内存里的，而非显存里。
            embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])    # embedding shape: [len(words)+1, rnn_size] 个 -1 ~ 1 之间的随机唯一值
            inputs = tf.nn.embedding_lookup(embedding, input_data)  # input_data shape: [batch_size, length]；inputs shape: [batch_size, length, rnn_size]
            # inputs = tf.Print(inputs, [inputs], 'inputs')
    
    # with tf.control_dependencies([tf.Print(initial_state, [initial_state], "initial_state")]):  # [0, 0, 0, ......]
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
def train_neural_network():  
    logits, last_state, _, _, _ = neural_network()
    last_state = tf.convert_to_tensor(last_state)  # 不要放在循环中，类似的还有 tf.train.Saver()。其他 tf.zeros_like(), tf.ones_like() 都不要放循环里

    targets = tf.reshape(output_targets, [-1])
    if tf.__version__ < '1':
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    else:
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)  
    learning_rate = tf.Variable(0.0, trainable=False)  
    tvars = tf.trainable_variables()  
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)  
    optimizer = tf.train.AdamOptimizer(learning_rate)  
    train_op = optimizer.apply_gradients(zip(grads, tvars))  

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if tf.__version__ < '1':  
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.global_variables_initializer())
    
        for epoch in range(50):     # 对全部数据进行 50 轮次重复训练
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))  
            n = 0  
            for batche in range(n_chunk):  # 按 batch_size 对全部数据分批进行一轮训练
                train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})  
                n += 1  
                print(epoch, batche, train_loss)  

            if epoch % 7 == 0:
                print('save epoch = ', epoch, ' train_loss = ', train_loss) 
                saver.save(sess, MODEL_FILE, global_step=epoch)  




    #-------------------------------生成古诗---------------------------------#  
# 使用训练完成的模型  
   
def gen_poetry():  
    def to_word(weights):  
        t = np.cumsum(weights)  
        s = np.sum(weights)  
        sample = int(np.searchsorted(t, np.random.rand(1)*s))    # sample = int(np.searchsorted(t, np.random.rand(1)*s)) 
        return words[sample]  
   
    _, last_state, probs, cell, initial_state = neural_network()

    with tf.Session() as sess:
        if tf.__version__ < '1':
            saver = tf.train.Saver( tf.all_variables() )
            sess.run(tf.initialize_all_variables())
        else:
            saver = tf.train.Saver( tf.global_variables() )
            sess.run(tf.global_variables_initializer())
   
        saver.restore( sess, MODEL_FILE + '-49' )  
   
        state_ = sess.run(cell.zero_state(1, tf.float32))  
   
        x = np.array([list(map(word_num_map.get, '['))])  
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
        word = to_word(probs_)  
        #word = words[np.argmax(probs_)]  
        poem = ''  
        while word != ']':  
            poem += word  
            x = np.zeros((1,1))  
            x[0,0] = word_num_map[word]  
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
            word = to_word(probs_)  
            #word = words[np.argmax(probs_)]  
        return poem  
   
  

def gen_poetry_with_head(head):  
    def to_word(weights):  
        t = np.cumsum(weights)  
        s = np.sum(weights)  
        sample = int(np.searchsorted(t, np.random.rand(1)*s))  
        return words[sample]  
   
    _, last_state, probs, cell, initial_state = neural_network()  
   
    with tf.Session() as sess:  
        if tf.__version__ < '1':
            saver = tf.train.Saver( tf.all_variables() )
            sess.run(tf.initialize_all_variables())
        else:
            saver = tf.train.Saver( tf.global_variables() )
            sess.run(tf.global_variables_initializer())

        saver.restore(sess, MODEL_FILE + '-49')  
   
        state_ = sess.run(cell.zero_state(1, tf.float32))  
        poem = ''  
        i = 0  
        for word in head:  
            while word != '，' and word != '。':  
                poem += word  
                x = np.array([list(map(word_num_map.get, word))])  
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                word = to_word(probs_)  
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
    train_neural_network() 

    # 场景二：使用验证数据来生成
    # print(gen_poetry())

    # 场景三：使用训练好的模型来生成藏头诗
    # print(gen_poetry_with_head('一二三四'))