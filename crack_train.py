from DL_CNN import DL_CNN

one_cnn = DL_CNN('/home/yangyuqi/work/crack_sample/tf_record/', 350, 430, 1, 7)
one_cnn.train(1000, 165)