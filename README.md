# tf_algorithm_example
Tensorflow stand algorithm implementation example

这个工程是我对学习 Tensorflow 的总结，是各种算法的练习实现，包括中间调试输出，以利于对 Tensorflow 的学习。

关于 Tensorflow 的基础用法，网上相关教程已经很多，就不在这里累述。

大家可以参考学习，引用请注明出处：https://github.com/jerryyyq/tf_algorithm_example

谢谢。

代码在 python 3.5 和 Tensorflow 0.9 上调试通过

# 内容说明
## 机器学习
1. ML_Model.py 是机器学习模型基类
1. ML_Linear_Regression 是线性回归实现类
1. ML_Sigma_Regression 是对数几率回归实现类
1. ML_Softmax_Regression 是 Softmax 回归实现类

## 深度学习
1. DL_CNN.py 是 CNN 模型基类

## 辅助函数
1. common.py 是一些基础函数
1. read_csv 从csv文件读取数据的函数
1. img_to_tf_record.py 将 ’类型/本类下的图片‘ 这种组织形式下的图片集转化为 tf_record 形式的文件，供学习时按流式提供数据
1. split_olivettiface.py 将 data_source/olivettifaces.gif 分解为 ’类型/本类下的图片‘ 的标准目录形式


## 目录说明
1. data_source 目录内是学习用的数据样本


# 版权声明
Author: 杨玉奇

email: yangyuqi@sina.com

url: https://github.com/jerryyyq/tf_algorithm_example

copyright yangyuqi

著作权归作者 杨玉奇 所有。商业转载请联系作者获得授权，非商业转载请注明出处。

date: 2018-06-22