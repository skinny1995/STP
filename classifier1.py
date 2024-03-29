# -*- coding: utf-8 -*-
# !@time: 2019/6/21 19:11
# !@author: superMC @email: 18758266469@163.com
# !@fileName: classifier.py

# -*- coding: utf-8 -*-
# !@time: 19-6-2 下午6:24
# !@author: superMC @email: 18758266469@163.com
# !@fileName: data_processing.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset

HIDDEN_UNITS = [25, 25, 15]  # 节点数，隐藏层数
MODER_DIR = 'model'  # 保存变量
tf.logging.set_verbosity(tf.logging.ERROR)  # 将 TensorFlow 日志信息输出到屏幕
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format  # 为了观察数据方便，最多只显示10行数据
dataframe = pd.read_csv("data.csv", sep=",")  # 加载数据集
dataframe_predicted = pd.read_csv("packet_f.csv", sep=",")
train_dataframe = dataframe.head(700000)  # 取前700000个数据
test_dataframe = dataframe.tail(74013)  # 取后4013个数据
train_dataframe = train_dataframe.reindex(np.random.permutation(train_dataframe.index))
# permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组
# reindex重新索引

vocab = {'SERVICES': 0, 'VOIP': 1, 'P2P': 2, 'ATTACK': 3}  # 标签字典
vocab_opposite = {0: 'SERVICES', 1: 'VOIP', 2: 'P2P', 3: 'ATTACK'}
features_names = ["xSrcPort",
                  "xDstPort",
                  "xPkt",
                  "xMin_Pbyte0",
                  "xMin_Pbyte1",
                  "xMax_Pbyte0",
                  "xMax_Pbyte1",
                  "xIni_Pbyte0",
                  "xMax_ConsPkt0"]


# 特征名字
# 取特征，取标签
def preprocess_dataframe(dataframe):
    selected_features = dataframe[features_names]
    selected_labels = dataframe[["Class"]]
    processed_features = selected_features.copy()
    preprocess_labels = selected_labels.copy()
    preprocess = preprocess_labels['Class'].apply(lambda x: vocab[x])
    return processed_features, preprocess


def preprocess_dataframe2(dataframe):
    selected_features = dataframe[features_names]
    # selected_labels = dataframe[["Class"]]
    processed_features = selected_features.copy()
    # preprocess_labels = selected_labels.copy()
    # preprocess = preprocess_labels['Class'].apply(lambda x: vocab[x])
    return processed_features


def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    # 将熊猫数据转换成NP数组
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    # 构造数据集，并配置批处理/重复
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    # 如果指定，则对数据进行洗牌。
    if shuffle:
        ds = ds.shuffle(buffer_size=100000)
        # 从训练集中随机取1万个数据

    # Return the next batch of data.
    # 返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def my_input_fn2(features, batch_size=1, shuffle=False, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    # 将熊猫数据转换成NP数组
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    # 构造数据集，并配置批处理/重复
    ds = Dataset.from_tensor_slices((features))  # warning: 2GB limit

    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    # 如果指定，则对数据进行洗牌。
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        # 从训练集中随机取1万个数据

    # Return the next batch of data.
    # 返回下一批数据
    features = ds.make_one_shot_iterator().get_next()

    return features


train_features, train_labels = preprocess_dataframe(train_dataframe)
test_features, test_labels = preprocess_dataframe(test_dataframe)
dataframe_predicted_features = preprocess_dataframe2(dataframe_predicted)
features_columns = [tf.feature_column.numeric_column(x) for x in features_names]  # 读特征列

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
run_config = tf.estimator.RunConfig().replace(session_config=run_config)

classifier = tf.estimator.DNNClassifier(feature_columns=features_columns, hidden_units=HIDDEN_UNITS, n_classes=4,
                                        optimizer=tf.train.ProximalGradientDescentOptimizer(l2_regularization_strength=0.001, learning_rate=0.0001),
                                        model_dir=MODER_DIR, dropout=0.1, batch_norm=True, config=run_config)
# feature_columns作为特征列，但是这里不添加数据，仅仅是使用tf.feature_column添加数据特征；
# 数据特征相当于一个字典的键值，这个键值是真正训练时输入数据的特征列的名称。

# hidden_units 隐藏层网络
# n_classes 输出层的分类
# optimizer表示使用的训练函数
periods = 10
for period in range(0, periods):
    classifier.train(input_fn=lambda: my_input_fn(train_features, train_labels, batch_size=1024), steps=5000)
# max_steps=775000
# classifier.train(input_fn=lambda: my_input_fn(train_features, train_labels, batch_size=2000), steps=5000)
    # 训练
    evaluates = classifier.evaluate(
        input_fn=lambda: my_input_fn(test_features, test_labels, batch_size=1024, shuffle=False, num_epochs=1))
    # predictions = list(
    #   classifier.predict(input_fn=lambda: my_input_fn(test_features, test_labels, shuffle=False, num_epochs=1)))
    # validation_probabilities = np.array([item['probabilities'] for item in predictions])
    # predicts = list(np.argmax(validation_probabilities, axis=1))
    # test_labels = list(test_labels)
    # n = 0
    # for i in range(74013):
    #   if test_labels[i] == predicts[i]:
    #      n += 1
    # acc2=n / 74013
    acc1 = evaluates['accuracy']
    print("---------------------------")
    print("  period %d :acc1= %f" % (period, acc1))
    print("---------------------------")
'''
predictions = list(
    classifier.predict(input_fn=lambda: my_input_fn2(dataframe_predicted_features, shuffle=False, num_epochs=1)))
validation_probabilities = np.array([item['probabilities'] for item in predictions])
predicts = list(np.argmax(validation_probabilities, axis=1))
# print("predicts=",predicts)
print("predictsnumber", len(predicts))
predicts2 = pd.DataFrame(data=predicts, columns=['Class'])
print("predictsDataFrame=", predicts2)
predicts3 = predicts2['Class'].apply(lambda x: vocab_opposite[x])
dataframe_predicted['Class'] = predicts3

dataframe_predicted.to_csv('packet_f2.csv', index=False, header=True)
'''


