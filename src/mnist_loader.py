import gzip
import pickle

import numpy as np


def load_data():
    """
    https://yann.lecun.com/exdb/mnist/
    60000 训练数据 + 10000 测试数据
    经过改造之后 50000 训练 + 10000 验证 + 10000 测试
    每个都是两个数组，第一个数据是784元素的数组numpy 的一维array，对应每个图里784个元素，第二个是对应的数字 int64
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """

    :return: training_data = [(array of 784*1,array of 10*1,int64),(),()], validation_data, test_data
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    结果向量化，10*1的二维数组，对应的数字位上的值位[1]
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


normal_result_vector = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])


def re_vectorized_result(j):
    """
    结果向量化，10*1的二维数组，对应的数字位上的值位[1]
    """
    return int(np.matmul(normal_result_vector, j)[0][0])
