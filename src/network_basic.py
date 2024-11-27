"""
s型神经元 、反向传播基本实现
"""
import time

import numpy as np
import activation_funs
import cost_funs


class FullConnectionLayer:
    """全连接层"""

    def __init__(self, pre_layer_size, size, activation_function, is_output=False):
        """
        :param size: 自身规模
        :param pre_layer_size: 前输入层的规模
        """
        if size < 1 or pre_layer_size < 1:
            raise ValueError(
                'invalid params for FullConnectionLayer: size {} pre_layer size {}'.format(size, pre_layer_size))
        self.size = size
        self.pre_layer_size = pre_layer_size
        # self.weights = np.random.randn(size, pre_layer_size)
        # self.biases = np.random.randn(size, 1)
        self.weights = np.ones((size, pre_layer_size))
        self.biases = np.ones((size, 1))
        self.activation_fun = activation_function
        self.is_output = is_output

    def forward(self, input_x):
        """
        计算本层的output_z output_a
        :param input_x:
        :return: output_z, output_a
        """
        output_z = np.matmul(self.weights, input_x) + self.biases
        output_a = self.activation_fun.active(output_z)
        return output_z, output_a

    def info(self):
        """
        print info about this layer
        :return:
        """
        print('size :{}  input_size: {} Activation fun : {} '.format(self.size, self.pre_layer_size,
                                                                     self.activation_fun.get_name()))
        print('Weights[{}] :{} '.format(self.weights.shape, self.weights))
        print('Biases :{} '.format(self.biases))


class NetworkBasic:
    def info(self):
        """
        打印网络结构
        :return:
        """
        print("Sizes of layers: {}".format(self.sizes))
        for lay in self.layers:
            lay.info()

    def __init__(self, layer_sizes, cost_fun):
        """
        :param layer_sizes: [num_input,num_hidden,,,num_output]
        """
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.cost_fun = cost_fun
        self.layers = []
        # 构造每一层，使用sigmoid激活函数
        for i in range(1, self.num_layers - 1):
            self.layers.append(FullConnectionLayer(layer_sizes[i - 1], layer_sizes[i], activation_funs.Sigmoid))
        # 最后一层单独考虑，暂时不使用激活函数
        self.layers.append(
            FullConnectionLayer(layer_sizes[-2], layer_sizes[-1], activation_funs.DefaultActFunc,True))

    def forward(self, input_x):
        """
        根据单个参数计算网络输出，需要记录计算过程中每一层的中间产出
        :param input_x:
        :return:output_z, output_a
        """
        # 逐层向前计算输出
        output_z = []
        output_a = []
        x = input_x
        for layer in self.layers:
            o_z, o_a = layer.forward(x)
            output_z.append(o_z)
            output_a.append(o_a)
            x = o_a
        # 网络的输出
        return output_z, output_a

    def cost(self, input_as, output_ys):
        """
        计算代价
        :param input_as:
        :param output_ys:
        :return:
        """
        return self.cost_fun.cost(input_as, output_ys)

    def save(self, filename):
        """
        网络参数持久化
        :param filename:
        :return:
        """
        return

    def load(self, filename):
        """
        加载网络参数
        :param filename:
        :return:
        """
        return

    def backward(self, o_z, o_a):
        """
        从输出层往输入层，逐层计算各参数偏导
        :param o_z:
        :param o_a:
        :return:
        """
        for layer in reversed(self.layers):
            if layer.is_output:
                pass
            else:
                pass


    def train_degrade(self, train_data, learning_rate, num_epochs):
        """
         梯度下降算法训练
        :param train_data: 输入的训练数据的格式
            input_xs = [np.array([1, 2, 3]).reshape((3, 1))]
            expect_ys = [np.array([4, 5]).reshape(2, 1)]
            train_data = list(zip(input_xs, expect_ys))
        :param learning_rate:
        :param num_epochs:
        :return:
        """
        last_checked_ts = time.time()
        batch_epoch = 10
        for epoch in range(num_epochs):
            if epoch % batch_epoch == 0:
                last_checked_ts = time.time()
                print("Epoch {}/{} begin".format(epoch + 1, num_epochs))

            a_s = []
            cast_s = []
            for x, y in train_data:
                # 这里输入和输出拆成了单个一对一对的
                o_z, o_a = self.forward(x)
                a_s.append(o_a[-1])  # 记录网络最终的输出，用于计算评估代价
                cast_s.append(self.cost(o_a[-1], y))  # 每个输入输出对应一个代价值
                # TODO：对应每次输出，计算当前x输入下的各参数偏导
                self.backward(o_z, o_a)
                # TODO: 优化成小批量的算法

            cost = np.mean(cast_s)
            # TODO：所有偏导求（整体样本空间的）均值；

            # TODO：使用偏导，结合学习率，修正参数；
            # self.

            # 一次epoch完成

            print("output: {}".format(a_s))
            print("Cost: {:.6f}".format(cost))

            if epoch % batch_epoch == 0:
                print("Epoch {}/{} end, spend {:.2f}s".format(epoch + 1, num_epochs, time.time() - last_checked_ts))


net = NetworkBasic(layer_sizes=[3, 2, 2], cost_fun=cost_funs.QuadraticCost)
net.info()
# output = net.forward([[[1], [1], [1]], [[1], [1], [1]]])
# print(output)
# training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
# training_results = [vectorized_result(y) for y in tr_d[1]]
# training_data = list(zip(training_inputs, training_results))
tr_dx = [[1, 1, 1], [1, 1, 1]]
tr_dy = [[1, 1], [1, 1]]
input_xs = [np.reshape(x, (3, 1)) for x in tr_dx]
expect_ys = [np.reshape(y, (2, 1)) for y in tr_dy]
trains = list(zip(input_xs, expect_ys))  # data like pre
print('train datas :{}'.format(trains))
net.train_degrade(trains, 0.01, 1)
