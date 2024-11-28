"""
s型神经元 、反向传播基本实现
"""
import time
import random

import numpy as np
import activation_funs
import cost_funs
import full_con_layer


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
            self.layers.append(
                full_con_layer.FullConnectionLayer(layer_sizes[i - 1], layer_sizes[i], activation_funs.Sigmoid))
        # 最后一层单独考虑，暂时不使用激活函数
        self.layers.append(
            full_con_layer.FullConnectionLayer(layer_sizes[-2], layer_sizes[-1], activation_funs.Sigmoid, True))

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

    def backward(self, input_x, expect_y, o_z, o_a, mini_num):
        """
        从输出层往输入层，逐层计算各参数偏导
        :param mini_num: 小样本数量
        :param expect_y:
        :param input_x:
        :param o_z:
        :param o_a:
        :return:
        """
        for layer in reversed(self.layers):
            if layer.is_output:  # 首先执行的
                layer.factor = self.cost_fun.cost_prime(layer.output_a,
                                                        expect_y) * layer.activation_fun.active_derivative(
                    layer.output_z)
                layer.degrade_b += layer.factor / mini_num
                layer.degrade_w += np.matmul(layer.factor, layer.input_x.T) / mini_num
            else:
                layer.factor = np.matmul(next_layer.weights.T,
                                         next_layer.factor) * layer.activation_fun.active_derivative(
                    layer.output_z)
                layer.degrade_b += layer.factor / mini_num
                layer.degrade_w += np.matmul(layer.factor, layer.input_x.T) / mini_num

            next_layer = layer  # 传到下一次计算

    def step(self, l_rate):
        """
        使用计算出来的偏导优化参数
        :return:
        """
        for layer in self.layers:
            layer.step(l_rate)

    def clear(self):
        """
        清楚当前的梯度数据
        :return:
        """
        for layer in self.layers:
            layer.clear()

    def train_degrade(self, train_data, learning_rate, num_epochs, mini_num, test_data=None):
        """
         梯度下降算法训练
        :param test_data:
        :param mini_num:
        :param train_data: 输入的训练数据的格式
            input_xs = [np.array([1, 2, 3]).reshape((3, 1))]
            expect_ys = [np.array([4, 5]).reshape(2, 1)]
            train_data = list(zip(input_xs, expect_ys))
        :param learning_rate:
        :param num_epochs:
        :return:
        """
        last_checked_ts = time.time()
        epoch_print_threshold = 1
        # mini_num = len(train_data)  # mini_num

        for epoch in range(num_epochs):
            if epoch % epoch_print_threshold == 0:
                last_checked_ts = time.time()
                # print("Epoch {}/{} begin".format(epoch + 1, num_epochs))

            a_s = []
            cast_s = []
            random.shuffle(train_data)
            mini_train_batches = [
                train_data[k:k + mini_num]
                for k in range(0, len(train_data), mini_num)]
            for mini_train_data in mini_train_batches:
                self.clear()  # 清理梯度信息，为这次迭代做准备；
                for x, y in mini_train_data:
                    # 这里输入和输出拆成了单个一对一对的
                    o_z, o_a = self.forward(x)
                    a_s.append(o_a[-1])  # 记录网络最终的输出，用于计算评估代价
                    cast_s.append(self.cost(o_a[-1], y))  # 每个输入输出对应一个代价值
                    # 对应每次输出，计算当前x输入下的各参数偏导  backward 反向传播
                    self.backward(x, y, o_z, o_a, mini_num)

                cost = np.mean(cast_s)
                # 使用偏导，结合学习率，修正参数；
                self.step(learning_rate)
            # 一次epoch完成

            # print("Output: {}".format(a_s))

            if epoch % epoch_print_threshold == 0:
                if test_data:
                    print(
                        "Epoch {}/{} end, cost {:.6f}, "
                        "took {:.2f}s, accuracy {}/{}".format(epoch + 1, num_epochs, cost,
                                                              time.time() - last_checked_ts,
                                                              self.evaluate(test_data),
                                                              len(test_data)))
                else:
                    print("Epoch {}/{} end, cost {:.6f} "
                          "took {:.2f}s".format(epoch + 1, num_epochs, cost,
                                                time.time() - last_checked_ts))

        print("Train end!")

    def evaluate(self, test_data):
        """
        验证结果
        :param test_data:
        :return:
        """
        test_results = []
        for (x, y) in test_data:
            _, a = self.forward(x)
            test_results.append((np.argmax(a[-1]), y))
        return sum(int(x == y) for (x, y) in test_results)

# net = NetworkBasic(layer_sizes=[4, 3, 2], cost_fun=cost_funs.QuadraticCost)
# net.info()
# # output = net.forward([[[1], [1], [1]], [[1], [1], [1]]])
# # print(output)
# tr_dx = [[1, 1, 1, 1], [1, 1, 1, 1]]
# tr_dy = [[5, 5], [5, 5]]
# input_xs = [np.reshape(x, (4, 1)) for x in tr_dx]
# expect_ys = [np.reshape(y, (2, 1)) for y in tr_dy]
# trains = list(zip(input_xs, expect_ys))  # data like pre
# print('Train datas :{}'.format(trains))
# net.train_degrade(trains, 0.1, 100)
# net.info()
