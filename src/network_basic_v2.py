"""
s型神经元 、反向传播基本实现
"""
import random
import time

import numpy as np

import activation_funs
import cost_funs


class NetworkBasic:
    def info(self):
        """
        打印网络结构
        :return:
        """
        print("num_layer:{}".format(self.num_layers))
        print("sizes:{}".format(self.sizes))
        print("weights:{}".format(self.weights))
        print("biases:{}".format(self.biases))
        print("cost_fun:{}".format(self.cost_fun))
        print("activation_fun:{}".format(self.atv_fun))
        print("out_activation_fun:{}".format(self.output_atv_fun))

    def __init__(self, layer_sizes, cost_fun=cost_funs.QuadraticCost, atv_fun=activation_funs.Sigmoid,
                 output_atv_fun=activation_funs.Sigmoid, std_w=False, std_b=False, lbd=None):
        """
        :param layer_sizes: [num_input,num_hidden,,,num_output]
        """
        print("Initializing...")
        s = time.time()
        self.num_layers = len(layer_sizes)  # 网络的层数，包括输入和输出
        self.sizes = layer_sizes  # 网络的结构   [784, 30, 10]
        if std_w:  # 使用跟输入规模有关的正太分布
            self.weights = [np.random.normal(0, 1/np.sqrt(pre_size), (size, pre_size)) for pre_size, size in
                            zip(layer_sizes[:-1], layer_sizes[1:])]
        else:
            self.weights = [np.random.randn(size, pre_size) for pre_size, size in
                            zip(layer_sizes[:-1], layer_sizes[1:])]
        if std_b:  # 直观上理解，偏置的初始化根据输出的需要来确定
            self.biases = [np.random.uniform(-1, 1, (size, 1)) for size in layer_sizes[1:]]
        else:
            self.biases = [np.random.randn(size, 1) for size in layer_sizes[1:]]
        self.cost_fun = cost_fun
        self.atv_fun = atv_fun
        self.output_atv_fun = output_atv_fun
        self.lbd = lbd
        print("Init time:{}".format(time.time() - s))

    def forward(self, input_x):
        """
        :param input_x:  单个输入
        根据单个参数计算网络输出，返回每层的输出，作为输出的第一个元素；
        """
        # 逐层向前计算输出
        output_z = []
        output_a = [input_x]
        for w_s, b_s in zip(self.weights[:-1], self.biases[:-1]):
            o_z = np.matmul(w_s, output_a[-1]) + b_s
            o_a = self.atv_fun.active(o_z)
            output_z.append(o_z)
            output_a.append(o_a)

        # 最后一层
        o_z = np.matmul(self.weights[-1], output_a[-1]) + self.biases[-1]
        o_a = self.output_atv_fun.active(o_z)
        output_z.append(o_z)
        output_a.append(o_a)

        # 网络的输出
        return output_z, output_a

    def backward(self, expect_y, o_z, o_a, mini_w_d_s, mini_b_d_s):
        """
        从输出层往输入层，逐层计算各参数偏导
        :param expect_y:
        :param o_z:
        :param o_a: 第一个元素是网络的输入x
        :param mini_w_d_s: 生成的梯度直接迭代进去，采用这种方式为了点性能
        :param mini_b_d_s: 生成的梯度直接迭代进去，采用这种方式为了点性能
        :return:
        """

        # 最后一层的直接算；
        if self.cost_fun == cost_funs.CrossEntropyCost:
            factor = o_a[-1] - expect_y  # 交叉熵
        else:
            factor = self.cost_fun.cost_prime(o_a[-1], expect_y) * self.output_atv_fun.active_derivative(o_z[-1])
        mini_b_d_s[-1] += factor * 1
        mini_w_d_s[-1] += np.matmul(factor, o_a[-2].T)

        for layer_index in range(-2, self.num_layers * (-1), -1):  # 遍历不包括最后一层和第一层；# TODO cpu大户
            factor = np.matmul(self.weights[layer_index + 1].T, factor) * self.atv_fun.active_derivative(
                o_z[layer_index])
            mini_b_d_s[layer_index] += factor * 1
            mini_w_d_s[layer_index] += np.matmul(factor, o_a[layer_index - 1].T)

    def train_degrade(self, train_data, learning_rate, num_epochs, mini_num, test_data=None, dynamic_lr=False):
        """
         梯度下降算法训练
        """
        last_checked_ts = time.time()
        epoch_print_threshold = 1
        train_data_len = len(train_data)

        for epoch in range(num_epochs):
            if epoch % epoch_print_threshold == 0:
                last_checked_ts = time.time()
                # print("Epoch {}/{} begin".format(epoch + 1, num_epochs))

            a_s = []
            epoch_cost = 0.0
            factor_lr = 0.1

            if mini_num > len(train_data):
                mini_num = len(train_data)

            # 随机批量，加快学习；
            random.shuffle(train_data)
            mini_train_batches = [
                train_data[k:k + mini_num]
                for k in range(0, len(train_data), mini_num)]

            pre_delta_cost = 0.0
            pre_mini_cost = 0.0
            delta_cost = 0.0

            for mini_train_data in mini_train_batches:
                mini_cost_s = 0.0
                mini_b_d_s = [np.zeros(b.shape) for b in self.biases]  # copy.deepcopy(zero_mini_b_d_s)
                mini_w_d_s = [np.zeros(w.shape) for w in self.weights]  # copy.deepcopy(zero_w_d_s)

                for x, y in mini_train_data:
                    # 这里输入和输出拆成了单个一对一对的
                    o_z, o_a = self.forward(x)
                    a_s.append(o_a[-1])  # 记录网络最终的输出，用于计算评估代价
                    cur_cost = self.cost(o_a[-1], y)
                    epoch_cost += cur_cost / train_data_len  # 每个输入输出对应一个代价值
                    mini_cost_s += cur_cost / mini_num  # 计算代价很耗时，每个样本都要计算一次
                    # 对应每次输出，计算当前x输入下的各参数偏导  backward 反向传播
                    self.backward(y, o_z, o_a, mini_w_d_s, mini_b_d_s)

                # 动态调整学习率
                if dynamic_lr and pre_mini_cost > 0.0:
                    """在连续的小样本批次之间调整学习率，第一次和第二次信息不足，除外"""
                    delta_cost = mini_cost_s - pre_mini_cost
                    if 0 > delta_cost > pre_delta_cost:  # 代价在下降 比上一次降的慢 加速
                        learning_rate += factor_lr
                    if delta_cost > pre_delta_cost > 0 and learning_rate > 10 * factor_lr:  # 代价回升，并且比上一次回升的更厉害
                        learning_rate -= factor_lr  # 需要缩短距离更精确的去寻找极值点

                pre_mini_cost = mini_cost_s
                pre_delta_cost = delta_cost

                # 使用偏导，结合学习率，修正参数；
                self.step(learning_rate, mini_w_d_s, mini_b_d_s, mini_num, train_data_len)

            # 打印本次epoch信息
            if epoch % epoch_print_threshold == 0:
                if test_data:
                    print(
                        "Epoch {}/{} end, cost {:.6f}, lr:{:.2f} mini_num:{} "
                        "took {:.2f}s, accuracy {}/{}".format(epoch + 1, num_epochs, epoch_cost, learning_rate,
                                                              mini_num, time.time() - last_checked_ts,
                                                              self.evaluate(test_data),
                                                              len(test_data)))
                else:
                    print("Epoch {}/{} end, cost {:.6f} lr:{:.2f} mini_num:{} "
                          "took {:.2f}s".format(epoch + 1, num_epochs, epoch_cost, learning_rate,
                                                mini_num, time.time() - last_checked_ts))

            # 一次epoch完成

        print("Train end!")

    def step(self, l_rate, w_d_s, b_d_s, mini_num, train_data_size):
        """
        使用计算出来的偏导优化参数
        :return:
        """
        self.biases = [b - (l_rate / mini_num) * b_d for b, b_d in zip(self.biases, b_d_s)]
        if self.lbd:  # 权重衰减的参数跟样本的规模成反比
            self.weights = [w * (1 - l_rate * self.lbd / train_data_size) - (l_rate / mini_num) * w_d for w, w_d in
                            zip(self.weights, w_d_s)]
        else:
            self.weights = [w - (l_rate / mini_num) * w_d for w, w_d in zip(self.weights, w_d_s)]

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
