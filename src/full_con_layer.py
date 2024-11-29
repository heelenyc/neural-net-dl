import numpy as np


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
        self.weights = np.random.randn(size, pre_layer_size)
        self.biases = np.random.randn(size, 1)
        # self.weights = np.ones((size, pre_layer_size))
        # self.biases = np.ones((size, 1))
        self.activation_fun = activation_function
        self.is_output = is_output

        # 梯度数据
        self.input_x = np.zeros((pre_layer_size, 1))
        self.output_z = np.zeros((size, 1))
        self.output_a = np.zeros((size, 1))
        self.factor = np.zeros((size, 1))
        self.degrade_b = np.zeros((size, 1))
        self.degrade_w = np.zeros((size, pre_layer_size))

    def forward(self, input_x, update_self=False):
        """
        计算本层的output_z output_a
        :param update_self: 是否更新自己的状态
        :param input_x:
        :return: output_z, output_a
        """
        if update_self:
            self.input_x = input_x
            self.output_z = np.matmul(self.weights, input_x) + self.biases
            self.output_a = self.activation_fun.active(self.output_z)
            return self.output_z, self.output_a
        else:
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
        print('Weights_degrade :{} '.format(self.degrade_w))
        print('Biases_degrade :{} '.format(self.degrade_b))

    def clear(self):
        self.input_x = np.zeros((self.pre_layer_size, 1))
        self.output_z = np.zeros((self.size, 1))
        self.output_a = np.zeros((self.size, 1))
        self.factor = np.zeros((self.size, 1))
        self.degrade_b = np.zeros((self.size, 1))
        self.degrade_w = np.zeros((self.size, self.pre_layer_size))

    def step(self, l_rate):
        self.biases = self.biases - l_rate * self.degrade_b
        self.weights = self.weights - l_rate * self.degrade_w
