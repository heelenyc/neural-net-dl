import numpy as np


class DefaultActFunc:
    fun_name = 'default'

    @staticmethod
    def active(x):
        y = x
        return y

    @staticmethod
    def active_derivative(x):
        return 1

    @classmethod
    def info(cls):
        print('{}'.format(cls.fun_name))

    @classmethod
    def get_name(cls):
        return cls.fun_name


class Sigmoid(DefaultActFunc):
    fun_name = 'sigmoid'

    @staticmethod
    def active(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def active_derivative(x):
        a = Sigmoid.active(x)
        return a * (1 - a)

# print(DefaultActFunc.info())
