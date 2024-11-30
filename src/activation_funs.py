import numpy as np
from scipy.special import expit


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
        """
        if np.all(z >= 0):  # 对sigmoid函数优化，避免出现极大的数据溢出
            return 1.0 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (1 + np.exp(z))
        # return expit(z)
        """
        if np.any(z > 100) or np.any(z < -100):
            pass
            # print(z)
        return 1.0 / (1 + np.exp(-z))

    @staticmethod
    def active_derivative(z):
        a = Sigmoid.active(z)
        return a * (1 - a)

# print(DefaultActFunc.info())
