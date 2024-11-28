import numpy as np


class DefaultCost:
    fun_name = 'default_cost'

    @staticmethod
    def cost(ax_s, y_s):
        return np.abs(y_s - ax_s)

    @staticmethod
    def cost_prime(ax_s, y_s):
        return 1 if ax_s > y_s else -1

    @classmethod
    def info(cls):
        print('{}'.format(cls.fun_name))

    @classmethod
    def get_name(cls):
        return cls.fun_name


class MeanSquaredErrorCost(DefaultCost):
    """均方误差 代价函数"""
    fun_name = 'mean_squared_error'

    @staticmethod
    def cost_prime(ax_s, y_s):
        return y_s - ax_s

    @staticmethod
    def cost(ax_s, y_s):
        return np.mean((y_s - ax_s) ** 2)


class QuadraticCost(DefaultCost):
    """二次代价函数, 向量v的模"""
    fun_name = 'quadratic_cost'

    @staticmethod
    def cost_prime(ax_s, y_s):
        return ax_s - y_s

    @staticmethod
    def cost(ax_s, y_s):
        """
        计算单次输出与预期的二次代价值
        :param ax_s:  N * 1 的矩阵
        :param y_s:  N * 1 的矩阵
        :return:
        """
        return np.sum([(a - y) ** 2 / 2 for a, y in zip(ax_s, y_s)])

# print(DefaultActFunc.info())
# print(MeanSquaredErrorCost.cost(np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])))
# print(MeanSquaredErrorCost.cost(np.array([[1, 2]]), np.array([[3, 4]])))
# print(QuadraticCost.cost(np.array([1, 2, 3]), np.array([3, 4, 5])))
# print(QuadraticCost.cost(np.array([[1, 2, 3]]), np.array([[3, 4, 5]])))
# print(QuadraticCost.cost(np.array([[1, 2, 3], [1, 2, 3]]), np.array([[3, 4, 5], [3, 4, 5]])))
# print(np.mean([QuadraticCost.cost(x, y) for x, y in
#                zip(np.array([[1, 2, 3], [1, 2, 3]]), np.array([[3, 4, 5], [3, 4, 5]]))]))

# xs = np.array([[1, 2, 3]])
# ys = np.array([[4, 5, 6]])
# for x, y in zip(xs, ys):
#     print('{}, {}, {}'.format(x, y, np.sum((x - y) ** 2)))
# print(QuadraticCost.cost(xs, ys))
# print(np.sum((xs - ys) ** 2)/2)
# print(np.linalg.norm(xs - ys))
