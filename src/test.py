import numpy as np

import activation_funs

# x = np.array([[1],[1]])
# y = np.array([[1],[1],[1]]).T
# print(x)
# print(y)
# print(np.multiply(x,y)) # hadamard 积；对应位置的乘积
# print(x*y)  # multiply 一样
# print(y*x)  # multiply 一样
#
# print(np.matmul(x,y)) # 矩阵乘法，对应的元素量要一样；

# print(np.dot(x,y))  # 向量点乘，标量，维数相等，对应位置乘积后相加，会退化成matmul 和multiply

""" 
hadamard  multiply  *  操作对应是n维数组，不仅仅是矩阵，相同结构的对应的位置相乘；
"""
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(a.T)
# print(np.multiply(a, b))
# print(np.multiply(a, 2))  # 进行了广播

"""
matmul  矩阵乘积
np.matmul(a, b) 的操作对象 a, b 只能是多维数组 (array-like)，不能是标量 (scalar)。
而如果 a 或 b 中有一个的维度是1，即如果是向量，则会进行这样的转换：

若 a 是向量，则把 a 的 dim 前面插入一个1，把它变成一个行向量（一行的矩阵）
若 b 是向量，则把 b 的 dim 后面插入一个1，把它变成一个列向量（一列的矩阵）

Numpy 实现了运算符重载，使用 @ 可以代替 np.matmul()
"""
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[1, 1], [1, 1]])
# print(np.matmul(A, B))

# 发生了维度转换，这和后面要讲的 np.dot 等价
# a = np.array([1, 2])
# b = np.array([3, 4])
# print(np.matmul(a, b))

"""
矩阵点乘，就是矩阵各个对应元素相乘，要求矩阵必须维数相等，即MxN维矩阵乘以MxN维矩阵 。
"""
# a = np.array([1, 2])
# b = np.array([3, 4])
# print(np.dot(a, b))  # 11
# print(np.multiply(a, b))  # [3 8]

# for i in range(1, 2):
#     print(i)

"""向量的模"""

#
# def quadratic_cost_function(h, y):
#     # 计算二次代价函数
#     J = (h - y) ** 2 / (2 * len(y))
#     return J.sum()
#
#
# a = np.array([[1], [2]])
# b = np.array([[3], [4]])
# m = [(a - y) ** 2 / 2 for a, y in zip(a, b)]
# print(m)
# print(np.sum(m))  #

# print([1, 2, 3][1:-1])

# print(1/activation_funs.Sigmoid.active(-100))
# print(1/activation_funs.Sigmoid.active(-100))
# for i in range(-1, -3,-1):
#     print(i)

# list = []
# narray = np.zeros((1,2))
# list.append(narray)
# narray[0][0] = 1
# print(list)

list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(list[1:11])
