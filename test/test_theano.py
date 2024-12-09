import numpy
import numpy as np
import theano
import theano.tensor as tt
from matplotlib import pyplot as plt
from theano import function, pp
from theano.tensor import shared_randomstreams

import seaborn as sns

"""
# 定义零维数组（即标量）
# 利用T.dscalar('a')可实现变量a代替给定的浮点标量
x = T.dscalar('x')
y = T.dscalar('y')

# 定义z为x与y的和。可通过print(theano.pp(z))来查询z的内容
z = x + y

# function([input],output)用来定义输入输出
f = function([x, y], z)
"""

# theano.config.aaa = 'float321'
# class MyConfig:
#     def __init__(self):
#         self.name = 'myconfig'
#
#     def print_config(self):
#         print(self.name)
#
#
# cfg = MyConfig()
# cfg.name = 'myconfig1'
# cfg.aaa = 'aaa'
# cfg.print_config()

# x0 = tt.dscalar('x0')
# x1 = tt.dscalar('x1')
# aver = (x0 + x1) / 2
# f = theano.function([x0, x1], aver)
# y = f(1.4, 2.5)
# print(y)
# print(type(f))
# print(type(x0))

# x = tt.fmatrix('x')
# y = tt.fmatrix('y')
# z = tt.fmatrix('z')
# su = x + y + z
# f = theano.function([x, y, z], su)
# a = f([[1, 2, 3]], [[4, 5, 6]], [[4, 5, 6]])
# print(a)
# print(type(a))

# a, b = tt.dmatrices('a', 'b')
# diff = a - b
# diff_abs = abs(diff)
# diff_squared = diff ** 2
# f = theano.function([a, b], [diff, diff_abs, diff_squared])
# f_return = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
# print(f_return)

# # 使变量为共享变量
# state = theano.shared(0)
# # 定义一个标量('name')
# inc = tt.iscalar('inc')
# # state = state + inc / updates=[old_w,new_w],当函数被调用的时候,这个会用new_w替换old_w
# f = theano.function([inc], state, updates=[(state, state + inc)])
# print('state:', state.get_value())
#
# # 传一个5进去,state被state+inc更新后　state＝5
# f(5)
# print('f(5):', state.get_value())
#
# # 传一个100进去,state被state+inc更新后　state＝105
# f(100)
# print('f(100):', state.get_value())
#
# # 利用set_value将state的设置为-1000
# state.set_value(-1000)
# print('state.set_value(-1000):', state.get_value())
# # 传一个-1000进去　state＝-1000
# f(-1000)
# print('f(-1000):', state.get_value())
# # 传一个800进去,state被state+inc更新后　state＝800-1000＝-200
# f(800)
# print('f(800):', state.get_value())


# # 定义一个int类型的变量
# inc = tt.iscalar('inc')
# # 返回共享变量变量,使用“value”的副本或引用初始化
# # 该函数迭代构造函数以找到合适的共享变量子类,合适的是接受给定值的第一个构造函数
# # shared(value, name=None, strict=False, allow_downcast=None, **kwargs)
# state = theano.shared(0)
# fn_of_state = state * 2 + inc
# # 定义一个标量
# # scalar(name=None, dtype=None) /int64
# foo = tt.scalar(dtype=state.dtype)
#
# print("foo is :", foo)
# f = theano.function([inc, foo], fn_of_state, givens={state: foo})
# print(f(5, 2))
# print('state :', state.get_value())

# print(np.prod((2, 2)))

# print("{}",theano.config.device)

# # 创建一个共享变量
# np_array = np.ones(2, dtype='float32')
# s_default = theano.shared(np_array, borrow=True)
# print(s_default)
# print(s_default.get_value())
# s_default.set_value([5., 5.])
# print(s_default.get_value())
# print(np_array)
#
# # 使用borrow参数
# np_array = np.ones(2, dtype='float32')
#
# s_default = theano.shared(np_array)
# s_true = theano.shared(np_array, borrow=True)
# s_false = theano.shared(np_array, borrow=False)
#
# np_array += 1
#
# print('s_default:', s_default.get_value())  # 输出 [6. 6.]
# print('s_true:', s_true.get_value())  # 输出 [1. 1.]
# print('s_false:', s_false.get_value())  # 输出 [1. 1.]

# x = np.array([1, 2])
# print(x)
# x_shared = theano.shared(x, name='x_shared')
# print(x_shared.get_value())
#
# x_as_int = tt.cast(x_shared, 'int32')
# print(x_as_int)

# x = tt.dscalar('x')
# y = x ** 2
# gygx = tt.grad(y, x)
# pp(gygx)
# grad = theano.function([x], gygx)
#
# print(grad(2.344))
#
# x = tt.dmatrix('x')
# s = tt.sum(1 / (1 + tt.exp(-x)))
# gsgx = tt.grad(s, x)
# dlogistic = theano.function([x], gsgx)
#
# print(dlogistic([[2, 3], [5, 7]]))

# seed = np.random.RandomState(0).randint(999999)
# r = shared_randomstreams.RandomStreams(seed)
# print(r)

# 二项分布
x = tt.matrix('x')
y = tt.fscalars('y')
theano.config.floatX = 'float32'

a = np.random.randn(10, 3)
print('a: {}'.format(a))

srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
mask = srng.binomial(n=1, p=1 - y, size=x.shape)
d = x * tt.cast(mask, theano.config.floatX)

fun = theano.function([x, y], d)

print(fun(a, 0.2))

# sns.distplot(np.random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
#
# plt.show()

# print(np.random.binomial(n=1, p=0.5, size=(50, 10)))
