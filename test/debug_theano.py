import os

os.environ[
    "THEANO_FLAGS"] = "device=cpu,floatX=float32,exception_verbosity=high,traceback.limit=10,compute_test_value=warn"

import theano
import theano.tensor as tt
import numpy as np

"""堆栈异常 增加,exception_verbosity=high,traceback.limit=10"""
# x = tt.vector()
# y = tt.vector()
# z = x + x
# z = z + y
# f = theano.function([x, y], z)
# f(
#     np.array([1, 2], dtype=theano.config.floatX), np.array([3, 4, 5], dtype=theano.config.floatX))

"""使用test value 增加标记 ,compute_test_value=warn"""
# x = tt.vector()
# y = tt.vector()
# x.tag.test_value = np.array([1, 2], dtype=theano.config.floatX)
# y.tag.test_value = np.array([3, 4, 5], dtype=theano.config.floatX)
# z = x + x
# print(z.tag.test_value)
# z = z + y
# print(z.tag.test_value)
# f = theano.function([x, y], z)


"""使用theano.printing"""
# x = tt.vector()
# y = tt.vector()
# z = x + x
# z = theano.printing.Print('z1')(z)
# z = z + y
# z = theano.printing.Print('z2')(z)
# f = theano.function([x, y], z)
# f(np.array([1, 2], dtype=theano.config.floatX), np.array([1, 2], dtype=theano.config.floatX))

