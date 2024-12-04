import activation_funs
import cost_funs
import mnist_loader
import tools
import network_basic_v2 as nt

# trains, validates, tests = mnist_loader.load_data()
# tools.show_pic(trains[0][0], trains[1][0])

trains, validates, tests = mnist_loader.load_data_wrapper()

# Epoch 30/30 end, cost 0.041856, took 20.91s, accuracy 9400/10000
# vs
# Epoch 30/30 end, cost 0.020091, lr:8.40 took 14.51s, accuracy 9607/10000
# net = nt.NetworkBasic(layer_sizes=[784, 40, 10], cost_fun=cost_funs.QuadraticCost)
# Epoch 30/30 end, cost 0.018822, lr:15.00 took 21.57s, accuracy 9656/10000
# net = nt.NetworkBasic(layer_sizes=[784, 40, 10], cost_fun=cost_funs.QuadraticCost,
#                       output_atv_fun=activation_funs.Sigmoid)
# net.info()
# net.train_degrade(trains, 3, 30, 20, tests, True)

# Epoch 30/30 end, cost 0.041353, took 22.85s, accuracy 9382/10000
# Epoch 28/30 end, cost 0.083697, took 23.80s, accuracy 8572/10000
# 学习缓慢 周期不够导致拟合不充分
# net = nt.NetworkBasic(layer_sizes=[784, 50, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests)

# 动态调整 学习率，加快学习
# Epoch 30/30 end, cost 0.023280, lr:20.92 took 21.93s, accuracy 9505/10000
# 代价还在下降，但是精度已经开始震荡，应该是过拟合了
# net = nt.NetworkBasic(layer_sizes=[784, 40, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests, True)

# Epoch 24/30 end, cost 0.123668, lr:64.59 took 33.02s, accuracy 7822/10000
# 明显是过拟合了
# net = nt.NetworkBasic(layer_sizes=[784, 100, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests, True)

# Epoch 12/30 end, cost 0.040927, lr:32.94 took 27.23s, accuracy 9434/10000
# 学习的很快，但是后期震荡
# net = nt.NetworkBasic(layer_sizes=[784, 50, 25, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests, True)

# Epoch 21/30 end, cost 0.028269, lr:1.90 took 24.85s, accuracy 9419/10000
# net = nt.NetworkBasic(layer_sizes=[784, 40, 20, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests, True)


# Epoch 30/30 end, cost 0.039751, lr:10.00 took 17.52s, accuracy 9393/10000
# net = nt.NetworkBasic(layer_sizes=[784, 20, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests, True)

# Epoch 30/30 end, cost 0.017528, lr:2.40 took 35.51s, accuracy 9455/10000
# 更多的隐藏层，并不能带来更多的准确率
# net = nt.NetworkBasic(layer_sizes=[784, 100, 20, 10], cost_fun=cost_funs.QuadraticCost)
# net.train_degrade(trains, 3, 30, 100, tests, True)

"""交叉熵代价函数"""
# Epoch 35/50 end, cost 0.008742, lr:0.50 mini_num:10 took 17.73s, accuracy 9682/10000
# net = nt.NetworkBasic(layer_sizes=[784, 100, 10], cost_fun=cost_funs.CrossEntropyCost,
#                       output_atv_fun=activation_funs.Sigmoid)
# net.info()
# net.train_degrade(trains, 0.5, 50, 10, tests)

"""标准化，效果很明显"""
# Epoch 32/50 end, cost 0.006109, lr:0.50 mini_num:10 took 19.04s, accuracy 9773/10000
# net = nt.NetworkBasic(layer_sizes=[784, 40, 10], cost_fun=cost_funs.CrossEntropyCost,
#                       output_atv_fun=activation_funs.Sigmoid, std_w=True)
# net.info()
# net.train_degrade(trains, 0.5, 30, 10, tests)
# net.info()

"""L2规范化，权重衰减，好像没啥明显效果"""
# Epoch 54/60 end, cost 0.133630, lr:0.50 mini_num:10 took 18.84s, accuracy 9782/10000
# net = nt.NetworkBasic(layer_sizes=[784, 40, 10], cost_fun=cost_funs.CrossEntropyCost,
#                       output_atv_fun=activation_funs.Sigmoid, std_w=True, lbd=5)
# net.info()
# net.train_degrade(trains, 0.5, 30, 10, tests)
# net.info()

#
net = nt.NetworkBasic(layer_sizes=[784, 100, 30, 10], cost_fun=cost_funs.CrossEntropyCost,
                      output_atv_fun=activation_funs.Sigmoid, std_w=True, lbd=5)
net.info()
net.train_degrade(trains, 0.5, 60, 10, tests)
net.info()
