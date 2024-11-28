import cost_funs
import mnist_loader
import tools
import network_basic as nt

# trains, validates, tests = mnist_loader.load_data()
# tools.show_pic(trains[0][0], trains[1][0])

trains, validates, tests = mnist_loader.load_data_wrapper()
# tools.show_pic(trains[0][0], mnist_loader.re_vectorized_result(trains[0][1]))

# Epoch 30/30 end, cost 0.044379, took 21.85s, accuracy 97/100
net = nt.NetworkBasic(layer_sizes=[784, 40, 10], cost_fun=cost_funs.QuadraticCost)


net.info()
net.train_degrade(trains, 3, 30, 100, tests[0:100])
net.info()
