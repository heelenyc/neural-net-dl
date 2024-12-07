import configuration
import network_cnn as nt
from network_cnn import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


mini_batch_size = 10
training_data, validation_data, test_data = nt.load_data_shared()

# 之前同结构模型
net = nt.NetworkCnn([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

# net = nt.NetworkCnn([
# nt.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                  filter_shape=(20, 1, 5, 5),
#                  poolsize=(2, 2)), FullyConnectedLayer(n_in=20*12*12, n_out=100),
# SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size) >>> net.SGD(training_data, 60, mini_batch_size, 0.1,
# validation_data, test_data)
