import pickle
import gzip

import numpy as np
import theano
import theano.tensor as tt
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

import logger_factory as lf

logger = lf.logger_factory.create_logger('network_cnn', log_format='%(asctime)s %(message)s')


# Activation functions for neurons
def linear(z): return


def ReLU(z): return tt.maximum(0.0, z)


# Constants
logger.info("Theano use device : {}.  If this is not desired, then modify the env config in configuration.py".format(
    theano.config.device))


# Load the MNIST data
def load_data_shared(filename="../../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, tt.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]


class NetworkCnn:

    def __init__(self, layers, mini_batch_size):
        """takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tt.matrix("x")
        self.y = tt.ivector("y")

        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """
        Train the network using mini-batch stochastic gradient descent.
        """

        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of mini batches for training, validation and testing
        num_training_batches = int(size(training_data) / mini_batch_size)
        num_validation_batches = int(size(validation_data) / mini_batch_size)
        num_test_batches = int(size(test_data) / mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])  # l2 规范化项
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = tt.grad(cost, self.params)  # 这么简单？
        updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = tt.lscalar()  # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                    training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        # TODO:
        test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = -1
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index  # 这个迭代对应的是一个mini样本

                cost_ij = train_mb(minibatch_index)
                if iteration % 1000 == 0:
                    logger.info(
                        "Training mini-batch number {}/{}*{}  cost:{:.6f}".format(iteration, num_training_batches,
                                                                                  epoch + 1, cost_ij))
                if (iteration + 1) % num_training_batches == 0:
                    # 每个minibatch里的开始时，对应验证集上的精度
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])

                    logger.info("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))

                    if validation_accuracy >= best_validation_accuracy:
                        # 历史最好验证集精度
                        logger.info("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                            logger.info('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))

        logger.info("Finished training network.")
        logger.info("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy,
                                                                                           best_iteration))
        logger.info("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class ConvPoolLayer:

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2), activation_fn=sigmoid):

        self.output_dropout = None
        self.output = None
        self.inpt = None

        self.filter_shape = filter_shape  # (num_filters, num_input_feature_map, filter_height, filter_width)
        self.image_shape = image_shape  # (mini_batch_size, num_input_feature_map, image_height, image_width)
        self.pool_size = pool_size
        self.activation_fn = activation_fn

        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                       dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)), dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        pooled_out = pool_2d(input=conv_out, ws=self.pool_size, ignore_border=True)
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output  # no dropout in the convolutional layers


class FullyConnectedLayer:

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.output_dropout = None
        self.inpt_dropout = None
        self.y_out = None
        self.output = None
        self.inpt = None

        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1 - self.p_dropout) * tt.dot(self.inpt, self.w) + self.b)
        self.y_out = tt.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(tt.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return tt.mean(tt.eq(y, self.y_out))


class SoftmaxLayer:

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.output_dropout = None
        self.inpt_dropout = None
        self.y_out = None
        self.output = None
        self.inpt = None

        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * tt.dot(self.inpt, self.w) + self.b)
        self.y_out = tt.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(tt.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        """Return the log-likelihood cost."""
        return -tt.mean(tt.log(self.output_dropout)[tt.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return tt.mean(tt.eq(y, self.y_out))


# Miscellanea
def size(data):
    """Return the size of the dataset `data`."""
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * tt.cast(mask, theano.config.floatX)
