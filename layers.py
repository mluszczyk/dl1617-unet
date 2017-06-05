import tensorflow as tf


def weight_variable(shape, *, trainable, stddev=0.1, mean=0.0):
    initializer = tf.truncated_normal(shape, stddev=stddev, mean=mean)
    return tf.Variable(initializer, name='weight', trainable=trainable)


def bias_variable(shape, *, trainable, bias=0.1):
    initializer = tf.constant(bias, shape=shape)
    return tf.Variable(initializer, name='bias', trainable=trainable)


class Relu:
    def contribute(self, signal, idx, trainable, save_variable):
        return tf.nn.relu(signal)


class Reshape:
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def contribute(self, signal, idx, trainable, save_variable):
        return tf.reshape(signal, self.output_shape)


class Conv:
    def __init__(self, output_channels: int, filter_size: int=3):
        self.output_channels = output_channels
        self._filter_size = filter_size
        self.vars = {}

    def register_vars(self, signal, idx, trainable, save_variable):
        assert len(signal.get_shape()) == 4
        input_channels = int(signal.get_shape()[3])

        with tf.variable_scope('conv_' + str(idx + 1)):
            self.vars['W_conv'] = weight_variable(
                [self._filter_size, self._filter_size, input_channels, self.output_channels],
                trainable=trainable)
            save_variable(self.vars['W_conv'])
            self.vars['b_conv'] = bias_variable([self.output_channels], trainable=trainable)
            save_variable(self.vars['b_conv'])

    def contribute(self, signal, idx, trainable, save_variable):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return conv2d(signal, self.vars['W_conv']) + self.vars['b_conv']


class ConvUp:
    def __init__(self, output_channels: int):
        self.output_channels = output_channels
        self.vars = {}

    def register_vars(self, signal, idx, trainable, save_variable):
        assert len(signal.get_shape()) == 4
        input_channels = int(signal.get_shape()[3])

        with tf.variable_scope('conv_' + str(idx + 1)):
            self.vars['W_conv'] = weight_variable([5, 5, input_channels, self.output_channels], trainable=trainable)
            save_variable(self.vars['W_conv'])
            self.vars['b_conv'] = bias_variable([self.output_channels], trainable=trainable)
            save_variable(self.vars['b_conv'])

    def contribute(self, signal, idx, trainable, save_variable):
        output_shape = (tf.shape(signal)[0], tf.shape(signal)[1] * 2, tf.shape(signal)[2] * 2, self.output_channels)

        def conv2d(x, W):
            return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, padding='SAME', strides=[1, 2, 2, 1])

        return conv2d(signal, self.vars['W_conv']) + self.vars['b_conv']


class MaxPool:
    def contribute(self, signal, idx, trainable, save_variable):
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
        return max_pool_2x2(signal)


class BatchNormalization:
    def __init__(self):
        self.vars = {}

    def register_vars(self, signal, idx, trainable, save_variable):
        input_shape = signal.get_shape()
        assert len(input_shape) == 4

        with tf.variable_scope('batch_norm_' + str(idx + 1)):
            gamma = weight_variable([int(input_shape[-1])], trainable=trainable, mean=1.0)
            save_variable(gamma)
            self.vars['gamma'] = gamma
            beta = bias_variable([int(input_shape[-1])], trainable=trainable)
            save_variable(beta)
            self.vars['beta'] = beta

    def contribute(self, signal, idx, trainable, save_variable):
        input_shape = signal.get_shape()
        mean = tf.reduce_mean(signal, axis=[0, 1, 2])
        assert len(mean.get_shape()) == 1
        stdvarsq = tf.reduce_mean((signal - mean) ** 2, axis=[0, 1, 2])
        assert len(stdvarsq.get_shape()) == 1
        eps = 1e-5
        normalized = ((signal - mean) / tf.sqrt(stdvarsq + eps))
        assert (str(normalized.get_shape()) == str(input_shape))
        return tf.multiply(self.vars['gamma'], normalized) + self.vars['beta']


class FullyConnected:
    def __init__(self, neuron_num):
        self.neuron_num = neuron_num
        self.vars = {}

    def register_vars(self, signal, idx, trainable, save_variable):
        cur_num_neurons = int(signal.get_shape()[1])
        stddev = 0.1

        with tf.variable_scope('fc_' + str(idx + 1)):
            W_fc = weight_variable([cur_num_neurons, self.neuron_num], stddev=stddev, trainable=trainable)
            save_variable(W_fc)
            self.vars['W_fc'] = W_fc
            b_fc = bias_variable([self.neuron_num], bias=0.1, trainable=trainable)
            save_variable(b_fc)
            self.vars['b_fc'] = b_fc

    def contribute(self, signal, idx, trainable, save_variable):
        signal = tf.matmul(signal, self.vars['W_fc']) + self.vars['b_fc']
        return signal


class AssertShape:
    def __init__(self, shape):
        self.shape = shape

    def contribute(self, signal, idx, trainable, save_variable):
        shape = signal.get_shape()
        assert tuple(shape.as_list()[1:]) == self.shape
        return signal


class Link:
    def __init__(self, var):
        self.var = var

    def contribute(self, signal, idx, trainable, save_variable):
        self.var.signal = signal
        return signal


class Concat:
    def __init__(self, var):
        self.var = var

    def contribute(self, signal, idx, trainable, save_variable):
        return tf.concat([signal, self.var.signal], 3)


class SkipVar:
    def __init__(self):
        self.signal = None
