import tensorflow as tf

from layers import Reshape, Conv, BatchNormalization, Relu, MaxPool, FullyConnected, AssertShape, ConvUp


def create_model(x, y_target):
    signal, var_list = inner_model(x)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=y_target))
    accuracy = tf.constant(1)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return var_list, loss, accuracy, train_step, signal


def inner_model(x):
    layers_list = [
        AssertShape((650, 650, 3)),
        Conv(4),
        AssertShape((650, 650, 4)),
        Relu(),
        MaxPool(),

        AssertShape((325, 325, 4)),
        BatchNormalization(),
        Conv(4),
        Relu(),
        BatchNormalization(),
        Conv(4),
        Relu(),

        ConvUp(4),
        AssertShape((650, 650, 4)),
        BatchNormalization(),
        AssertShape((650, 650, 4)),
        Conv(3),
        AssertShape((650, 650, 3)),
    ]
    variable_saver = VariableSaver()
    signal = x
    print('shape', signal.get_shape())
    for idx, layer in enumerate(layers_list):
        signal = layer.contribute(signal, idx, save_variable=variable_saver.save_variable, trainable=True)
        print('shape', signal.get_shape())
    return signal, variable_saver.var_list


class VariableSaver:
    def __init__(self):
        self.var_list = []

    def save_variable(self, var):
        self.var_list.append(var)


CHECKPOINT_FILE_NAME = "checkpoint-mnist"