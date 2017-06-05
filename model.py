import tensorflow as tf

from layers import Reshape, Conv, BatchNormalization, Relu, MaxPool, FullyConnected, AssertShape, ConvUp, SkipVar, Link, \
    Concat


def create_model(x, y_target):
    signal, var_list = inner_model(x)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=y_target))
    accuracy = tf.constant(1)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return var_list, loss, accuracy, train_step, signal


def inner_model(x):
    var512 = SkipVar()
    var256 = SkipVar()
    var128 = SkipVar()
    var64 = SkipVar()
    var32 = SkipVar()
    var16 = SkipVar()

    layers_list = [
        AssertShape((512, 512, 3)),
        Conv(8),
        AssertShape((512, 512, 8)),
        Relu(),

        AssertShape((512, 512, 8)),
        Conv(8),
        Relu(),
        Link(var512),
        BatchNormalization(),
        MaxPool(),

        AssertShape((256, 256, 8)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        Link(var256),
        MaxPool(),

        AssertShape((128, 128, 8)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        Link(var128),
        MaxPool(),

        AssertShape((64, 64, 8)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        Link(var64),
        MaxPool(),

        AssertShape((32, 32, 8)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        Link(var32),
        MaxPool(),

        AssertShape((16, 16, 8)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        Link(var16),
        MaxPool(),

        AssertShape((8, 8, 8)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        BatchNormalization(),
        ConvUp(8),
        Relu(),

        # AssertShape((16, 16, 8)),
        Concat(var16),
        AssertShape((16, 16, 16)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        BatchNormalization(),
        ConvUp(8),
        Relu(),

        # AssertShape((32, 32, 8)),
        Concat(var32),
        AssertShape((32, 32, 16)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        BatchNormalization(),
        ConvUp(8),
        Relu(),

        # AssertShape((64, 64, 8)),
        Concat(var64),
        AssertShape((64, 64, 16)),
        BatchNormalization(),
        Conv(8),
        Relu(),
        BatchNormalization(),
        ConvUp(8),
        Relu(),

        # AssertShape((128, 128, 8)),
        Concat(var128),
        AssertShape((128, 128, 16)),
        BatchNormalization(),
        Conv(8),
        AssertShape((128, 128, 8)),
        Relu(),
        BatchNormalization(),
        ConvUp(8),
        Relu(),

        # AssertShape((256, 256, 8)),
        Concat(var256),
        AssertShape((256, 256, 16)),
        BatchNormalization(),
        Conv(8),
        AssertShape((256, 256, 8)),
        Relu(),
        BatchNormalization(),
        ConvUp(8),
        Relu(),

        # AssertShape((512, 512, 8)),
        Concat(var512),
        AssertShape((512, 512, 16)),
        BatchNormalization(),
        Conv(8),
        AssertShape((512, 512, 8)),
        Relu(),

        # AssertShape((512, 512, 8)),
        Conv(3, filter_size=1),
        # AssertShape((512, 512, 3)),
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
