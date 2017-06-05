"""Test the model on the dataset."""

import os

import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import DataSource, ImageCache, TransposeAugment, NoCache

from model import CHECKPOINT_FILE_NAME, InnerModel, loss_func


class Tester:
    def test(self):
        def log(*messages):
            print(datetime.datetime.now().time(), *messages)

        x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='x')
        y_target = tf.placeholder(tf.float32, [None, 512, 512, 3], name='y')

        model = InnerModel()
        model.register_variables(x)

        y_pred_trans = model.apply(tf.transpose(x, [0, 2, 1, 3]))
        y_pred_aug = tf.transpose(y_pred_trans, [0, 2, 1, 3])
        y_pred_orig = model.apply(x)

        y_pred = tf.reduce_mean([y_pred_aug, y_pred_orig], axis=[0])

        saver = Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()  # initialize variables

            log("Restoring existing weights")
            saver.restore(sess, CHECKPOINT_FILE_NAME)

            log("Load data")

            def transform(X):
                return X.astype(numpy.float32) / 255.

            data_source = DataSource(
                "data",
                train_num=10593 - 1024, test_num=1024, batch_size=10, cache=ImageCache(), transformer=transform,
                augment=TransposeAugment())
            data_source.load()

            def test(func):
                test_losses = []
                for X_test, y_test in data_source.test.iter_batches():
                    loss = sess.run(func, feed_dict={x: X_test,
                                                     y_target: y_test})
                    test_losses.append(loss)
                log("Test results", numpy.mean(test_losses))

            def test_augment():
                loss = loss_func(y_pred, y_target)
                return test(loss)

            test_augment()

if __name__ == '__main__':
    trainer = Tester()
    trainer.test()


