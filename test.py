"""Test the model on the dataset."""

import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import DataSource, ImageCache, TransposeAugment, NoCache

from model import CHECKPOINT_FILE_NAME, InnerModel, loss_func, create_test_model


class Tester:
    def test(self):
        def log(*messages):
            print(datetime.datetime.now().time(), *messages)

        x, y_pred, y_target = create_test_model()

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


