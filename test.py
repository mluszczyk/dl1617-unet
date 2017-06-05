"""Test the model on the dataset."""

import os

import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import DataSource, ImageCache, TransposeAugment, NoCache

from model import create_model, CHECKPOINT_FILE_NAME


class Tester:
    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 512, 512, 3], name='y')

        self.var_list, self.loss, self.accuracy, self.train_step, y_prob = create_model(self.x, self.y_target)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def test(self):
        def log(*messages):
            print(datetime.datetime.now().time(), *messages)

        self.create_model()

        saver = Saver()

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables

            log("Restoring existing weights")
            saver.restore(self.sess, CHECKPOINT_FILE_NAME)

            log("Load data")

            def transform(X):
                return X.astype(numpy.float32) / 255.

            data_source = DataSource(
                "data",
                train_num=10593 - 1024, test_num=1024, batch_size=10, cache=ImageCache(), transformer=transform,
                augment=TransposeAugment())
            data_source.load()

            def test(func=self.loss):
                test_losses = []
                for X_test, y_test in data_source.test.iter_batches():
                    loss = self.sess.run(func, feed_dict={self.x: X_test,
                                                          self.y_target: y_test})
                    test_losses.append(loss)
                log("Test results", numpy.mean(test_losses))

            test()

if __name__ == '__main__':
    trainer = Tester()
    trainer.test()


