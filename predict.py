"""Test the model on the dataset."""

import sys
import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import load_single_image, save_image

from model import CHECKPOINT_FILE_NAME, query_model


class Tester:
    def test(self, name, dst_name):
        def log(*messages):
            print(datetime.datetime.now().time(), *messages)

        x, y_pred = query_model()

        saver = Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()  # initialize variables

            log("Restoring existing weights")
            saver.restore(sess, CHECKPOINT_FILE_NAME)

            log("Load data")

            def transform(X):
                return X.astype(numpy.float32) / 255.

            X = load_single_image(name)
            y = sess.run(y_pred, feed_dict={x: X})
            save_image(y, dst_name)


if __name__ == '__main__':
    trainer = Tester()
    trainer.test(sys.argv[1], sys.argv[2])


