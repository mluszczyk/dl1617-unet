"""Test the model on the dataset."""

import sys
import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import load_single_image, save_image, NoAugment, ImageCache, DataSource, save_images

from model import CHECKPOINT_FILE_NAME, query_model


class Tester:
    def test(self, dst_dir):
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

            data_source = DataSource(
                "data",
                train_num=10593 - 1024, test_num=1024, batch_size=4, cache=ImageCache(), transformer=transform,
                augment=NoAugment())
            data_source.load()

            for ((X, y_unused), items) in data_source.train.iter_batches_with_names():
                names = [i[0] for i in items]
                print("Batch")
                y = sess.run(y_pred, feed_dict={x: X})
                save_images(y, dst_dir, names)


if __name__ == '__main__':
    trainer = Tester()
    trainer.test(sys.argv[1])


