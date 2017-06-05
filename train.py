"""Train the model on MNIST dataset."""

import os
import statistics

import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import DataSource, ImageCache, TransposeAugment, NoCache

from model import create_model, CHECKPOINT_FILE_NAME


class MnistTrainer:
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})
        return results[1:]

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 512, 512, 3], name='y')

        self.var_list, self.loss, self.accuracy, self.train_step, y_prob = create_model(self.x, self.y_target)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):
        def log(*messages):
            print(datetime.datetime.now().time(), *messages)

        self.create_model()

        saver = Saver()

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables

            if os.path.exists(CHECKPOINT_FILE_NAME + ".meta"):
                log("Restoring existing weights")
                saver.restore(self.sess, CHECKPOINT_FILE_NAME)
            else:
                log("Training a new model")

            log("Load data")

            report_n = 100
            test_n = 1000
            epoch_n = 10

            def transform(X):
                return X.astype(numpy.float32) / 255.

            data_source = DataSource(
                "data-bmp",
                train_num=10593 - 1024, test_num=1024, batch_size=10, cache=NoCache(), transformer=transform,
                augment=TransposeAugment())
            data_source.load()

            def test():
                test_losses = []
                for X_test, y_test in data_source.test.iter_batches():
                    loss = self.sess.run(self.loss, feed_dict={self.x: X_test,
                                                          self.y_target: y_test})
                    test_losses.append(loss)
                log("Test results", numpy.mean(test_losses))

            try:
                log("Start training")

                for epoch_idx in range(epoch_n):
                    log("Shuffle")
                    data_source.train.shuffle()

                    log("Epoch {}/{}".format(epoch_idx, epoch_n))

                    for batch_idx, (batch_X, batch_y) in enumerate(data_source.train.iter_batches()):

                        vloss = self.train_on_batch(batch_X, batch_y)

                        if (batch_idx + 1) % report_n == 0:
                            log('Batch {epoch_idx},{batch_idx}: loss {loss}'.format(
                                epoch_idx=epoch_idx,
                                batch_idx=batch_idx, loss=vloss)
                            )

                        if (batch_idx + 1) % test_n == 0:
                            test()
                            saver.save(self.sess, CHECKPOINT_FILE_NAME)

            except KeyboardInterrupt:
                log('Stopping training!')
                pass

            test()

if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()


