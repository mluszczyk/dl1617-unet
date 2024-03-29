"""Train the model on the dataset."""

import os
import statistics

import datetime
import numpy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from datasource import DataSource, ImageCache, TransposeAugment, NoCache

from model import create_model, CHECKPOINT_FILE_NAME


class Trainer:
    def train_on_batch(self, sess, batch_xs, batch_ys):
        results = sess.run([self.train_step, self.loss],
                           feed_dict={self.x: batch_xs, self.y_target: batch_ys})
        return results[1:]

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 650, 650, 3], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 650, 650, 3], name='y')

        self.var_list, self.loss, self.train_step, y_prob = create_model(self.x, self.y_target)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):
        def log(*messages):
            print(datetime.datetime.now().time(), *messages)

        self.create_model()

        saver = Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()  # initialize variables

            if os.path.exists(CHECKPOINT_FILE_NAME + ".meta"):
                log("Restoring existing weights")
                saver.restore(sess, CHECKPOINT_FILE_NAME)
            else:
                log("Training a new model")

            log("Load data")

            report_n = 100
            test_n = 1000
            epoch_n = 10

            def transform(X):
                return X.astype(numpy.float32) / 255.

            data_source = DataSource(
                "data",
                train_num=10593 - 1024, test_num=1024, batch_size=4, cache=ImageCache(), transformer=transform,
                augment=TransposeAugment())
            data_source.load()

            def test(func=self.loss):
                test_losses = []
                for X_test, y_test in data_source.test.iter_batches():
                    loss = sess.run(func, feed_dict={self.x: X_test,
                                                          self.y_target: y_test})
                    test_losses.append(loss)
                log("Test results", numpy.mean(test_losses))

            try:
                log("Start training")

                for epoch_idx in range(epoch_n):
                    log("Shuffle")
                    data_source.train.shuffle()

                    log("Epoch {}/{}".format(epoch_idx, epoch_n))

                    losses = []
                    for batch_idx, (batch_X, batch_y) in enumerate(data_source.train.iter_batches()):

                        vloss = self.train_on_batch(sess, batch_X, batch_y)[0]
                        losses.append(vloss)

                        if (batch_idx + 1) % report_n == 0:
                            log('Batch {epoch_idx},{batch_idx}: loss {loss}'.format(
                                epoch_idx=epoch_idx,
                                batch_idx=batch_idx, loss=numpy.mean(losses[-report_n:]))
                            )

                        if (batch_idx + 1) % test_n == 0:
                            test()
                            saver.save(sess, CHECKPOINT_FILE_NAME)

            except KeyboardInterrupt:
                log('Stopping training!')
                pass

            test()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()


