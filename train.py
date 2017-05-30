"""Train the model on MNIST dataset."""

import os

import numpy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.training.saver import Saver

import data
from model import create_model, CHECKPOINT_FILE_NAME


class MnistTrainer:
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})
        return results[1:]

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, 650, 650, 3], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 650, 650, 3], name='y')

        self.var_list, self.loss, self.accuracy, self.train_step, y_prob = create_model(self.x, self.y_target)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self):

        self.create_model()

        saver = Saver(var_list=self.var_list)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables

            if os.path.exists(CHECKPOINT_FILE_NAME + ".meta"):
                print("Restoring existing weights")
                saver.restore(self.sess, CHECKPOINT_FILE_NAME)
            else:
                print("Training a new model")

            print("Load data")
            X, y = data.load_data(1000)
            X = X.astype(numpy.float32) / 256.
            y = y.astype(numpy.float32) / 256.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)


            batches_n = 50
            mb_size = 10
            report_n = 1
            epoch_n = 10



            losses = []
            try:
                print("Start training")

                for epoch_idx in range(epoch_n):
                    print("Shuffle")
                    X_train, y_train = shuffle(X_train, y_train)

                    print("New epoch")

                    for batch_idx in range(batches_n):
                        batch_X, batch_y = X_train[mb_size * batch_idx:mb_size * (batch_idx + 1)], y_train[mb_size * batch_idx:mb_size * (batch_idx + 1)]

                        vloss = self.train_on_batch(batch_X, batch_y)

                        losses.append(vloss)

                        if batch_idx % report_n == 0:
                            print('Batch {epoch_idx},{batch_idx}: mean_loss {mean_loss}'.format(
                                epoch_idx=epoch_idx,
                                batch_idx=batch_idx, mean_loss=np.mean(losses[-200:], axis=0))
                            )
                            print('Test results', self.sess.run([self.loss, self.accuracy],
                                                                feed_dict={self.x: X_test,
                                                                           self.y_target: y_test}))

                            saver.save(self.sess, CHECKPOINT_FILE_NAME)
                    print("End of epoch")
 
            except KeyboardInterrupt:
                print('Stopping training!')
                pass
 
            # Test trained model
            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: X_test,
                                                self.y_target: y_test}))


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()


