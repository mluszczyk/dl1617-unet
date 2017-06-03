import random
import math

import numpy
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from itertools import product
import io


def load_image(byte_data):
    img = Image.open(io.BytesIO(byte_data))
    array = numpy.asarray(img)
    return array


def load_image_list(dir, name_list, bytes_form_dir_and_name):
    data = [load_image(bytes_form_dir_and_name(dir, p)) for p in name_list]
    return numpy.asarray(data)


def get_image_list():
    items = list(os.listdir("data/images/"))
    return items


class ImageCache:
    def __init__(self):
        self.data = {}

    def load(self, paths):
        for path in paths:
            with open(path, 'br') as f:
                self.data[path] = f.read()

    def get(self, path):
        return self.data[path]


class DataSource:
    def __init__(self, train_num=None, *, test_num, batch_size, cache, transformer):
        self.train_num = train_num
        self.test_num = test_num
        self._train_names = None
        self._test_names = None
        self.batch_size = batch_size
        self.data = {}
        self.cache = cache
        self._transformer = transformer

    def _join(self, dir, name):
        return os.path.join(dir, name)

    def load(self):
        name_list = get_image_list()
        if self.train_num is not None:
            name_list = name_list[:self.train_num + self.test_num]
        self.cache.load([self._join(d, n) for (d, n) in product(["data/images", "data/heatmaps"], name_list)])
        self._train_names, self._test_names = train_test_split(name_list, test_size=self.test_num)

    def _get_batch(self, name_list):
        X = load_image_list("data/images", name_list, lambda x, y: self.cache.get(self._join(x, y)))
        y = load_image_list("data/heatmaps", name_list, lambda x, y: self.cache.get(self._join(x, y)))
        return self._transformer(X), self._transformer(y)

    def get_test(self):
        return self._get_batch(self._test_names)

    def get_train(self, num):
        names = self._train_names[num * self.batch_size:(num + 1) * self.batch_size]
        assert names
        return self._get_batch(names)

    def shuffle_train(self):
        random.shuffle(self._train_names)

    def train_batch_num(self) -> int:
        return int(math.ceil(self.train_num / self.batch_size))

    def iter_train_batches(self):
        return (self.get_train(num) for num in range(self.train_batch_num()))
