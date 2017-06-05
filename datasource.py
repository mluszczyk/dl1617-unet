import random
import math
from typing import Any

import PIL
import numpy
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from itertools import product
import io


def load_image(byte_data, augment, mode):
    img = Image.open(io.BytesIO(byte_data)).resize((512, 512))
    img = augment.transform(img, mode)
    array = numpy.asarray(img)
    return array


def load_image_list(dir, item_list, bytes_form_dir_and_name, augment):
    data = [
        load_image(bytes_form_dir_and_name(dir, item[0]), augment, item[1]) for item in item_list
    ]
    return numpy.asarray(data)


def get_image_list(path):
    items = list(os.listdir(os.path.join(path, "images")))
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


class NoCache:
    def load(self, paths):
        pass

    def get(self, path):
        with open(path, 'br') as f:
            return f.read()


class Subset:
    def __init__(self, path, names, cache, batch_size, transformer, augment):
        self.items = list(product(names, augment.trans_list()))
        self.cache = cache
        self.batch_size = batch_size
        self._transformer = transformer
        self._augment = augment
        self._path = path

    def _get_batch(self, item_list: [(str, Any)]):
        X = load_image_list(os.path.join(self._path, "images"), item_list, lambda x, y: self.cache.get(self._join(x, y)), self._augment)
        y = load_image_list(os.path.join(self._path, "heatmaps"), item_list, lambda x, y: self.cache.get(self._join(x, y)), self._augment)
        return self._transformer(X), self._transformer(y)

    def _get_batch_by_num(self, num):
        return self._get_batch(self.items[num * self.batch_size:(num + 1) * self.batch_size])

    def shuffle(self):
        random.shuffle(self.items)

    def batch_num(self) -> int:
        return int(math.ceil(len(self.items) / self.batch_size))

    def iter_batches(self):
        return (self._get_batch_by_num(num) for num in range(self.batch_num()))

    def _join(self, dir, name):
        return os.path.join(dir, name)


class DataSource:
    def __init__(self, path, train_num=None, *, test_num, batch_size, cache, transformer, augment):
        self.train_num = train_num
        self.test_num = test_num
        self._train_names = None
        self._test_names = None
        self.batch_size = batch_size
        self.data = {}
        self.cache = cache
        self._transformer = transformer
        self._augment = augment
        self._path = path

    def _join(self, dir, name):
        return os.path.join(dir, name)

    def load(self):
        name_list = get_image_list(self._path)
        if self.train_num is not None:
            name_list = name_list[:self.train_num + self.test_num]
        self.cache.load([self._join(d, n) for (d, n) in product(["data-bmp/images", "data-bmp/heatmaps"], name_list)])
        train_names, test_names = train_test_split(name_list, test_size=self.test_num)
        self.train = Subset(self._path, train_names, cache=self.cache,
                            batch_size=self.batch_size, transformer=self._transformer,
                            augment=self._augment)
        self.test = Subset(self._path, test_names, cache=self.cache,
                           batch_size=self.batch_size, transformer=self._transformer,
                           augment=NoAugment())


class NoAugment:
    def trans_list(self):
        return [None]

    def transform(self, image, mode):
        return image


class TransposeAugment:
    def trans_list(self):
        return [
            PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_90,
            PIL.Image.ROTATE_180, PIL.Image.ROTATE_270 or PIL.Image.TRANSPOSE]

    def transform(self, image, mode):
        return image.transpose(mode)
