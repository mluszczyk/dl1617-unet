import numpy
import os
from PIL import Image


def load_image(path):
    img = Image.open(path)
    array = numpy.asarray(img)
    return array


def load_image_list(dir, name_list):
    data = [load_image(os.path.join(dir, p)) for p in name_list]
    return numpy.asarray(data)


def get_image_list():
    items = list(os.listdir("data/images/"))
    return items


def load_data(num=None):
    name_list = get_image_list()
    if num is not None:
        name_list = name_list[:num]
    X = load_image_list("data/images", name_list)
    y = load_image_list("data/heatmaps", name_list)
    return X, y
