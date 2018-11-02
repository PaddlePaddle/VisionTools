""" a demo to show how to create a pipline of transformation on imagenet data
"""
import sys
import os
import time
import logging
from .. import operator
from .. import source
from .. import pipeline

g_settings = {'img_size': 224, 'normalize': False, 
        'part_id': 0, 'part_num': 1, 'cache': None,
        'workers': 16, 'decoded_bufsize': 10000, 
        'shuffle_size': 100, 'accelerate': True}

def train_image_mapper(img_size=None, normalize=None):
    """ a image mapper for training data
    """
    if img_size is None:
        img_size = g_settings['img_size']
    if normalize is None:
        normalize = g_settings['normalize']
    img_ops = [operator.DecodeImage()]
    img_ops += [operator.RotateImage(10, rand=True)]
    img_ops += [operator.RandCropImage(img_size)]
    img_ops += [operator.RandFlipImage()]
    img_ops += [operator.ToCHWImage()]

    if normalize:
        img_ops += [operator.NormalizeImage()]

    return img_ops


def test_image_mapper(img_size=None, normalize=None):
    """ a image mapper for testing data
    """
    if img_size is None:
        img_size = g_settings['img_size']
    if normalize is None:
        normalize = g_settings['normalize']

    img_ops = [operator.DecodeImage()]
    img_ops += [operator.ResizeImage(resize_short=img_size)]
    img_ops += [operator.CropImage(img_size)]
    img_ops += [operator.ToCHWImage()]
    if normalize:
        img_ops += [operator.NormalizeImage()]

    return img_ops


def make_reader(mode, uri, part_id=None, part_num=None, cache=None, pre_maps=None):
    infinite = False
    if mode == 'train':
        infinite = True

    if part_id is None:
        part_id = g_settings['part_id']

    if part_num is None:
        part_num = g_settings['part_num']

    if cache is None:
        cache = g_settings['cache']

    sc = source.load(uri=uri, part_id=part_id, part_num=part_num,
            cache=cache, infinite=infinite)

    p = pipeline.Pipeline()
    shuffle_size = g_settings['shuffle_size']
    if mode == 'train' and shuffle_size > 0:
        p.shuffle(shuffle_size)

    maps = []
    if pre_maps is not None:
        assert type(pre_maps) is list, 'invalid pre_maps param in build_pipeline'
        maps += pre_maps

    workers = g_settings['workers']
    decoded_bufsize = g_settings['decoded_bufsize']
    accelerate = g_settings['accelerate']

    img_ops = train_image_mapper() if mode == 'train' else test_image_mapper()
    for m in maps:
        p.map(m)
    p.map_ops(img_ops, workers=workers,
        decoded_bufsize=decoded_bufsize, accelerate=accelerate)
    return p.transform(sc.reader())


def train(uri, **kwargs):
    return make_reader('train', uri, **kwargs)


def val(uri, **kwargs):
    return make_reader('val', uri, **kwargs)
