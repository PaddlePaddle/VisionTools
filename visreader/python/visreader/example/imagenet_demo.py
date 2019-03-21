"""
# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function
#    a demo to show how to create a pipline of transformation on imagenet data
#
"""
import sys
import os
import time
import logging
import copy
from .. import operators as ops
from .. import source
from ..pipeline import Dataset

g_settings = {
    'img_size': 224,
    'normalize': False,
    'part_id': 0,
    'part_num': 1,
    'cache': None,
    'shuffle_size': 10000,
    'worker_args': {
        'cpp_xmap': False,
        'worker_num': 16,
        'buffer_size': 3000,
        'use_process': True,
        'use_sharedmem': True
    }
}


def train_image_mapper(img_size=None, normalize=None, default_class='pil'):
    """ a image mapper for training data
    """
    if img_size is None:
        img_size = g_settings['img_size']
    if normalize is None:
        normalize = g_settings['normalize']

    ops.default_class = default_class
    img_ops = [ops.DecodeImage()]
    img_ops += [ops.RotateImage(10, rand=True)]
    img_ops += [ops.RandCropImage(img_size)]
    img_ops += [ops.RandFlipImage()]
    img_ops += [ops.ToCHWImage()]

    if normalize:
        img_ops += [ops.NormalizeImage()]

    return img_ops


def test_image_mapper(img_size=None, normalize=None, default_class='pil'):
    """ a image mapper for testing data
    """
    if img_size is None:
        img_size = g_settings['img_size']
    if normalize is None:
        normalize = g_settings['normalize']

    ops.default_class = default_class
    img_ops = [ops.DecodeImage()]
    img_ops += [ops.ResizeImage(resize_short=img_size)]
    img_ops += [ops.CropImage(img_size)]
    img_ops += [ops.ToCHWImage()]
    if normalize:
        img_ops += [ops.NormalizeImage()]

    return img_ops


def make_reader(mode,
                uri,
                part_id=None,
                part_num=None,
                cache=None,
                pre_maps=None,
                pass_num=1,
                **kwargs):
    if part_id is None:
        part_id = g_settings['part_id']

    if part_num is None:
        part_num = g_settings['part_num']

    if cache is None:
        cache = g_settings['cache']

    ds = Dataset.load(
        uri=uri,
        part_id=part_id,
        part_num=part_num,
        cache=cache,
        pass_num=pass_num)

    if 'shuffle_size' in kwargs:
        sf_sz = kwargs['shuffle_size']
    else:
        sf_sz = g_settings['shuffle_size']

    if mode == 'train' and sf_sz > 0:
        ds.shuffle(sf_sz)

    maps = []
    if pre_maps is not None:
        assert type(
            pre_maps) is list, 'invalid pre_maps param in build_pipeline'
        maps += pre_maps

    args = copy.deepcopy(g_settings['worker_args'])
    args.update(kwargs)
    if 'lua_fname' in kwargs and kwargs['lua_fname']:
        img_ops = [
            ops.LuaProcessImage(
                lua_fname=kwargs['lua_fname'], tochw=True)
        ]
    else:
        img_ops = train_image_mapper(
        ) if mode == 'train' else test_image_mapper()

    for m in maps:
        ds.map(m)

    ds.map_ops(img_ops, **args)
    return ds.reader()


def train(uri, **kwargs):
    return make_reader('train', uri, **kwargs)


def val(uri, **kwargs):
    return make_reader('val', uri, **kwargs)
