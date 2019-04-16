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
"""
"""
# function
#    build pipelines of data processing on imagenet for model training and validation
#
"""
import copy
import logging
from ... import operators as ops
from ... import pipeline

#mode type for concurrent processing of image data
WORKER_MODE_TYPES = ['native_thread', 'python_thread', 'python_process']

logger = logging.getLogger(__name__)
default_settings = {
    'sample_parser': None,
    'image_size': 224,
    'lua_fname': None, #use lua code to process images
    'image_op_class': 'pil', #default to using PIL
    'normalize': True, #whether substract mean and divide std of image
    'worker_args': { #config for concurrent processing
        'worker_mode': WORKER_MODE_TYPES[0],
        'worker_num': 16,
        'buffer_size': 3000,
        'use_sharedmem': False
    }
}


def train(settings=None):
    """ build a pipeline of imagenet data processing
        for model training
    """
    #prepare trainning default settings
    df_sets = copy.deepcopy(default_settings)
    df_sets['shuffle_size'] = 10000
    if settings is not None:
        for k, v in settings.items():
            if v != 'worker_args':
                df_sets[k] = v
            else:
                df_sets[k].update(v)

    logger.debug('build pipeline of imagenet.train with settings[%s]' %
                 (str(df_sets)))

    pl = pipeline.Pipeline()
    if df_sets['shuffle_size'] > 0:
        pl.shuffle(df_sets['shuffle_size'])

    if df_sets['sample_parser'] is not None:
        pl.map(df_sets['sample_parser'])

    worker_args = df_sets['worker_args']
    if df_sets['lua_fname'] is None:
        ops.default_class = df_sets['image_op_class']
        img_ops = [ops.DecodeImage()]
        img_ops += [ops.RotateImage(10, rand=True)]
        img_ops += [ops.RandCropImage(df_sets['image_size'])]
        img_ops += [ops.RandFlipImage()]
        img_ops += [ops.ToCHWImage()]

        if df_sets['normalize']:
            img_ops += [ops.NormalizeImage()]
    else:
        if worker_args['worker_mode'] != 'native_thread':
            logger.warn(
                'lua code can only be running in mode[native_thread], so switch to that'
            )
        img_ops = [
            ops.LuaProcessImage(
                lua_fname=df_sets['lua_fname'], tochw=True)
        ]
        worker_args['worker_mode'] = 'native_thread'

    if worker_args['worker_mode'] == 'native_thread':
        worker_args['use_process'] = False
        worker_args['use_sharedmem'] = False
    elif worker_args['worker_mode'] == 'python_thread':
        worker_args['use_process'] = False
        worker_args['use_sharedmem'] = False
    elif worker_args['worker_mode'] == 'python_process':
        worker_args['use_process'] = True
        worker_args['use_sharedmem'] = True
    else:
        raise ValueError('not recognized mode[%s] for worker_args' %
                         (worker_args['worker_mode']))

    pl.map_ops(img_ops, **worker_args)
    return pl


def val(settings=None):
    """ build a pipeline of image data preprocessing
        for model validation
    """
    #prepare validation default settings
    df_sets = copy.deepcopy(default_settings)
    if settings is not None:
        for k, v in settings.items():
            if v != 'worker_args':
                df_sets[k] = v
            else:
                df_sets[k].update(v)

    logger.debug('build pipeline of imagenet.val with settings[%s]' %
                 (str(df_sets)))
    pl = pipeline.Pipeline()
    if df_sets['sample_parser'] is not None:
        pl.map(df_sets['sample_parser'])

    worker_args = df_sets['worker_args']
    if df_sets['lua_fname'] is None:
        ops.default_class = df_sets['image_op_class']
        img_ops = [ops.DecodeImage()]
        img_ops += [ops.ResizeImage(resize_short=df_sets['image_size'])]
        img_ops += [ops.CropImage(df_sets['image_size'])]
        img_ops += [ops.ToCHWImage()]

        if df_sets['normalize']:
            img_ops += [ops.NormalizeImage()]
    else:
        if worker_args['worker_mode'] != 'native_thread':
            logger.warn(
                'lua code can only be running in mode[native_thread], so switch to that'
            )

        img_ops = [
            ops.LuaProcessImage(
                lua_fname=df_sets['lua_fname'], tochw=True)
        ]
        worker_args['use_process'] = False
        worker_args['use_sharedmem'] = False

    pl.map_ops(img_ops, **worker_args)
    return pl
