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
#    build pipelines of data processing on coco for model training and validation
#
"""
import copy
import random
import numpy as np
import functools
import cv2
import logging

from . import box_utils
from ... import pipeline
from ...misc.edict import AttrDict
#mode type for concurrent processing of image data
WORKER_MODE_TYPES = ['native_thread', 'python_thread', 'python_process']

logger = logging.getLogger(__name__)

#default configs for sample processing
df_cfg = AttrDict()
df_cfg.TRAIN = AttrDict()
df_cfg.TRAIN.scales = [800]
df_cfg.TRAIN.max_size = 1333
df_cfg.TEST = AttrDict()
df_cfg.TEST.max_size = 1333
df_cfg.normlize_scale = 1.0 / 255
df_cfg.img_mean = [0.485, 0.456, 0.406]
df_cfg.img_std = [0.229, 0.224, 0.225]
df_cfg.to_rgb = True
df_cfg.infer_scale = 800

default_settings = {
    'sample_parser': lambda r: (r['image'], r['label']),
    'sample_mapper': None,
    'sample_filter': lambda r: r is not None,
    'process_cfg' : df_cfg,
    'sample_post_mapper': None,
    'worker_args': { #config for concurrent processing
        'worker_mode': WORKER_MODE_TYPES[1],
        'worker_num': 16,
        'buffer_size': 200,
        'use_sharedmem': True,
        'shared_memsize': 4 * (1024 ** 3)
    }
}


def get_label_info(height, width, label):
    """ get bounding boxes info and class info from 'label'
    """
    boxes = label['boxes']
    valid_boxes = []
    for box in boxes:
        if box['_area'] < -1 or \
            ('_ignore' in box and box['_ignore'] == 1):
            continue

        x1, y1, x2, y2 = box_utils.xywh_to_xyxy(box['pos'])
        x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(\
            x1, y1, x2, y2, height, width)
        if box['_area'] > 0 and x2 > x1 and y2 > y1:
            box['_clean_pos'] = [x1, y1, x2, y2]
            valid_boxes.append(box)

    obj_num = len(valid_boxes)
    gt_boxes = np.zeros((obj_num, 4), dtype='float32')
    gt_classes = np.zeros((obj_num), dtype='int32')
    is_crowd = np.zeros((obj_num), dtype='int32')
    for ix, box in enumerate(valid_boxes):
        gt_classes[ix] = box['_category_id']
        gt_boxes[ix, :] = box['_clean_pos']
        is_crowd[ix] = box['_iscrowd']

    return gt_boxes, gt_classes, is_crowd


def get_image_info(img, target_size, max_size, cfg):
    """ process image for detection model training
    """
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    img_scale = float(target_size) / float(im_size_min)

    if np.round(img_scale * im_size_max) > max_size:
        img_scale = float(max_size) / float(im_size_max)

    img = cv2.resize(img, None, None, \
            fx=img_scale, fy=img_scale, \
            interpolation=cv2.INTER_LINEAR)

    # normalize
    if cfg.img_mean is not None:
        img = img.astype(np.float32, copy=False)
        if cfg.normlize_scale:
            img = img * cfg.normlize_scale

        mean = np.array(cfg.img_mean, dtype='float32')
        std = np.array(cfg.img_std, dtype='float32')
        img = (img - mean) / std

    # swap from hwc to chw
    channel_swap = (2, 0, 1)  #(batch, c, h, w)
    img = img.transpose(channel_swap)
    return (img, img_scale)


def default_sample_mapper(sample, mode, cfg):
    """ process sample to prepare training or validation data

    Args:
        @sample (dict): raw data parsed from file
        @mode (str): data for training or validation eg: 'train' 'val' or 'test'
        @cfg (AttrDict): configs for this mapper
    """
    if sample is None:
        return None

    image, label = sample
    data = np.frombuffer(image, dtype='uint8')
    img = cv2.imdecode(data, 1)  # BGR mode

    if cfg.to_rgb:
        img = img[:, :, ::-1]

    h, w = img.shape[:2]
    if mode == 'train':
        gt_boxes, gt_classes, is_crowd = get_label_info(h, w, label)
        if len(gt_boxes) == 0:
            return None

        flag_flip = random.randint(0, 1)
        if flag_flip == 1:
            oldx1 = gt_boxes[:, 0].copy()
            oldx2 = gt_boxes[:, 2].copy()
            gt_boxes[:, 0] = w - oldx2 - 1
            gt_boxes[:, 2] = w - oldx1 - 1
            img = img[:, ::-1, :]

        target_size = random.choice(cfg.TRAIN.scales)
        max_size = cfg.TRAIN.max_size
        img, img_scale = get_image_info(img, target_size, max_size, cfg)

        img_id = label['_id'] if '_id' in label else 0
        im_height = np.round(h * img_scale)
        im_width = np.round(w * img_scale)
        img_info = np.array([im_height, im_width, \
            img_scale], dtype=np.float32)
        return (img, gt_boxes, gt_classes, is_crowd, img_info, img_id)
    else:
        target_size = cfg.infer_scale
        max_size = cfg.TEST.max_size
        img, img_scale = get_image_info(img, target_size, max_size, cfg)
        img_id = label['_id'] if '_id' in label else 0
        im_height = np.round(h * img_scale)
        im_width = np.round(w * img_scale)
        img_info = np.array([im_height, im_width, \
            img_scale], dtype=np.float32)
        return (img, img_info, img_id)


def train(settings=None):
    """ build a pipeline of coco data processing for model training
    """
    #prepare trainning default settings
    df_sets = copy.deepcopy(default_settings)
    df_sets['shuffle_size'] = 10000
    if settings is not None:
        df_sets.update(settings)
        for k, v in default_settings['worker_args'].items():
            if k not in df_sets['worker_args']:
                df_sets['worker_args'][k] = v

    pl = pipeline.Pipeline()
    if df_sets['shuffle_size'] > 0:
        pl.shuffle(df_sets['shuffle_size'])

    if df_sets['sample_parser'] is not None:
        pl.map(df_sets['sample_parser'])

    worker_args = df_sets['worker_args']
    if worker_args['worker_mode'] == 'python_thread':
        worker_args['use_process'] = False
        worker_args['use_sharedmem'] = False
    else:
        worker_args['use_process'] = True

    del worker_args['worker_mode']
    if df_sets['sample_mapper'] is None:
        sample_mapper = default_sample_mapper
    else:
        sample_mapper = df_sets['sample_mapper']

    mapper = functools.partial(sample_mapper, \
        mode='train', cfg=df_sets['process_cfg'])
    reader_mapper = pipeline.decorator.Xmap(\
        mapper, **worker_args)
    pl.map(reader_mapper=reader_mapper)

    if df_sets['sample_post_mapper'] is not None:
        pl.map(df_sets['sample_post_mapper'])

    if df_sets['sample_filter'] is not None:
        pl.filter(df_sets['sample_filter'])

    return pl


def val(settings=None):
    """ build a pipeline of coco data processing for model validation
    """
    #prepare validation default settings
    df_sets = copy.deepcopy(default_settings)
    if settings is not None:
        df_sets.update(settings)
        for k, v in default_settings['worker_args'].items():
            if k not in df_sets['worker_args']:
                df_sets['worker_args'][k] = v

    pl = pipeline.Pipeline()
    if df_sets['sample_parser'] is not None:
        pl.map(df_sets['sample_parser'])

    worker_args = df_sets['worker_args']
    if worker_args['worker_mode'] == 'python_thread':
        worker_args['use_process'] = False
        worker_args['use_sharedmem'] = False
    else:
        worker_args['use_process'] = True

    del worker_args['worker_mode']
    if df_sets['sample_mapper'] is None:
        sample_mapper = default_sample_mapper
    else:
        sample_mapper = df_sets['sample_mapper']

    mapper = functools.partial(sample_mapper, \
        mode='val', cfg=df_sets['process_cfg'])
    reader_mapper = pipeline.decorator.Xmap(\
        mapper, **worker_args)
    pl.map(reader_mapper=reader_mapper)

    if df_sets['sample_post_mapper'] is not None:
        pl.map(df_sets['sample_post_mapper'])

    if df_sets['sample_filter'] is not None:
        pl.filter(df_sets['sample_filter'])

    return pl
