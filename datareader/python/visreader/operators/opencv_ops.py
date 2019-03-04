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
# image operators implemened usinig cv2 module,
# note that most of them support 'make_plan' which will be
# processed by cpp transformer
"""

import os
import math
import random
import io
import copy
import functools
import numpy as np
from PIL import Image
import cv2

from .base import OperatorParamError
from .base import NormalizeImage
from .base import ToCHWImage


class DecodeImage(object):
    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.to_rgb = to_rgb
        self.to_np = to_np  #to numpy
        self.channel_first = channel_first  #only enabled when to_np is True

    def __call__(self, img):
        assert type(img) is str and len(
            img) > 0, "invalid input 'img' in DecodeImage"

        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


class ResizeImage(object):
    def __init__(self, size=None, resize_short=None):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h

        return cv2.resize(img, (w, h))


class RotateImage(object):
    def __init__(self, rg, rand=True):
        assert type(rg) == int and rg > 0, "only positive interger "\
            "are allowed for RandRotateImage"
        self.range = rg
        self.rand = rand

    def __call__(self, img):
        rg = self.range
        if self.rand:
            angle = random.randint(-rg, rg)
        else:
            angle = rg

        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) / 2
        h_start = (img_h - h) / 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class RandCropImage(object):
    def __init__(self, size, scale=None, ratio=None):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(\
            scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]
        return cv2.resize(img, size)


class RandFlipImage(object):
    def __init__(self, flip_dir=None):
        self.flip_dir = flip_dir if flip_dir is not None else Image.FLIP_LEFT_RIGHT

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            if self.flip_dir == Image.FLIP_LEFT_RIGHT:
                return cv2.flip(img, 0)
            else:
                return cv2.flip(img, 1)
        else:
            return img
