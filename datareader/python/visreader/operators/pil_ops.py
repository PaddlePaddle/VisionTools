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
# image operators implemened usinig PIL module,
# note that most of them support 'make_plan' which will be
# processed by cpp transformer
"""

import os
import math
import random
import io
import copy
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)

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
        stream = io.BytesIO(img)
        img = Image.open(stream)
        if self.to_rgb and img.mode != 'RGB':
            img = img.convert('RGB')

        if self.to_np is True:
            img = np.array(img)
            if channel_first:
                img = img.transpose((2, 0, 1))

        return img

    def make_plan(self, planner):
        """ plan pipeline of operators using 'planner'
        """
        return planner.decode(self.to_rgb)


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
        assert isinstance(img,
                          Image.Image), "invalid input 'img' in ResizeImage"
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img.size[0], img.size[1])
            w = int(round(img.size[0] * percent))
            h = int(round(img.size[1] * percent))
        else:
            w = self.w
            h = self.h

        return img.resize((w, h), Image.LANCZOS)

    def make_plan(self, planner):
        intp = 'INTER_LANCZOS4'
        if self.resize_short is not None:
            return planner.resize_short(self.resize_short, interpolation=intp)
        else:
            return planner.resize(self.w, self.h, interpolation=intp)


class RotateImage(object):
    def __init__(self, rg, rand=True):
        assert type(rg) == int and rg > 0, "only positive interger "\
            "are allowed for RandRotateImage"
        self.range = rg
        self.rand = rand

    def __call__(self, img):
        assert isinstance(
            img, Image.Image), "invalid input 'img' in RandRotateImage"
        rg = self.range
        if self.rand:
            angle = random.randint(-rg, rg)
        else:
            angle = rg

        return img.rotate(angle)

    def make_plan(self, planner):
        if self.rand:
            return planner.rotate(random_range=self.range)
        else:
            return planner.rotate(angle=self.range)


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        assert isinstance(img,
                          Image.Image), "invalid input 'img' in ResizeImage"
        w, h = self.size
        width, height = img.size
        w_start = (width - w) / 2
        h_start = (height - h) / 2

        w_end = w_start + w
        h_end = h_start + h
        return img.crop((w_start, h_start, w_end, h_end))

    def make_plan(self, planner):
        return planner.center_crop(self.size)


class RandCropImage(object):
    def __init__(self, size, scale=None, ratio=None):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def __call__(self, img):
        assert isinstance(img,
                          Image.Image), "invalid input 'img' in RandCropImage"

        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                    (float(img.size[1]) / img.size[0]) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img.size[0] * img.size[1] * random.uniform(scale_min,
                                                                 scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img.size[0] - w)
        j = random.randint(0, img.size[1] - h)

        img = img.crop((i, j, i + w, j + h))
        img = img.resize(size, Image.LANCZOS)
        return img

    def make_plan(self, planner):
        return planner.random_crop(
            self.size, scale=self.scale, ratio=self.ratio)


class RandFlipImage(object):
    def __init__(self, flip_dir=None):
        self.flip_dir = flip_dir if flip_dir is not None else Image.FLIP_LEFT_RIGHT

    def __call__(self, img):
        assert isinstance(img,
                          Image.Image), "invalid input 'img' in RandFlipImage"
        if random.randint(0, 1) == 1:
            return img.transpose(self.flip_dir)
        else:
            return img

    def make_plan(self, planner):
        if self.flip_dir == Image.FLIP_LEFT_RIGHT:
            flip_dir = 'FLIP_LEFT_RIGHT'
        elif self.flip_dir == Image.FLIP_TOP_BOTTOM:
            flip_dir = 'FLIP_TOP_BOTTOM'
        else:
            raise OperatorParamError("not support this type of flip[%d]" %
                                     (self.flip_dir))

        return planner.flip(flip_dir, random=True)


class RandDistortColor(object):
    def __init__(self, brightness=[0.5, 1.5],\
        contrast=[0.5, 1.5], color=[0.5, 1.5]):
        def random_brightness(img):
            """ random_brightness """
            lower, upper = brightness
            e = random.uniform(lower, upper)
            return ImageEnhance.Brightness(img).enhance(e)

        def random_contrast(img):
            """ random_contrast """
            lower, upper = contrast
            e = random.uniform(lower, upper)
            return ImageEnhance.Contrast(img).enhance(e)

        def random_color(img):
            """ random_color """
            lower, upper = color
            e = random.uniform(lower, upper)
            return ImageEnhance.Color(img).enhance(e)

        self.ops = [random_brightness, random_contrast, random_color]

    def __call__(self, img):
        assert isinstance(
            img, Image.Image), "invalid input 'img' in RandomFlipImage"
        ops = copy.copy(self.ops)
        random.shuffle(ops)
        for f in ops:
            img = f(img)

        return img

    def make_plan(self, planner):
        raise NotImplementedError('%s.make_plan not implemented' \
            % (type(self).__name__))
