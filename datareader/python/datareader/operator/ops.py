""" operators used to applied on samples
"""

import os
import math
import random
import io
import copy
import functools
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)

class OperatorParamError(ValueError):
    pass

class Operator(object):
    """ base class for all kinds of operator which is used to transform a sample,
        such as decode/resize/crop image
    """
    def __init__(self):
        pass

    def execute(self, *args, **kwargs):
        """ execute the transformation plan defined by this op
        """
        return self._execute(*args, **kwargs)

    def _execute(self, *args, **kwargs):
        raise NotImplementedError('%s::_executor not implemented' \
            % (type(self).__name__))

    def make_plan(self, builder):
        """ register the transformation plan to 'builder' which
            can be used to create an accelerated pipeline of transformations
        """
        return self._make_plan(builder)

    def _make_plan(self, builder):
        raise NotImplementedError('%s::_make_plan not implemented' \
            % (type(self).__name__))


class DecodeImage(Operator):
    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.to_np = to_np #to numpy
        self.channel_first = channel_first #only enabled when to_np is True

    def _execute(self, img):
        assert type(img) is str and len(img) > 0, "invalid input 'img' in DecodeImage"
        stream = io.BytesIO(img)
        img = Image.open(stream)
        if self.to_rgb and img.mode != 'RGB':
            img = img.convert('RGB')

        if self.to_np is True:
            img = np.array(img)
            if channel_first:
                img = img.transpose((2, 0, 1))
        
        return img

    def _make_plan(self, builder):
        return builder.decode(self.to_rgb)


class NormalizeImage(Operator):
    def __init__(self, scale=None, mean=None, std=None, order='chw'):
        super(NormalizeImage, self).__init__()
        self.scale = scale if scale is not None else 1.0 / 255.0
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape)
        self.std = np.array(std).reshape(shape)

    def _execute(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToCHWImage(Operator):
    def __init__(self):
        super(ToCHWImage, self).__init__()

    def _execute(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))

    def _make_plan(self, builder):
        return builder.to_chw()


class ResizeImage(Operator):
    def __init__(self, size=None, resize_short=None):
        super(ResizeImage, self).__init__()
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

    def _execute(self, img):
        assert isinstance(img, Image.Image), "invalid input 'img' in ResizeImage"
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img.size[0], img.size[1])
            w = int(round(img.size[0] * percent))
            h = int(round(img.size[1] * percent))
        else:
            w = self.w
            h = self.h

        return img.resize((w, h), Image.LANCZOS)

    def _make_plan(self, builder):
        intp = 'INTER_LANCZOS4'
        if self.resize_short is not None:
            return builder.resize_short(self.resize_short, interpolation=intp)
        else:
            return builder.resize(self.w, self.h, interpolation=intp)


class RotateImage(Operator):
    def __init__(self, rg, rand=True):
        super(Operator, self).__init__()
        assert type(rg) == int and rg > 0, "only positive interger "\
            "are allowed for RandRotateImage"
        self.range = rg
        self.rand = rand

    def _execute(self, img):
        assert isinstance(img, Image.Image), "invalid input 'img' in RandRotateImage"
        rg = self.range
        if self.rand:
            angle = random.randint(-rg, rg)
        else:
            angle = rg

        return img.rotate(angle)
 
    def _make_plan(self, builder):
        if self.rand:
            return builder.rotate(random_range=self.range)
        else:
            return builder.rotate(angle=self.range)

class CropImage(Operator):
    def __init__(self, size):
        super(CropImage, self).__init__()
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def _execute(self, img):
        assert isinstance(img, Image.Image), "invalid input 'img' in ResizeImage"
        w, h = self.size
        width, height = img.size
        w_start = (width - w) / 2
        h_start = (height - h) / 2
       
        w_end = w_start + w
        h_end = h_start + h
        return img.crop((w_start, h_start, w_end, h_end))

    def _make_plan(self, builder):
        return builder.center_crop(self.size)


class RandCropImage(Operator):
    def __init__(self, size, scale=None, ratio=None):
        super(RandCropImage, self).__init__()
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def _execute(self, img):
        assert isinstance(img, Image.Image), "invalid input 'img' in RandCropImage"

        size = self.size
        scale = self.scale
        ratio = self.ratio
        
        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        bound = min((float(img.size[0]) / img.size[1]) / (w ** 2),
                    (float(img.size[1]) / img.size[0]) / (h ** 2))
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

    def _make_plan(self, builder):
        return builder.random_crop(self.size, scale=self.scale, ratio=self.ratio)


class RandFlipImage(Operator):
    def __init__(self, flip_dir=None):
        super(RandFlipImage, self).__init__()
        self.flip_dir = flip_dir if flip_dir is not None else Image.FLIP_LEFT_RIGHT

    def _execute(self, img):
        assert isinstance(img, Image.Image), "invalid input 'img' in RandFlipImage"
        if random.randint(0, 1) == 1:
            return img.transpose(self.flip_dir)
        else:
            return img

    def _make_plan(self, builder):
        if self.flip_dir == Image.FLIP_LEFT_RIGHT:
            flip_dir = 'FLIP_LEFT_RIGHT'
        elif self.flip_dir == Image.FLIP_TOP_BOTTOM:
            flip_dir = 'FLIP_TOP_BOTTOM'
        else:
            raise OperatorParamError("not support this type of flip[%d]" % (self.flip_dir))

        return builder.flip(flip_dir, random=True)


class RandDistortColor(Operator):
    def __init__(self):
        super(RandDistortColor, self).__init__()

        def random_brightness(img, lower=0.5, upper=1.5):
            """ random_brightness """
            e = random.uniform(lower, upper)
            return ImageEnhance.Brightness(img).enhance(e)

        def random_contrast(img, lower=0.5, upper=1.5):
            """ random_contrast """
            e = random.uniform(lower, upper)
            return ImageEnhance.Contrast(img).enhance(e)

        def random_color(img, lower=0.5, upper=1.5):
            """ random_color """
            e = random.uniform(lower, upper)
            return ImageEnhance.Color(img).enhance(e)

        self.ops = [random_brightness, random_contrast, random_color]

    def _execute(self, img):
        assert isinstance(img, Image.Image), "invalid input 'img' in RandomFlipImage"
        ops = copy.copy(self.ops)
        random.shuffle(ops)
        for f in ops:
            img = f(img)

        return img


def support_multiple_inputs(func):
    """ a decorator to make function can accept a tuple as argument,
        like: (img, label) (img, ) or img
    """
    def _decorated_func(args):
        if isinstance(args, tuple):
            first = args[0]
            others = args[1:]
        else:
            first = args
            others = None

        result = func(first)
        if others is None:
            return result
        else:
            return tuple([result] + list(others))

    return _decorated_func


def build_mapper(ops):
    if type(ops) is tuple:
        ops = list(ops)

    if type(ops) is not list or len(ops) == 0:
        raise OperatorParamError("invalid type of 'ops' in build_mapper")

    @support_multiple_inputs
    def _mapper(img):
        for op in ops:
            img = op.execute(img)

        return img

    return _mapper


def build_fast_mapper(ops, builder):
    noacc_ops = []
    for i, o in enumerate(ops):
        try:
            o.make_plan(builder)
        except NotImplementedError as e:
            noacc_ops += ops[i:]
            break

    post_mapper = None
    if len(noacc_ops) > 0:
        logger.debug('left last %d python ops', len(noacc_ops))
        post_mapper = build_mapper(noacc_ops)

    return post_mapper


def build(ops, workers=16, decoded_bufsize=10000, accelerate=False):
    """ build a function which accept a reader and return another processed reader
    """
    logger.debug('build image ops with workers:%d, decoded_bufsize:%d, acc:%s',
        workers, decoded_bufsize, str(accelerate))

    if not accelerate:
        from ..pipeline.decorator import xmap_readers
        mapper = build_mapper(ops)
        return functools.partial(
            xmap_readers,
            mapper=mapper,
            process_num=workers,
            buffer_size=decoded_bufsize)
    else:
        from ..transformer.pytransformer import Builder
        from ..transformer.pytransformer import fast_xmap_readers
        bd = Builder(thread_num=workers, queue_limit=decoded_bufsize)
        post_mapper = build_fast_mapper(ops, bd)
        return functools.partial(
            fast_xmap_readers,
            builder=bd,
            post_mapper=post_mapper)
