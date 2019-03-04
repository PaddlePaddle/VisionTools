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
#
# function
#   a python wrapper C++ class 'CTransformer'
"""

import os
import sys
import logging
import json

logger = logging.getLogger(__name__)
from .libpytransform import CyProcessor
from .libpytransform import CyTransformer
from .libpytransform import TransformerException

#ref to 'include/opencv2/imgproc.hpp' for detail
interpolation_flags = {
    'INTER_NEAREST': 0,
    'INTER_LINEAR': 1,  #bilinear interpolation
    'INTER_CUBIC': 2,  # bicubic interpolation
    'INTER_AREA': 3,
    'INTER_LANCZOS4': 4,  # Lanczos interpolation over 8x8 neighborhood
    'INTER_LINEAR_EXACT': 5,  # Bit exact bilinear interpolation
    'INTER_MAX': 7,  # mask for interpolation codes
    'WARP_FILL_OUTLIERS': 8,
}


class PyTransformer(object):
    """ a wrapper for 'CyTransformer' implemented in C++
    """

    def __init__(self, transformer, with_meta=False):
        self._cytransformer = transformer
        self._conf = {}
        self._buffer = {}
        self._id = 0
        self._with_meta = with_meta  #whether allowed to meta to pass through

    def start(self):
        """ start the underling CyTransformer
        """
        return self._cytransformer.start()

    def get(self, ctx=None):
        """ get a transformed data from underling CyTransformer
        """
        ctx = {} if ctx is None else ctx
        img, label = self._cytransformer.get(ctx)
        if img is None or not self._with_meta:
            return img, label

        id = ctx['id']
        meta = self._buffer[id]
        del self._buffer[id]
        return img, label, meta

    def put(self, image, label, meta=None):
        """ put a sample to CyTransformer
        """
        label = str(label) if type(label) is not str else label
        id = self._id
        self._cytransformer.put(image, label, {'id': id})

        if self._with_meta:
            assert id not in self._buffer
            self._buffer[id] = meta
        else:
            if meta is not None:
                logger.warn('cannot put meta to this pytransformer')

        self._id += 1
        if self._id >= sys.maxint:
            self._id = 0

    def stop(self):
        """ stop CTransformer (just indicate to stop)
        """
        self._cytransformer.stop()


class ImageOpConf(object):
    def __init__(self):
        self.reset()

    def reset(self):
        """ clear all operators
        """
        self._ops = []
        return self

    def lua(self, lua_fname='', lua_code='', tochw=False):
        """ add op which is implemented by lua code 'lua_fname' or 'lua_code'

        Args:
            @lua_fname(str): file path to a lua script 
            @lua_code(str): a lua code string
            @tochw (bool): whether convert to 'chw' from 'hwc' for final image
        """
        assert type(lua_fname) is str and type(
            lua_code) is str, "invalid type of params for lua op"
        assert len(lua_fname) > 0 or len(lua_code) > 0, 'invalid lua script'

        self._ops.append(("lua_op", {
            "lua_fname": lua_fname,
            "lua_code": lua_code,
            "tochw": int(tochw)
        }))
        return self

    def decode(self, to_rgb=None):
        """ decode image
        """
        if to_rgb is None or to_rgb is False:
            mode = 'UNCHANGED'
        else:
            mode = 'RGB'

        #defined in 'opencv2/imgcodecs/imgcodecs_c.h'
        mode2num = {'UNCHANGED': -1, 'GRAY': 0, 'RGB': 1}
        conf = {"mode": mode2num[mode]}
        self._ops.append(("decode", conf))
        return self

    def crop(self, x, y, w, h):
        """ crop a decoded image

        Args:
            x (int): start offset of left-to-right
            y (int): start offset of top-to-bottom
            w (int): width to crop
            h (int): height to crop

        Return:
            self
        """
        conf = {
            "crop_x": str(x),
            "crop_y": str(y),
            "crop_w": str(w),
            "crop_h": str(h),
        }
        self._ops.append(("crop", conf))
        return self

    def center_crop(self, size, center=True):
        """ random crop a sub area in decoded image and resize to 'size'

        Args:
            size (int): final size to return for this op
            center (bool): wheather really crop center

        Return:
            self
        """
        if type(size) is int:
            size = (size, size)

        conf = {
            "crop_center": str(1 if center else 0),
            "crop_w": str(size[0]),
            "crop_h": str(size[1]),
        }
        self._ops.append(("crop", conf))
        return self

    def random_crop(self, size, scale=None, ratio=None):
        """ random crop a sub area in decoded image and resize to 'size'

        Args:
            size (int): final size to return for this op
            scale (list of floats): max scale size of w and h
            ratio (list of floats):  

        Return:
            self
        """
        if type(size) is int:
            size = (size, size)

        scale = [0.08, 1.0] if scale is None else scale
        ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

        conf = {
            "scale": ",".join([str(i) for i in scale]),
            "ratio": ",".join([str(i) for i in ratio]),
            "interpolation": interpolation_flags['INTER_LANCZOS4'],
            "final_size": ",".join([str(i) for i in size])
        }
        self._ops.append(("random_crop", conf))
        return self

    def resize(self, w, h, interpolation=None):
        """ resize the image to target size

        Args:
            w (int), target width after resize
            h (int), target height after resize
        """
        if interpolation is None:
            interpolation = interpolation_flags['INTER_NEAREST']
        elif type(interpolation) is str:
            interpolation = interpolation_flags[interpolation]

        conf = {
            "resize_w": str(w),
            "resize_h": str(h),
            "interpolation": str(interpolation)
        }
        self._ops.append(("resize", conf))
        return self

    def resize_short(self, size, interpolation=None):
        """ resize short edge to 'size'

        Args:
            size (int): target size of short edge

        Return:
            self
        """
        if interpolation is None:
            interpolation = interpolation_flags['INTER_NEAREST']
        elif type(interpolation) is str:
            interpolation = interpolation_flags[interpolation]

        conf = {"short_size": str(size), "interpolation": interpolation}
        self._ops.append(("resize", conf))
        return self

    def rotate(self, angle=None, random_range=None):
        """ rotate the image

        Args:
            angle (int), degree to rotate
            random_range (uint), random rotate range in [-random_range, random_range]

        Return:
            self
        """
        if random_range is not None:
            conf = {"random_range": str(random_range)}
        else:
            assert angle is not None, "invalid param for both 'angle' and 'random_angle' are None"
            conf = {"angle": str(angle)}

        self._ops.append(("rotate", conf))
        return self

    def flip(self, flip_code, random=False):
        """ flip a image

        Args:
            flip_code (str): 'FLIP_LEFT_RIGHT' or 'FLIP_UP_DOWN'

        Return:
            self
        """
        code2num = {'FLIP_TOP_BOTTOM': 0, 'FLIP_LEFT_RIGHT': 1}
        conf = {
            "flip_code": str(code2num[flip_code]),
            "random": str(1 if random else 0),
        }
        self._ops.append(("flip", conf))
        return self

    def to_chw(self):
        self._ops.append(("tochw", {"value": 1}))
        return self

    def build_ops_conf(self):
        """ build operators conf for creating a image process from C++
        """
        assert len(self._ops) > 0, "no operators added"

        ops_conf = []
        for op_name, op_conf in self._ops:
            conf = {'op_name': op_name}
            conf.update(op_conf)
            ops_conf.append({k: str(v) for k, v in conf.items()})
        return ops_conf

    def __str__(self):
        op_info = ['[%s:%s]' % (n, json.dumps(c)) for n, c in self._ops]
        ret = 'ImageOpConf has operators:{%s}' % (','.join(op_info))
        return ret


class Builder(ImageOpConf):
    """ a builder to configure and create CTransformer from C++
    """

    def __init__(self, thread_num=1, queue_limit=None):
        super(Builder, self).__init__()
        if queue_limit is None:
            queue_limit = 1000

        self.reset()
        self._conf['thread_num'] = thread_num
        self._conf['worker_queue_limit'] = queue_limit
        self._conf['swapaxis'] = 0

    def reset(self):
        """ reset
        """
        super(Builder, self).reset()
        self._conf = {'thread_num': 1, 'worker_queue_limit': 1000}
        return self

    def init(self, conf):
        """ init with user's configs
        """
        self._conf = conf
        return self

    def set_conf(self, k, v):
        """ set configuration items
        """
        self._conf[k] = v
        return self

    def build(self, with_meta=False):
        """ create C++ CTransformer from current configuration

        Args:
            None

        Return:
            CTransformer instance which provide [start, get, put, stop] interfaces
        """
        assert len(self._ops) > 0, "no ops added for this transformer"

        ops_conf = self.build_ops_conf()
        cf = {k: str(v) for k, v in self._conf.items()}
        ctransformer = CyTransformer(cf, ops_conf)
        return PyTransformer(ctransformer, with_meta)


class PyProcessor(ImageOpConf):
    def __init__(self):
        super(PyProcessor, self).__init__()
        self._cyprocessor = None

    def reset(self):
        """ clear all ops
        """
        super(PyProcessor, self).reset()
        self._cyprocessor = None
        return self

    def _init(self):
        """ init CyProcessor
        """
        ops_conf = self.build_ops_conf()
        self._cyprocessor = CyProcessor(ops_conf)
        return self._cyprocessor

    def __call__(self, image, label=None):
        if self._cyprocessor is None:
            self._init()

        if label is None:
            img, _ = self._cyprocessor.process(image, '')
            return img
        else:
            return self._cyprocessor.process(image, str(label))


class Keeper(object):
    """ a class for holding resource object and stop them when no reference exist
    """

    def __init__(self, o):
        self._resources = [o]

    def add(self, o):
        """ add a new resource object
        """
        self._resources.append(o)

    def __del__(self):
        for o in self._resources:
            o.stop()


def xmap_reader(reader, planner, buffer_size=1000, \
        worker_num=16, with_label=True, post_mapper=None, **kwargs):
    logger.debug('not used params in pytransformer.xmap_reader:[%s]' %
                 (str(kwargs)))

    planner.set_conf('thread_num', worker_num)
    planner.set_conf('worker_queue_limit', buffer_size)

    def _mapper(r):
        if post_mapper is not None:
            return post_mapper(r)
        else:
            return r

    def _fetch_data(transformer):
        ctx = {}
        try:
            img, label, meta = transformer.get(ctx)
        except TransformerException as exp:
            logger.warn('failed to fetch result from transformer.get with '\
                'exception:%s' % (str(exp)))
            return True

        if img is None:
            return False
        elif 'err_no' in ctx and ctx['err_no'] != 0:
            logger.info('faield convert image err_no[%d] and err_msg[%s]',
                        ctx['err_no'], ctx['err_msg'])
            return True
        else:
            if len(meta) > 0:
                sample = tuple([img, label] + list(meta)) if with_label \
                        else tuple([img] + list(meta))
            else:
                sample = (img, label) if with_label else (img, )
            return _mapper(sample)

    def _sync_reader():
        cpp_transformer = planner.build(with_meta=True)
        keeper = Keeper(cpp_transformer)

        cpp_transformer.start()
        count = 0
        for r in reader():
            img = r[0]
            assert (len(img) > 0), "invalid image with lenght[%d]" % (len(img))

            label = ''
            if with_label:
                label = r[1]

            meta = r[2:] if with_label else r[1:]
            cpp_transformer.put(img, label, meta)
            count += 1
            if count >= buffer_size:
                r = _fetch_data(cpp_transformer)
                if r is False:
                    return
                elif r is True:
                    count -= 1
                else:
                    count -= 1
                    yield r
        logging.debug('queue count %s buffer_size %s', count, buffer_size)
        for i in range(count):
            r = _fetch_data(cpp_transformer)
            if r is False:
                return
            elif r is True:
                pass
            else:
                yield r

    return _sync_reader


class CppXmap(object):
    def __init__(self, planner, worker_num=16, \
        buffer_size=1000, pre_feed=None, **kwargs):
        self.args = {}
        for k, v in locals().items():
            if k not in ['self', 'kwargs']:
                self.args[k] = v

        self.args.update(kwargs)

    def __call__(self, reader):
        return xmap_reader(reader, **self.args)


#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
