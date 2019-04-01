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
import numpy as np
import functools
import logging
logger = logging.getLogger(__name__)


class OperatorParamError(ValueError):
    pass


class LuaProcessImage(object):
    """ an lua operator which can execute any code in lua env
    """

    def __init__(self, lua_fname='', lua_code='', tochw=False):
        self._lua_fname = lua_fname
        self._lua_code = lua_code
        self._tochw = tochw
        self._pyprocessor = None

    def __call__(self, img):
        assert len(self._lua_script) > 0, 'invalid lua script'

        if self._pyprocessor is None:
            from ..pytransformer import PyProcessor
            self._pyprocessor = PyProcessor()
            self._pyprocessor.lua(lua_fname=self._lua_fname,
                                  lua_code=self._lua_code,
                                  tochw=self._tochw)

        res, _ = self._pyprocessor(img)
        return res

    def make_plan(self, planner):
        planner.lua(lua_fname=self._lua_fname,
                    lua_code=self._lua_code,
                    tochw=self._tochw)


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None, order='chw'):
        self.scale = scale if scale is not None else 1.0 / 255.0
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape)
        self.std = np.array(std).reshape(shape)

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        return (img.astype('float32') * self.scale - self.mean) / self.std

    def make_plan(self, planner):
        raise NotImplementedError('%s::_make_plan not implemented' \
            % (type(self).__name__))


class ToCHWImage(object):
    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))

    def make_plan(self, planner):
        return planner.to_chw()


def choose_first_param(func):
    """ A decorator to choose the first element of the input(if it's tuple)
        as the param of 'func' which only accept one param. And the output
        of 'func' will be concated with other params input.
        The decorated function can accept params like: (img, label) (img, ) or img
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

    @choose_first_param
    def _mapper(img):
        for op in ops:
            img = op(img)

        return img

    return _mapper


def make_cpp_plan(ops, planner):
    noacc_ops = []
    for i, o in enumerate(ops):
        try:
            o.make_plan(planner)
        except NotImplementedError as e:
            noacc_ops += ops[i:]
            break

    post_mapper = None
    if len(noacc_ops) > 0:
        logger.debug('left last %d python ops', len(noacc_ops))
        post_mapper = build_mapper(noacc_ops)

    return post_mapper


def build(ops, worker_num=16, buffer_size=1000, \
        worker_mode='python_thread', \
        use_sharedmem=False, **kwargs):
    """ build a concurrently processing reader decorator which accept 
        a reader as input and return the processed reader as output

    Args:
        @ops (list): list of operator instance
        @worker_num (int): num of workers to process in the decorator
        @worker_mode (str): concurrency mode, eg: python_thread, python_process or native_thread
        @use_sharedmem (bool): whether to use shared memory for IPC

    Returns:
        decorator of reader
    """
    logger.debug('build concurrent mapper in mode[%s]' % (worker_mode))
    if worker_mode == 'native_thread':
        if use_sharedmem:
            logger.warn('not supported use_sharedmem in native_thread mode')
        from ..transformer.pytransformer import Builder
        from ..transformer.pytransformer import CppXmap
        planner = Builder()
        post_mapper = make_cpp_plan(ops, planner)
        return CppXmap(
            planner,
            buffer_size=buffer_size,
            worker_num=worker_num,
            post_mapper=post_mapper)
    else:
        mapper = build_mapper(ops)
        from ..pipeline.decorator import Xmap
        return Xmap(mapper, worker_num=worker_num, buffer_size=buffer_size, \
                use_sharedmem=use_sharedmem, **kwargs)
