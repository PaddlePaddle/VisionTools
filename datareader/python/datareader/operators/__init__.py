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

from .base import OperatorParamError
from .base import build

op_names = [
    'DecodeImage',
    'NormalizeImage',
    'ResizeImage',
    'RotateImage',
    'CropImage',
    'RandCropImage',
    'RandFlipImage',
    'RandDistortColor',
    'ToCHWImage',
    'LuaProcessImage',
]

from . import pil_ops
from . import opencv_ops
from . import base

# default class of operators to use
default_class = 'pil'


def _op_proxy(name):
    """ proxy for operators
    """

    def _proxy(*args, **kwargs):
        if 'op_class' in kwargs:
            op_class = kwargs['op_class']
            del kwargs['op_class']
        else:
            op_class = default_class
        op_class = op_class.lower()

        op = getattr(base, name, None)
        if op is None:
            op_mod = pil_ops if op_class == 'pil' else opencv_ops
            op = getattr(op_mod, name, None)

        assert op is not None, "not found %s in %s" % (name, op_mod.__name__)
        return op(*args, **kwargs)

    return _proxy


# install operators
for name in op_names:
    globals()[name] = _op_proxy(name)
