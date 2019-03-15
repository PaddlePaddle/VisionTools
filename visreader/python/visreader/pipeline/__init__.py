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

__all__ = ['decorator', 'Pipeline']

from . import decorator
from .. import source
from .pipeline import Pipeline


class Dataset(Pipeline):
    """ a helper for load data and apply transformers
    """

    def __init__(self, sc):
        super(Dataset, self).__init__()
        self._sc = sc

    @classmethod
    def load(cls, *args, **kwargs):
        """ load data from source
        """
        sc = source.load(*args, **kwargs)
        return Dataset(sc)

    def reader(self):
        """ return sample-generator maker for this source
        """
        return self.transform(self._sc.reader())
