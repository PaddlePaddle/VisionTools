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

import os
import random
import logging

from ..misc import filetool
from .source import DataSource
from .file_reader import FileReader

logger = logging.getLogger(__name__)


class LocalSource(DataSource):
    """ source for data on local disk
    """
    _type_name = 'LOCAL_SOURCE'

    def __init__(self, meta):
        super(LocalSource, self).__init__()
        assert self.is_supported(meta.uri)
        if not meta.uri.startswith('file://'):
            meta.uri = 'file:/' + os.path.abspath(meta.uri)

        self.meta = meta
        self._setup()

    def _setup(self):
        """ setup
        """
        m = self.meta
        flist = sorted(filetool.list_dir(m.uri))
        m.total_file_num = len(flist)
        m.flist = self.partition(flist, m.part_id, m.part_num)

    @classmethod
    def is_supported(cls, uri):
        """ whether this uri is supported by this class
        """
        if uri.startswith('file://'):
            return True
        elif uri.startswith('.') or uri.startswith('/'):
            return True
        else:
            return False

    @classmethod
    def strip_prefix(cls, uri):
        """ lstrip prefix
        """
        assert cls.is_supported(uri), 'invalid uri[%s] for this source' % (uri)
        return uri[len('file:/'):]

    @classmethod
    def get_type(cls):
        """ get type of this source
        """
        return cls._type_name

    def _make_reader(self):
        """ make a reader of this source
        """
        m = self.meta

        notified = {'index': None, 'samples': None}

        def _notify(index, samples=None):
            notified['index'] = index
            notified['samples'] = samples

        def _fd_reader():
            indices = range(len(m.flist))
            random.shuffle(indices)

            total_samples = 0
            for index, i in enumerate(indices):
                fname = m.flist[i]
                with open(self.strip_prefix(fname), 'r') as f:
                    yield f

                if index == notified['index'] and notified[
                        'samples'] is not None:
                    s = notified['samples']
                    total_samples += s
                    logger.debug('read %d/%d from file[%s]' % \
                            (s, total_samples, os.path.basename(fname)))
                    notified['samples'] = None

        return FileReader(m.filetype, _fd_reader, _notify).reader


DataSource.register(LocalSource)
#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
