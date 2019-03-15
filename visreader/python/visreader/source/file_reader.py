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

import logging
from .source import SourceError
from ..misc import kvtool

logger = logging.getLogger(__name__)


def line_reader(f, bufsize=10240):
    """ line reader from file 'f'
    """
    d = ''
    while True:
        buf_data = f.read(bufsize)
        if buf_data is None or len(buf_data) == 0:
            break

        d += buf_data
        lines = d.split('\n')
        if len(lines) > 1:
            for l in lines[:-1]:
                yield l
            d = lines[-1]

    if len(d) > 0:
        yield d


class FileReader(object):
    """ file reader
    """

    def __init__(self, filetype, fd_reader, notify=None):
        """ init

        Args:
            @filetype (str): file formate, eg: seqfile or textfile
            @fd_reader (generator): provide opened fds
            @notify (function): cb for notify the samples and bytes readed
        """
        self.fd_reader = fd_reader
        self.filetype = filetype
        self.notify = notify

    def reader(self):
        """ generator to yield records from these fds
        """
        ft = self.filetype
        if ft == 'textfile':
            for i, f in enumerate(self.fd_reader()):
                ct = 0
                for l in line_reader(f):
                    ct += 1
                    yield l.rstrip('\n')

                if self.notify is not None:
                    self.notify(i, samples=ct)
        elif ft == 'seqfile':
            for i, f in enumerate(self.fd_reader()):
                ct = 0
                for r in kvtool.get_reader(f, type=ft):
                    ct += 1
                    yield r

                if self.notify is not None:
                    self.notify(i, samples=ct)
        else:
            raise SourceError('not supported filetype[%s]' % (ft))


#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
