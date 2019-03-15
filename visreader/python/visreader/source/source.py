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
#   base class for different data source
"""

import os
from urlparse import urlparse


class SourceError(ValueError):
    """ source error
    """
    pass


def strip_spaces(l):
    """ strip spaces
    """
    if type(l) is str:
        return l.strip(' ').strip('\t')
    else:
        return l


class SourceMeta(object):
    """ meta of data source
    """

    def __init__(self,
                 uri,
                 filetype=None,
                 part_id=None,
                 part_num=None,
                 cache=None,
                 pass_num=1,
                 to_dict=True):
        """ init
        """
        self.uri = strip_spaces(uri)

        if filetype is not None:
            assert filetype in ['textfile', 'seqfile'], 'not supported filetype[%s]' \
                    % (filetype)

        self.filetype = filetype if filetype is not None else 'seqfile'
        self.part_id = part_id if part_id is not None else 0
        self.part_num = part_num if part_num is not None else 1
        self.cache = strip_spaces(cache)
        self.pass_num = pass_num
        self.to_dict = to_dict

        if self.cache is not None:
            uri_path = urlparse(self.uri).path
            self.cache_sub_dir = uri_path.replace('/', '_').lstrip('_')

        self.total_file_num = None  #total file num for this dataset before partition
        self.flist = None
        self.sample_num = None

    def copy(self):
        """ copy myself
        """
        import copy
        return copy.deepcopy(self)

    def set_uri(self, uri, cache_sub_dir=None, filelist=None):
        """ set new uri
        """
        self.uri = strip_spaces(uri)
        if self.cache is not None:
            if cache_sub_dir is not None:
                self.cache_sub_dir = cache_sub_dir
            else:
                uri_path = urlparse(self.uri).path
                self.cache_sub_dir = uri_path.replace('/', '_').lstrip('_')


class DataSource(object):
    """ abstraction of different data sources
    """
    _supported_sources = []

    @classmethod
    def create(cls, meta):
        """ create a datasource
        """
        assert isinstance(meta, SourceMeta)
        rd = None
        for s in cls._supported_sources:
            if s.is_supported(meta.uri):
                return s(meta)

        raise SourceError('not supported this type of source[%s]' % (meta.uri))

    @classmethod
    def register(cls, source_cls):
        """ register implemented sources to this class
        """
        cls._supported_sources.append(source_cls)

    @classmethod
    def is_supported(self, uri):
        """ whether this datasource support this uri
        """
        raise NotImplementedError()

    @classmethod
    def partition(cls, flist, part_id, part_num):
        """ split files in 'flist' into 'part_num' parts and
            return one part of them specified by 'part_id'
        """
        assert len(flist) > 0, "not found any files when partition"

        files = []
        for i in range(len(flist)):
            if i % part_num == part_id:
                files.append(flist[i])

        return files

    def get_meta(self, param=None):
        """ get meta info about this datasource
        """
        if param is None:
            return self.meta
        else:
            return getattr(self.meta, param)

    def _make_reader(self):
        """ make a reader of this source
        """
        raise NotImplementedError(
            'invalid callinig to _make_reader of DataSource')

    def reader(self, pass_num=None):
        """ get a reader of this source

        Args:
            pass_num (int): number of times to replay data, 
                            <= 0 means infinite

        Returns:
            iterator maker
        """
        if pass_num is None:
            pass_num = self.meta.pass_num
        rd = self._make_reader()

        def _reader():
            ct = 0
            while True:
                for i in rd():
                    yield i
                ct += 1

                if pass_num > 0 and ct >= pass_num:
                    break

        return _reader


def load(uri, filetype=None, part_id=None, part_num=None, **kwargs):
    """ load data from local disk

        TODO:
            support hdfs and other source of data

    Args:
        @uri (str): location of data, eg: file://path/to/your/data
        @filetype (str): 'seqfile' or 'textfile' according to your data format
        @part_id (int): id of this node when partition the data
        @part_num (int): number of nodes

    Returns:
        DataSource instance
    """
    m = SourceMeta(
        uri=uri,
        filetype=filetype,
        part_id=part_id,
        part_num=part_num,
        **kwargs)

    return DataSource.create(m)


#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
