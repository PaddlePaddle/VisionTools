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
#   tools used for accessing files
"""

import os


class FileProxy(object):
    """ class to wrap a file for read and close,
        which is used for the purpose of cacheing data
    """

    def __init__(self, fd, cache):
        self.fd = fd
        self.cache_file = cache
        self.cached_fd = open(cache, 'wb')
        assert self.cached_fd is not None, "failed to open file[%s] to cache" % (
            cache)

    def close(self):
        """ close proxy
        """
        self.fd.close()
        self.cached_fd.close()

    def read(self, *args, **kwargs):
        """ read proxy
        """
        data = self.fd.read(*args, **kwargs)
        if len(data) > 0:
            self.cached_fd.write(data)
        return data


def list_dir(path, with_prefix=True):
    """list all files in the directory 'path' if it's a directory

    Args:
        @path (str): path to list

    Returns:
        @list of file names
    """
    if path.startswith('file:/'):
        path = path[len('file:/'):]

    assert os.path.exists(path), 'local path[%s] not exist' % (path)
    ret = []
    if os.path.isfile(path):
        ret.append(path)
    else:
        for i in os.listdir(path):
            fname = os.path.join(path, i)
            if os.path.isfile(fname):
                ret.append(fname)

    if with_prefix:
        return ['file:/' + os.path.abspath(i) for i in ret]
    else:
        return ret


def uri2path(uri):
    """ return real path for this uri
    """
    if uri.startswith('file:/'):
        return uri[len('file:/'):]
    elif uri.startswith('afs:/') or uri.startswith('hdfs:/'):
        pos = uri.find('@')
        if pos < 0:
            return uri
        else:
            return uri[pos + 1:]
    else:
        return uri


def is_local_path(uri_path):
    """ check whether this is a local path
    
    Args:
        @uri_path (str) path for a data in local disk or afs cluster
    """
    local_prefixes = ['.', '/', 'file:/']

    for p in local_prefixes:
        if uri_path.startswith(p):
            return True

    if uri_path.startswith('afs:/') or uri_path.startswith('hdfs:/'):
        return False
    else:
        return True


def open_file(uri, cache_to=None, suffix=''):
    """ open a uri, and return a fd
    """
    if is_local_path(uri):
        return open(uri2path(uri))
    else:
        return DfsTool.open(uri, cache_to, suffix)


#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
