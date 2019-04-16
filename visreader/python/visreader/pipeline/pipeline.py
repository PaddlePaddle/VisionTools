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
# This module provide a tool to facilitate the chainning of different readers
"""

import types
import json
import functools
import logging
import traceback
import threading
from . import decorator

logger = logging.getLogger(__name__)


class PipelineError(ValueError):
    """ PipelineError
    """
    pass


class SafeIter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        """__init__"""
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        """__iter__"""
        return self

    def __call__(self):
        """__call__"""
        self.call = self.it()
        return SafeIter(self.call)

    def next(self):
        """next"""
        with self.lock:
            return self.it.next()


def _batch(reader, batch_size, drop):
    """
    Create a batched reader.
    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :return: the batched reader.
    :rtype: callable
    """

    def _batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []

        if b and (not drop or len(b) == batch_size):
            yield b

    return _batch_reader


def filter_reader(func, reader):
    """ filter
    """

    def _reader():
        for r in reader():
            if func(r):
                yield r

    return _reader


def cache_reader(reader, where='memory'):
    """ cache data to memory
    """
    assert where == 'memory', 'now only memory cache is supported for this api'

    cache_status = {'cached': False, 'data': []}

    def _reader():
        if cache_status['cached']:
            logger.debug('use data in memory cache')
            for r in cache_status['data']:
                yield r
        else:
            for r in reader():
                cache_status['data'].append(r)
                yield r
            cache_status['cached'] = True
            logger.debug('cached data to memory')

    return _reader


def chain_funcs(funcs):
    """ chain a list of functions
    """

    def chained(*args, **kwargs):
        """ chained function
        """
        ret = funcs[0](*args, **kwargs)
        for f in funcs[1:]:
            ret = f(ret)
        return ret

    return chained


class Context(object):
    """ a class to record the context of a transformation in pipeline
    """

    def __init__(self):
        self._in_num = 0
        self._out_num = 0

    @property
    def in_num(self):
        """ in_num getter
        """
        return self.in_num

    @in_num.setter
    def in_num(self, value):
        """ in_num setter
        """
        self.in_num = value

    @property
    def out_num(self):
        """ out_num getter
        """
        return self.out_num

    @out_num.setter
    def out_num(self, value):
        """ out_num setter
        """
        self.out_num = value


class Pipeline(object):
    """ a class to facilitate chainning the transformations applied to 'reader'
    """

    def __init__(self, reader=None, threadsafe=False):
        """ init
        """
        self._reader = reader
        self.threadsafe = threadsafe
        self.reset()

    def reset(self, reader=None):
        """ reset the pipeline of transformations to initial state
        Args:
            reader (callable): a reader to provide data records
        Returns:
            None
        Raises:
            None
        """
        if reader is not None:
            self._reader = reader

        self._transformed = None
        self._pipeline = []

    def shuffle(self, size):
        """ shuffle the records in range of 'size'

        Args:
            size (int): size of shuffle range,
                        > 0: shuffle range
                        0: no shuffle
                        < 0: shuffle all

        Returns:
            self

        Raises:
            None
        """
        if size != 0:
            self._pipeline.append(('shuffle', {'size': size}))
        else:
            #0 means no shuffle
            pass

        return self

    def batch(self, size, drop=False):
        """ make batches from data items

        Args:
            size (int): size of one batch
            drop (bool): whether drop the last batch when not enough sample

        Returns:
            self

        Raises:
            None
        """
        self._pipeline.append(('batch', {'size': size, 'drop': drop}))
        return self

    def map(self, record_mapper=None, reader_mapper=None):
        """ do a function 'record_mapper' on every record or
            do a function 'reader_mapper' on one reader

        Args:
            record_mapper (function): the function to be applied to every record
            reader_mapper (function): the function to be applied to whole reader

        Returns:
            self

        Raises:
            None
        """
        assert (record_mapper is None and reader_mapper is not None) or \
                (record_mapper is not None and reader_mapper is None)

        self._pipeline.append(('map', {
            'record_mapper': record_mapper,
            'reader_mapper': reader_mapper
        }))
        return self

    def map_ops(self, ops, *args, **kwargs):
        """ map a list of Operators in 'ops'
        """
        from ..operators import build
        reader_mapper = build(ops, *args, **kwargs)
        return self.map(reader_mapper=reader_mapper)

    def filter(self, f):
        """ do a filtering 'f' on every record in this reader,
            only the records that meet the condition can be passed

        Args:
            f (function): the function to be applied to every record

        Returns:
            self

        Raises:
            None
        """
        self._pipeline.append(('filter', {'func': f}))
        return self

    def xmap(self,
             funcs,
             process_num=8,
             buffer_size=1000,
             order=False,
             use_process=False):
        """ use multipleprocess to map samples from previouse reader

        Args:
            funcs (function or list): one map or a list of maps
            process_num (int): process number to handle original sample
            buffer_size (int): max buffer size
            order (bool): keep the order of the reader

        Returns:
            self

        Raises:
            PipelineError when param not valid
        """

        if type(funcs) is list:
            f = chain_funcs(funcs)
        elif callable(funcs):
            f = funcs
        else:
            raise PipelineError(
                'invalid param for "funcs", not a function or a list of functions'
            )

        self._pipeline.append(('xmap', {'func': f, 'process_num': process_num, \
                'buffer_size': buffer_size, 'order': order, 'use_process': use_process}))

        return self

    def buffered(self, size):
        """ make the data records to be buffered without exceeding 'size' items

        Args:
            size (int): maximum buffer size

        Returns:
            self

        Raises:
            None
        """

        self._pipeline.append(('buffered', {'size': size}))
        return self

    def cache(self, where='memory'):
        """ cache the records to memory

        Args:
            where (str): where to cache the data

        Returns:
            self

        Raises:
            None
        """
        self._pipeline.append(('cache', {'where': where}))
        return self

    def transform(self, reader, infinite=False):
        """ transform the 'reader' using transformations defined in self._pipeline

        Args:
            reader (callable): a reader to provide data records
            infinite (bool): whether to repeat the reader forever

        Returns:
            trans_reader (callable): a transformed reader

        Raises:
            PipelineError when not supported op_name appears
        """
        assert callable(reader), "source reader is not a valid function"
        rd = reader
        for op_name, param in self._pipeline:
            if op_name == 'buffered':
                rd = decorator.buffered(rd, param['size'])
            elif op_name == 'cache':
                rd = cache_reader(rd, param['where'])
            elif op_name == 'shuffle':
                rd = decorator.shuffle(rd, param['size'])
            elif op_name == 'batch':
                rd = _batch(rd, param['size'], param['drop'])
            elif op_name == 'map':
                if param['record_mapper'] is not None:
                    rd = decorator.map_readers(param['record_mapper'], rd)
                else:
                    rd = param['reader_mapper'](rd)
            elif op_name == 'filter':
                rd = filter_reader(param['func'], rd)
            elif op_name == 'xmap':
                xmapper = decorator.Xmap(
                    mapper=param['func'],
                    process_num=param['process_num'],
                    buffer_size=param['buffer_size'],
                    order=param['order'],
                    use_process=param['use_process'])
                rd = xmapper(rd)
            else:
                raise PipelineError('not supported trasnfromation[%s]' %
                                    (op_name))

        def _guard_reader():
            while True:
                try:
                    for i in rd():
                        yield i
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logger.warn('exception occured in preprocessing pipeline '\
                            'with stack info[%s]' % (stack_info))
                    raise e
                if not infinite:
                    break

        _guard_reader = SafeIter(
            _guard_reader) if self.threadsafe else _guard_reader
        return _guard_reader

    def reader(self, infinite=False):
        """ get the transformed reader from 'self._reader'

        Args:
            None

        Returns:
            reader (callable): transformed reader

        Raises:
            None
        """
        if self._transformed is None:
            self._transformed = self.transform(self._reader, infinite)

        return self._transformed

    def __str__(self):
        """ readable representation for this object, used to debug
        Args:
            None

        Returns:
            readable string for this object

        Raises:
            None
        """
        id = 0
        ops = ['\nPipeline:']
        if len(self._pipeline) == 0:
            return '\n  '.join(ops + ['empty'])

        for op_name, param in self._pipeline:
            ops.append("{id:%d, op:%s, param:%s}" % (id, op_name, str(param)))
            id += 1
        return '\n  '.join(ops)


if __name__ == "__main__":
    """ test
    """

    def _data_reader():
        for i in xrange(11):
            yield i

    p = Pipeline(_data_reader)
    rd = p.xmap(lambda r: 2 * r, 2, 2).shuffle(4).batch(2).reader()
    for i in rd():
        print i

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
