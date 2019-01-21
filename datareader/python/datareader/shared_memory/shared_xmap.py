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
# a framework for dispatching tasks to multi-processes and 
# communicating data using shared memory
"""

import sys
import time
import traceback
import logging
import cPickle
import numpy as np

from .sharedmemory import SharedMemoryMgr
from .sharedmemory import SharedBuffer

logger = logging.getLogger(__name__)


def get_queue(queue_cap, use_process):
    """ get_queue
    """
    if use_process:
        from multiprocessing import Queue
    else:
        from Queue import Queue
    return Queue(queue_cap)


def concate_as_tuple(obj, *arrays):
    """ concate 'obj' and list in 'arrays' as one tuple
    """
    ret = [obj]
    for i in arrays:
        ret += list(i)
    return tuple(ret)


class FastQueue(object):
    """ A fast FIFO queue for transmiting large data between processes,
        it maybe highly efficient especially transmitting huge amount of data

        note that:
            1, only one process can produce and destroy 'SharedBuffer' with this queue
            2, other processes can only consume data from this queue but should
                not produce or destroy 'SharedBuffer' object
            3, the return 'SharedBuffer' from 'self.get(ret_buffer=True)' can be reused 
                in consumer processes
    """
    s_meta_size = 2048

    def __init__(self,
                 qsize,
                 mem_size=1024 * 1024,
                 mem_mgr=None,
                 use_process=True):
        """ init
        """
        self._qsize = qsize  #queue size
        self._inner_queue = get_queue(qsize, use_process)

        self._mem_mgr = mem_mgr  # manager of shared memory
        self._mem_size = mem_size  # size of memory to allocate from system
        if mem_mgr is None and self._mem_size > 0:
            self._mem_mgr = SharedMemoryMgr(self._mem_size)

    def get_mem_mgr(self):
        """ get shared memory manager """
        return self._mem_mgr

    def put(self, data, meta=None, lower_bd=None, buffer=None):
        """ put an element which may contains two parts
            one field is 'data' which will transmit usinig shared memory,
            and the other field 'meta' will trasmit using 'Queue'

        Args:
            @data (str): field used to store large size data
            @meta (str or pickable object): other fields of this element
            @lower_bd (int): size for this buffer should be greater than

        Returns:
            True if succeed, otherwise False
        """
        if data is None:
            assert buffer is None, "data and buffer is not consistent"
            self._inner_queue.put([data, meta])
            return True

        assert buffer is None or isinstance(buffer, SharedBuffer), \
            "FATAL: invalid type of buffer"

        assert type(data) is str, 'invalid type of input data'
        meta = cPickle.dumps(meta, -1)
        meta_len = len(meta)
        assert meta_len <= self.s_meta_size, \
            'too large meta object[%d]' % (meta_len)
        sz = len(data) + meta_len
        if lower_bd is not None and sz < lower_bd:
            sz = lower_bd

        if buffer is None:
            buffer = self._mem_mgr.malloc(sz + self.s_meta_size)
        else:
            assert buffer.capacity(
            ) >= sz, "not enough capacity for this buffer"
            buffer.resize(0)

        buffer.put(data + meta)
        self._inner_queue.put([len(meta), buffer.dump()])
        return True

    def get(self, ret_buff=False):
        """ get an element from this queue, and the shared buffer will be freed
            if ret_buff is False, otherwise the owner should free it explicitly

        Args:
            @ret_buff (bool): whether return the object wrapped in SharedBuffer

        Returns:
            tuple of an element or just data if not meta
            eg: (data, meta), data
        """
        pkg = list(self._inner_queue.get())
        if pkg[0] is None:
            return pkg
        else:
            meta_len, buff_info = pkg

        buffer = SharedBuffer.load(buff_info)
        if meta_len > 0:
            meta = buffer.get(-meta_len, meta_len)
            meta = cPickle.loads(meta)
            buffer.resize(buffer.size() - meta_len)

        if not ret_buff:
            data = buffer.get()
            buffer.free()
        else:
            data = buffer

        if meta_len > 0:
            return [data, meta]
        else:
            return data


class XmapEndSignal(ValueError):
    """ XmapEndSignal
    """

    def __init__(self, errmsg='', errno=-1):
        """ init
        """
        super(XmapEndSignal, self).__init__(errmsg)
        self._errmsg = errmsg
        self._errno = errno

    def get_errmsg(self):
        """ get errmsg
        """
        return self._errmsg

    def get_errno(self):
        """ get errno
        """
        return self._errno


class XmapProcessError(ValueError):
    """ XmapProcessError
    """
    pass


def get_worker(use_process, **kwargs):
    """ get_worker
    """
    if use_process:
        from multiprocessing import Process
        return Process(**kwargs)
    else:
        from threading import Thread
        return Thread(**kwargs)


# define a worker to handle samples from in_queue by mapper
# and put mapped samples into out_queue
def handle_worker(in_queue, out_queue, mapper):
    """ handle_worker
    """
    buff, meta = in_queue.get(ret_buff=True)
    end = XmapEndSignal('ok', 0)
    while not isinstance(meta, XmapEndSignal):
        try:
            param = concate_as_tuple(buff.get(), meta)
            res = mapper(param)
            out_queue.put(res[0], meta=res[1:], buffer=buff)
            buff = None
            buff, meta = in_queue.get(ret_buff=True)
        except Exception as e:
            stack_info = traceback.format_exc()
            end = XmapEndSignal(stack_info, -1)
            logger.warn(stack_info)
            if buff is not None:
                buff.free()
            break

    in_queue.put(None, meta=end)
    out_queue.put(None, meta=end)


def _xmap_reader(reader, mapper, data_size, worker_num=16, buffer_size=1000,\
        use_process=True, mem_size=None, pre_feed=None, **kwargs):
    """
    Executing 'mapper' using multiple processes with sample from 'reader',
    the processed sample will be yield out from the returned '_reader'.

    notes:
        * The inputs come from 'reader', and each input is tuple like this 
            (image, label, xxx), the 'image' must be a instance of str.
        * The outputs come from 'mapper(input)', and each output is also 
            a tuple like this (result_image, label, xxx), the 'result_image' 
            must be a instance of str.

    task of main process:
        1, create shared-memory-based input/output queues for communication
        2, init all workers and start them using these queues
        3, create a iterator of 'reader', and feed 'pre_feed' samples to input queue
        4, get next sample from this iterator
        5, if the sample is data then put it to input queue, otherwise go to step 9
        6, get a processed sample from output queue
        7, yield out this result
        8, goto step 4
        9, put exit-signal objects to input queue to notify worker processes exit
        10, wait workers' exit-signal object from output queue, then return

    task of worker process:
        1, get next sample from input queue
        2, if the sample is data then execute 'mapper' using this sample, 
            eg: mapper(sample), otherwise go to step 4
        3, put this processed sample to output queue
        4, put exit-signal object to input and output queue, then return

    Args:
        @reader (callable): iterator maker for the sample data
        @mapper (callable): a function to process a sample
        @data_size (int): result image size in bytes
        @worker_num (int): number of workers to execute 'mapper(sample)'
        @buffer_size (int): queue size for input and output queues
        @mem_size (int): the size of allocated shared memory used in queues
        @pre_feed (int): how many to feed samples to workers before fetching result

    Returns:
        @reader (callable): iterator maker for the processed sample data
    """
    logger.debug('not used params in shared_xmap.xmap_reader:[%s]' %
                 (str(kwargs)))

    def _feed_sample(iter, inq, num=1):
        # feed one sample to worker
        try:
            for i in xrange(num):
                sample = iter.next()
                assert isinstance(sample,
                                  tuple), "reader's result must be tuple"
                inq.put(sample[0], meta=sample[1:], lower_bd=data_size)
            return None
        except StopIteration as e:
            end = XmapEndSignal(errmsg='ok', errno=0)
            return end
        except Exception as e:
            stack_info = traceback.format_exc()
            end = XmapEndSignal(stack_info, -1)
            return end

    def _fetch_result(outq):
        # fetch one processed sample
        buf, meta = outq.get(ret_buff=True)
        if isinstance(meta, XmapEndSignal):
            return meta

        data = buf.get()
        buf.free()
        #if data and ret_imgshape:
        #    data = np.frombuffer(data, dtype=ret_imgtype)
        #    data = data.reshape(ret_imgshape)

        return concate_as_tuple(data, meta)

    pre_feed = buffer_size / 2 if pre_feed is None else pre_feed
    assert pre_feed <= buffer_size, 'invalid pre_feed[%d]' % (pre_feed)
    if mem_size is None:
        mem_size = 1 * 1024 * 1024 * 1024  # allocate at least 1G

    if mem_size < 2 * buffer_size * data_size:
        mem_size = 2 * buffer_size * data_size

    def _reader():
        in_queue = FastQueue(buffer_size,\
            mem_size=mem_size, use_process=use_process)
        out_queue = FastQueue(buffer_size,\
            mem_mgr=in_queue.get_mem_mgr(), use_process=use_process)

        # start handle_workers to do cpu-intensive tasks
        target = handle_worker
        args = (in_queue, out_queue, mapper)
        workers = []
        for i in xrange(worker_num):
            worker = get_worker(
                use_process=use_process, target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        # prefeed a bunch of samples to workers
        iter = reader()
        end = _feed_sample(iter, in_queue, num=pre_feed)

        finished = 0  # finished workers
        while end is None:
            # fetch one processed sample
            result = _fetch_result(out_queue)
            if isinstance(result, XmapEndSignal):
                finished += 1
                end = result
                break
            else:
                yield result
            end = _feed_sample(iter, in_queue)

        # no more task or failure happened, so notify all workers to stop
        for i in xrange(worker_num - finished):
            in_queue.put(None, end)

        if end.get_errno() != 0:
            raise XmapProcessError('faield to finish fast_xmap'\
                ' with error[errno:%d, errmsg:%s]' \
                % (end.get_errno(), end.get_errmsg()))

        while finished < worker_num:
            result = _fetch_result(out_queue)
            if isinstance(result, XmapEndSignal):
                finished += 1
            else:
                yield result

    return _reader


def xmap_reader(reader,
                mapper,
                ret_imgshape=(3, 224, 224),
                ret_imgtype='uint8',
                **kwargs):
    """
    Apply 'mapper' function to every sample from 'reader', 
    the 'mapper' should accept a tuple (image_str, label) 
    as it's arguments and return a another tuple (image_ndarray, label), 
    and will return the mapped reader

    notes:
        * The inputs come from 'reader', and each input is a tuple 
            (image, label, xxx), the 'image' must be a instance of str.
        * The outputs come from 'mapper(input)', and each output is also 
            a tuple like this (result_image, label, xxx), the 'result_image' 
            must be a np.ndarray with shape 'ret_imgshape' and type 'ret_imgtype'


    Args:
        @ret_imgshape (tuple): the shape of first element from mapper's result
                                which is the processed image with type of 
                                numpy.ndarray
        @ret_imgtype (str): the type of ndarray, eg:'uint8', 'int32', 'float32'

    Returns:
        @reader (callable): iterator maker for the processed sample data
    """

    data_size = reduce(lambda x, y: x * y, ret_imgshape)
    if ret_imgtype.endswith('8'):
        data_size *= 1
    elif ret_imgtype.endswith('32'):
        data_size *= 4
    elif ret_imgtype.endswith('64'):
        data_size *= 8
    else:
        raise ValueError('invalid ret_imgtype[%s]' % (ret_imgtype))

    def _serializeable_mapper(sample):
        result = mapper(sample)
        img = result[0]
        if img is not None and ret_imgshape:
            assert isinstance(img, np.ndarray), \
                'the type of first field from mapper must be numpy.ndarray'
            assert img.shape == ret_imgshape, 'invalid shape[real:%s,expected:%s]'\
                ' of processed image' % (str(img.shape), str(ret_imgshape))
            assert img.dtype == ret_imgtype, 'invalid type[real:%s,expected:%s]'\
                ' of processed image from worker' % (img.dtype, ret_imgtype)
            img = img.tobytes()
            assert len(img) == data_size, 'invalid size[real:%d, expect:%d] of '\
                'processed image from worker' % (len(img), data_size)
        return concate_as_tuple(img, result[1:])

    rd = _xmap_reader(reader, _serializeable_mapper, data_size, **kwargs)

    def _reader():
        for image, label in rd():
            image = np.frombuffer(image, dtype=ret_imgtype)
            image = image.reshape(ret_imgshape)

            yield (image, label)

    return _reader
