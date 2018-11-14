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

__all__ = [
    'map_readers', 'buffered', 'compose', 'chain', 'shuffle','xmap_readers'
]

from threading import Thread
import subprocess

from Queue import Queue
import itertools
import random
import zlib

import logging
import traceback

logger = logging.getLogger(__name__)
def map_readers(func, *readers):
    """
    Creates a data reader that outputs return value of function using
    output of each data readers as arguments.
    :param func: function to use. The type of func should be (Sample) => Sample
    :type: callable
    :param readers: readers whose outputs will be used as arguments of func.
    :return: the created data reader.
    :rtype: callable
    """

    def reader():
        """ reader """
        rs = []
        for r in readers:
            rs.append(r())
        for e in itertools.imap(func, *rs):
            yield e

    return reader


class ReaderEndSignal(object):
    """ ReaderEndSignal """
    pass


def shuffle(reader, buf_size):
    """
    Creates a data reader whose data output is shuffled.
    Output from the iterator that created by original reader will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.
    :param reader: the original reader whose output will be shuffled.
    :type reader: callable
    :param buf_size: shuffle buffer size
    :type buf_size: int
    :return: the new reader whose output is shuffled.
    :rtype: callable
    """

    assert buf_size > 0, "invalid buf_size in shuffle" % (buf_size)
    end = ReaderEndSignal()
    def _start_prefetch(rd, inq, outq):
        def _fetcher(rd, inq, outq):
            for i, d in enumerate(rd()):
                if i >= inq.maxsize:
                    if isinstance(inq.get(), ReaderEndSignal):
                        break

                outq.put(d)
            outq.put(end)

        p = Thread(target=_fetcher, args=(rd, inq, outq))
        p.daemon = True
        p.start()

    def _reader():
        token_q = Queue(buf_size)
        data_q = Queue(buf_size)
        _start_prefetch(reader, token_q, data_q)

        stopped = False
        buf = []
        yield_buf = []
        while True:
            if not stopped:
                e = data_q.get()
                if not isinstance(e, ReaderEndSignal):
                    buf.append(e)
                    if len(buf) >= buf_size:
                        random.shuffle(buf)
                        yield_buf += buf
                        buf = []
                else:
                    stopped = True
                    if buf:
                        yield_buf += buf
                        buf = []
                
            if len(yield_buf) > 0:
                yield yield_buf.pop(0)
                token_q.put(True) #need more
            elif stopped:
                break

    return _reader


def chain(*readers):
    """
    Creates a data reader whose output is the outputs of input data
    readers chained together.
    If input readers output following data entries:
    [0, 0, 0]
    [1, 1, 1]
    [2, 2, 2]
    The chained reader will output:
    [0, 0, 0, 1, 1, 1, 2, 2, 2]
    :param readers: input readers.
    :return: the new data reader.
    :rtype: callable
    """

    def reader():
        """ reader
        """
        rs = []
        for r in readers:
            rs.append(r())

        for e in itertools.chain(*rs):
            yield e

    return reader


class DecoratorError(ValueError):
    """ DecoratorError
    """
    pass


def buffered(reader, size):
    """
    Creates a buffered data reader.
    The buffered data reader will read and save data entries into a
    buffer. Reading from the buffered data reader will proceed as long
    as the buffer is not empty.
    :param reader: the data reader to read from.
    :type reader: callable
    :param size: max buffer size.
    :type size: int
    :returns: the buffered data reader.
    """
    assert size > 0, "invalid param size[%d] for buffered" % (size)

    end = ReaderEndSignal()
    def read_worker(r, q):
        """ read_reader """
        try:
            for d in r:
                q.put(d)
            q.put(end)
        except Exception as e:
            stack_info = traceback.format_exc()
            q.put(end)
            raise DecoratorError(stack_info)

    def data_reader():
        """ data_reader """
        r = reader()
        q = Queue(maxsize=size)
        t = Thread(
            target=read_worker, args=(
                r,
                q, ))
        t.daemon = True
        t.start()
        e = q.get()
        while e != end:
            yield e
            e = q.get()

    return data_reader


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


def get_queue(queue_cap, use_process):
    """ get_queue
    """
    if use_process:
        from multiprocessing import Queue
    else:
        from Queue import Queue
    return Queue(queue_cap)


def get_worker(use_process, **kwargs):
    """ get_worker
    """
    if use_process:
        from multiprocessing import Process
        return Process(**kwargs)
    else:
        return Thread(**kwargs)


# define a worker to read samples from reader to in_queue
def read_worker(reader, in_queue):
    """ read_worker
    """
    end = XmapEndSignal(errmsg='ok', errno=0)
    try:
        for i in reader():
            in_queue.put(i)

        in_queue.put(end)
    except Exception as e:
        stack_info = traceback.format_exc()
        in_queue.put(XmapEndSignal(stack_info, -1))


# define a worker to read samples from reader to in_queue with order flag
def order_read_worker(reader, in_queue):
    """ order_read_worker
    """
    try:
        in_order = 0
        for i in reader():
            in_queue.put((in_order, i))
            in_order += 1
    except Exception as e:
        stack_info = traceback.format_exc()
        end = XmapEndSignal(stack_info, -1)

    in_queue.put(end)


# define a worker to handle samples from in_queue by mapper
# and put mapped samples into out_queue
def handle_worker(in_queue, out_queue, mapper, flatmap):
    """ handle_worker
    """
    sample = in_queue.get()
    while not isinstance(sample, XmapEndSignal):
        try:
            if not flatmap:
                r = mapper(sample)
                out_queue.put(r)
            else:
                for r in mapper(sample):
                    out_queue.put(r)
            sample = in_queue.get()
        except Exception as e:
            stack_info = traceback.format_exc()
            sample = XmapEndSignal(stack_info, -1)

    end = sample
    in_queue.put(end)
    out_queue.put(end)


# define a worker to handle samples from in_queue by mapper
# and put mapped samples into out_queue by order
def order_handle_worker(in_queue, out_queue, mapper, out_order):
    """ order_handle_worker
    """
    ins = in_queue.get()
    while not isinstance(ins, XmapEndSignal):
        try:
            order, sample = ins
            r = mapper(sample)
            while order != out_order[0]:
                pass
            out_queue.put(r)
            out_order[0] += 1
            ins = in_queue.get()
        except Exception as e:
            stack_info = traceback.format_exc()
            ins = XmapEndSignal(stack_info, -1)

    end = ins
    in_queue.put(end)
    out_queue.put(end)


def xmap_readers(reader, mapper=None, process_num=16, buffer_size=1000,\
        order=False, use_process=False, flatmap=False):
    """
    Use multiprocess to map samples from reader by a mapper defined by user.
    And this function contains a buffered decorator.
    :param mapper:  a function to map sample.
    :type mapper: callable
    :param reader: the data reader to read from
    :type reader: callable
    :param process_num: process number to handle original sample
    :type process_num: int
    :param buffer_size: max buffer size
    :type buffer_size: int
    :param order: keep the order of reader
    :type order: bool
    :return: the decarated reader
    :rtype: callable
    """
    def xreader():
        """ xreader
        """
        #in_queue = Queue(buffer_size)
        #out_queue = Queue(buffer_size)
        in_queue = get_queue(buffer_size, use_process)
        out_queue = get_queue(buffer_size, use_process)
        out_order = [0]
        # start a read worker in a thread
        target = order_read_worker if order else read_worker
        p = Thread(target=target, args=(reader, in_queue))
        #p = get_worker(target=target, args=(reader, in_queue))
        #p = Process(target=target, args=(reader, in_queue))
        p.daemon = True
        p.start()
        # start several handle_workers
        target = order_handle_worker if order else handle_worker
        args = (in_queue, out_queue, mapper, out_order, flatmap) if order else (
            in_queue, out_queue, mapper, flatmap)
        workers = []
        for i in xrange(process_num):
            #worker = Thread(target=target, args=args)
            worker = get_worker(use_process=use_process, target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        errexp = None
        sample = out_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = out_queue.get()

        if sample.get_errno() != 0:
            errexp = sample

        finish = 1
        while finish < process_num:
            sample = out_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
                if sample.get_errno() != 0 and errexp is None:
                    errexp = sample
            else:
                yield sample

        if errexp is not None:
            raise errexp

    return xreader
