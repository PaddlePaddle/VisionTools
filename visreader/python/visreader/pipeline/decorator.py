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
    'map_readers', 'buffered', 'compose', 'chain', 'shuffle', 'xmap_reader',
    'Xmap'
]

from threading import Thread
import subprocess
import weakref
import time
from multiprocessing.util import Finalize
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
                token_q.put(True)  #need more
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


def get_queue(queue_cap,
              use_process,
              shared_memsize=None,
              shared_pagesize=None):
    """ get_queue
    """
    if shared_memsize is not None:
        from ..shared_queue import SharedQueue
        return SharedQueue(queue_cap, \
            memsize=shared_memsize, pagesize=shared_pagesize)
    elif use_process:
        from multiprocessing import Queue
        return Queue(queue_cap)
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


# define a worker to handle samples from in_queue by mapper
# and put mapped samples into out_queue
def handle_worker(in_queue, out_queue, mapper, order):
    """ handle_worker
    """
    sample = in_queue.get()
    while not isinstance(sample, XmapEndSignal):
        try:
            if not order:
                result = mapper(sample)
                out_queue.put(result)
            else:
                data, id = sample
                result = mapper(data)
                out_queue.put((result, id))
            sample = in_queue.get()
        except Exception as e:
            stack_info = traceback.format_exc()
            sample = XmapEndSignal(stack_info, -1)

    end = sample
    in_queue.put(end)
    out_queue.put(end)


class XMappedReader(object):
    def __init__(self, reader, mapper=None, worker_num=16, \
            buffer_size=1000, use_process=False, \
            shared_memsize=None, shared_pagesize=None, \
            order=False, pre_feed=None):
        logger.debug('create XMappedReader with shared_memsize[%s]' %
                     (str(shared_memsize)))

        assert buffer_size > 0, "invalid buffer_size[%d] in XMappedReader" \
            % (buffer_size)
        if pre_feed is None:
            pre_feed = 1 + buffer_size / 2

        self._pre_feed = pre_feed
        self._reader = reader
        self._inq, self._outq = self._init_queues(
            buffer_size, use_process, shared_memsize, shared_pagesize)

        self._order = order
        args = (self._inq, self._outq, mapper, order)
        self._workers = self._init_workers(worker_num, use_process, \
                            handle_worker, args)
        self._worker_num = len(self._workers)
        self._finished_workers = 0
        self._join_timeout = 3

    def _init_queues(self, buffer_size, use_process, shared_memsize,
                     shared_pagesize):
        inq = get_queue(buffer_size, use_process, shared_memsize,
                        shared_pagesize)
        outq = get_queue(buffer_size, use_process, shared_memsize,
                         shared_pagesize)
        return inq, outq

    def _init_workers(self, worker_num, use_process, target, args):
        workers = []
        for i in xrange(worker_num):
            worker = get_worker(
                use_process=use_process, target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()
        return workers

    def _feed_sample(self, iter, inq, feed_ctx, num=1):
        order = feed_ctx['order']
        id = feed_ctx['id']
        try:
            for i in xrange(num):
                sample = iter.next()
                if order:
                    sample = (sample, id)
                    id += 1
                inq.put(sample)

            feed_ctx['id'] = id
            return None
        except StopIteration as e:
            end = XmapEndSignal(errmsg='ok', errno=0)
            return end
        except Exception as e:
            stack_info = traceback.format_exc()
            logger.warn('failed to feed sample with stack info[%s]' %
                        (stack_info))
            end = XmapEndSignal(stack_info, -1)
            return end

    def __call__(self):
        """ xreader
        """
        iter = self._reader()
        inq = self._inq
        outq = self._outq
        pre_feed = self._pre_feed
        feed_ctx = {'id': 0, 'order': self._order}
        fetch_ctx = {'id': 0, 'results': {}}
        end = self._feed_sample(iter, inq, feed_ctx, num=pre_feed)
        self._finished_workers = 0
        while end is None:
            result = outq.get()
            if isinstance(result, XmapEndSignal):
                self._finished_workers += 1
                end = result
                break
            else:
                if not self._order:
                    yield result
                else:
                    sample, id = result
                    assert id not in fetch_ctx['results'], \
                        "duplicated id[%d] in order mode" % (id)
                    fetch_ctx['results'][id] = sample
                    if id in fetch_ctx['results']:
                        sample = fetch_ctx['results'][id]
                        del fetch_ctx['results'][id]
                        fetch_ctx['id'] += 1
                        yield sample

            end = self._feed_sample(iter, inq, feed_ctx)

        # no more task or failure happened, so notify all workers to exit
        if end.get_errno() != 0:
            raise XmapProcessError('faield to feed sample to other process'\
                ' with error[errno:%d, errmsg:%s]' \
                % (end.get_errno(), end.get_errmsg()))

        self._notify_exit(end)
        while self._finished_workers < self._worker_num:
            result = outq.get()
            if isinstance(result, XmapEndSignal):
                if result._errno != 0:
                    logger.warn('worker exit with error[errno:%d,errmsg:%s]' \
                        % (result._errno, result._errmsg))
                self._finished_workers += 1
            else:
                if not self._order:
                    yield result
                else:
                    sample, id = result
                    assert id not in fetch_ctx['results'], \
                        "duplicated id[%d] in order mode" % (id)
        while len(fetch_ctx['results'].keys()) > 0:
            id = fetch_ctx['id']
            if id in fetch_ctx['results']:
                sample = fetch_ctx['results'][id]
                del fetch_ctx['results'][id]
                fetch_ctx['id'] += 1
                yield sample
            else:
                logger.warn('not found result with id[%d]' % (id))
                break

    def _notify_exit(self, end=None):
        """ notify worker to finish it's task and exit
        """
        end = end if end is not None else XmapEndSignal('ok', 0)
        for i in xrange(self._worker_num - self._finished_workers):
            self._inq.put(end)

    def __del__(self):
        """ release all resources allocated by this instance
        """
        self._notify_exit()
        for i, w in enumerate(self._workers):
            w.join(self._join_timeout)
            if w.is_alive():
                logger.warn('worker[%d] still alive in XMappedReader' % (i))

        try:
            self._inq.release()
            self._outq.release()
        except Exception as e:
            pass


def xmap_reader(reader, mapper=None, worker_num=16, \
        buffer_size=1000, use_process=False, \
        use_sharedmem=None, shared_memsize=None, shared_pagesize=None, \
        order=False, pre_feed=None, **kwargs):
    """
    Use multiprocess to map samples from reader by a mapper defined by user.
    And this function contains a buffered decorator.

    Args:
        @mapper (callable function): a function to map sample.
        @reader (callable iterator): the data reader to read from
        @worker_num (int): process number to handle original sample
        @buffer_size (int): max number of samples to buffer
        @use_process (bool): whether use processes or threads as workers
        @shared_memsize (int): size of shared memory used in IPC 
        @shared_pagesize (int): page size of shared memory
        @order (bool): whether need to keep the order of processing samples
        @pre_feed (int): number of feeding samples before fetch the first result

    Returns:
        the decarated reader which yields mapped data from 'reader'
    """
    logger.debug('params in decorator.xmap_reader:[%s]' % (str(locals())))

    if use_sharedmem is False or shared_memsize is None:
        use_sharedmem = False
        shared_memsize = None
        shared_pagesize = None
    else:
        use_sharedmem = True
        assert use_process is True, 'sharedmemory mode can only be used '\
            'with "use_process" enabled'
        if shared_memsize is None:
            shared_memsize = 1 * 1024 * 1024 * 1024
        if shared_pagesize is None:
            shared_pagesize = 64 * 1024

    def _xreader():
        rd = XMappedReader(reader, mapper=mapper, worker_num=worker_num, \
                buffer_size=buffer_size, use_process=use_process, \
                shared_memsize=shared_memsize, shared_pagesize=shared_pagesize, \
                order=order, pre_feed=pre_feed)

        for i in rd():
            yield i

    return _xreader


class Xmap(object):
    def __init__(self, mapper, worker_num=16,\
        buffer_size=1000, order=False, \
        use_process=False, **kwargs):
        self.args = {}
        for k, v in locals().items():
            if k not in ['self', 'kwargs']:
                self.args[k] = v
        self.args.update(kwargs)

    def __call__(self, reader):
        return xmap_reader(reader, **self.args)
