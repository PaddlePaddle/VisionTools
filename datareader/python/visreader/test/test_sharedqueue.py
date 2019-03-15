import os
import time
import math
import random
import unittest
import sys
import logging
import numpy as np

from multiprocessing import Queue
from multiprocessing import Process
from threading import Thread

import set_env
import visreader
from visreader.shared_queue import SharedMemoryMgr
from visreader.shared_queue import SharedQueue

logging.basicConfig(level=logging.INFO)
SharedMemoryMgr.s_log_statis = True


class EndSignal(object):
    """ EndSignal """
    pass


class TestSharedQueue(unittest.TestCase):
    """Test cases for visreader.shared_queue
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """ tear down"""
        pass

    def test_sharedmem(self):
        mgr = SharedMemoryMgr(capacity=1024 * 1024, pagesize=1 * 1024)
        buf = mgr.malloc(10)
        buf.put('hello')

        self.assertEqual(buf.get(), 'hello')
        buf.free()

    def test_sharedqueue(self):
        sq = SharedQueue(maxsize=100)
        sq.put('hello')
        self.assertEqual(sq.get(), 'hello')

    def test_performance(self):
        data_num = 10000
        worker_num = 10

        test_data = np.random.randint(0, 10, 32 * 1024, dtype='int8')
        expect_sum = np.sum(test_data)

        def _data_source():
            for i in xrange(data_num):
                yield (test_data, i)

        end = EndSignal()

        def _processor(id, inq, outq):
            sample = inq.get()
            while not isinstance(sample, EndSignal):
                outq.put(
                    (sample[0], str(sample[1]) + ', processed by ' + str(id)))
                sample = inq.get()

            outq.put(end)

        qsize = 1000
        memsize = 10 * 1024 * 1024
        pagesize = 32 * 1024
        inqueue = SharedQueue(qsize, memsize=memsize, pagesize=pagesize)
        outqueue = SharedQueue(qsize, memsize=memsize, pagesize=pagesize)

        workers = []
        start_ts = time.time()
        for i in range(worker_num):
            p = Process(target=_processor, args=(i, inqueue, outqueue))
            p.daemon = True
            p.start()
            workers.append(p)

        ct = 0
        for i, data in enumerate(_data_source()):
            inqueue.put(data)
            if i < 10:
                continue

            data = outqueue.get()
            ct += 1
            if i % 1000 == 0:
                self.assertEqual(np.sum(data[0]), expect_sum)
                print('get sample_%d from subprocess: %s' % (i, data[1]))

        for i in range(worker_num):
            inqueue.put(end)

        finished_workers = 0
        while finished_workers < worker_num:
            data = outqueue.get()
            if isinstance(data, EndSignal):
                finished_workers += 1
            else:
                ct += 1

        print('total cost %dms to process %d samples using %d workers' \
            % (1000 * (time.time() - start_ts), data_num, worker_num))
        self.assertEqual(ct, data_num)
        self.assertEqual(finished_workers, worker_num)


if __name__ == '__main__':
    unittest.main()
