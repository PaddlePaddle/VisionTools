import os
import time
import unittest
import sys
import logging
import thread

import set_env
import visreader
from visreader.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)


def make_reader(num):
    """ make a reader to yield fake samples
    """

    def _reader():
        for i in range(num):
            yield i

    return _reader


class TestPipeline(unittest.TestCase):
    """Test cases for visreader.pipeline.Pipeline
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_map(self):
        """ test map
        """
        num = 10
        p = Pipeline(make_reader(num))
        p.map(lambda r: 2 * r)
        rd = p.reader()
        for _ in range(3):
            ct = 0
            for i, data in enumerate(rd()):
                ct += 1
                self.assertEqual(i * 2, data)
            self.assertEqual(ct, num)

        def _reader_mapper(rd):
            def _reader():
                for data in rd():
                    yield 2 * data

            return _reader

        p = Pipeline(make_reader(num))
        p.map(reader_mapper=_reader_mapper)
        rd = p.reader()
        for _ in range(3):
            ct = 0
            for i, data in enumerate(rd()):
                ct += 1
                self.assertEqual(i * 2, data)
            self.assertEqual(ct, num)

    def test_filter(self):
        """ test filter
        """
        num = 10
        p = Pipeline(make_reader(num))
        p.filter(lambda r: r % 2 == 0)
        rd = p.reader()
        for i, data in enumerate(rd()):
            self.assertEqual(i * 2, data)

    def test_xmap(self):
        """ test xmap
        """
        num = 10
        p = Pipeline(make_reader(num))

        def _map(r):
            time.sleep(0.1)
            return (thread.get_ident(), 2 * r)

        # test 1 thread
        rd = p.xmap(_map, 1).reader()
        threads = {}
        for i, data in enumerate(rd()):
            threads[data[0]] = True
            self.assertEqual(i * 2, data[1])
        self.assertEqual(1, len(threads.keys()))

        # test 2 threads
        p.reset(make_reader(num))
        rd = p.xmap(_map, 2).reader()
        threads = {}
        results = {2 * k: k for k in range(num)}
        for i, data in enumerate(rd()):
            threads[data[0]] = True
            self.assertTrue(data[1] in results)
            del results[data[1]]
        self.assertEqual(2, len(threads.keys()))

        # test ordered xmap with 2 threads
        p.reset(make_reader(num))
        rd = p.xmap(_map, 2, 10, order=True).reader()
        threads = {}
        results = {2 * k: k for k in range(num)}
        for i, data in enumerate(rd()):
            threads[data[0]] = True
            self.assertEqual(i * 2, data[1])
            del results[data[1]]
        self.assertEqual(2, len(threads.keys()))

    def test_shuffle(self):
        """ test shuffle
        """
        num = 10
        p = Pipeline(make_reader(num))
        rd = p.shuffle(5).reader()

        results = [[], []]
        sum = 0
        for i, data in enumerate(rd()):
            sum += data
            results[i // 5].append(data)

        self.assertEqual(45, sum)
        for i in results[0]:
            for j in results[1]:
                self.assertGreater(j, i)

    def test_batch(self):
        """ test batch
        """
        num = 10
        p = Pipeline(make_reader(num))
        rd = p.batch(2).reader()

        sum = 0
        for i, data in enumerate(rd()):
            self.assertEqual(2, len(data))
            sum += reduce(lambda x, y: x + y, data)

        self.assertEqual(45, sum)

    def test_buffered(self):
        """ test buffered
        """
        num = 10
        p = Pipeline(make_reader(num))
        rd = p.buffered(2).reader()

        for i, data in enumerate(rd()):
            self.assertEqual(i, data)

    def test_cache(self):
        """ test cache
        """
        num = 10
        p = Pipeline(make_reader(num))

        def _delayed_map(r):
            time.sleep(0.2)
            return r

        rd = p.map(_delayed_map).cache().reader()
        start_ts = time.time()
        for i, data in enumerate(rd()):
            self.assertEqual(i, data)
        cost = time.time() - start_ts
        self.assertGreater(cost, 1.0)

        start_ts = time.time()
        for i, data in enumerate(rd()):
            self.assertEqual(i, data)
        cost = time.time() - start_ts
        self.assertLess(cost, 1.0)

    def test_echo(self):
        """ test echo
        """
        num = 10
        p = Pipeline(make_reader(num))
        rd = p.echo(2).reader()
        for i, data in enumerate(rd()):
            if i % 2 == 0:
                expect = i // 2
            else:
                expect = (i - 1) // 2
            self.assertEqual(expect, data)


if __name__ == '__main__':
    unittest.main()

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
