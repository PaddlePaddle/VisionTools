import os
import time
import math
import random
import unittest
import sys
import logging

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
if path not in sys.path:
    sys.path.insert(0, path)

from shared_memory.sharedmemory import SharedMemoryMgr


class TestSharedMemory(unittest.TestCase):
    """Test cases for shared_memory.sharedmemory
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

    def test_malloc_and_free(self):
        """ test malloc
        """
        mgr = SharedMemoryMgr(capacity=1025)
        length = 1024
        buff = mgr.malloc(length)

        self.assertEqual(buff.capacity(), length)
        self.assertTrue(buff.free())
        self.assertFalse(buff.free())

        buff = mgr.malloc(1)
        self.assertEqual(buff.capacity(), length)
        self.assertTrue(buff.free())

    def test_put_and_get(self):
        """ test free
        """
        mgr = SharedMemoryMgr(capacity=1025)
        length = 1024
        buff = mgr.malloc(length)
        data = 'hello, shared memory'

        buff.put(data)

        new_data = buff.get()
        self.assertEqual(new_data, data)

        buff.put('hi', override=True)

        new_data = buff.get()
        self.assertEqual(new_data, 'hi')

        self.assertTrue(buff.free())


if __name__ == '__main__':
    unittest.main()
