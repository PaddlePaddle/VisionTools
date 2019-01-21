import os
import time
import math
import random
import unittest
import sys
import logging
import PIL
import numpy as np

logging.basicConfig(level=logging.DEBUG)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
if path not in sys.path:
    sys.path.insert(0, path)

from datareader.transformer.pytransformer import PyProcessor
lua_ops = {
    'decode': """
        local cv = require("luac_cv")
        local basic = require("luac_basic")

        function lua_main(sample)
            local img = sample[1]
            local dec = cv.imdecode(img, 1)

            local result = cv.Mat()
            cv.cvtColor(dec, result, cv.COLOR_BGR2RGB, 0)
            return {result, sample[2]}
        end
    """,
    'resize_224': """
        local cv = require("luac_cv")
        local basic = require("luac_basic")

        function lua_main(sample, str_sample)
            local img = sample[1]
            -- local label = basic.mat2str(sample[2])
            local dec = cv.imdecode(img, 1)
            local resizeimg = cv.Mat()
            cv.resize(dec, resizeimg, cv.Size(224, 224), 0, 0, cv.INTER_LINEAR)

            local rgb_img = cv.Mat()
            cv.cvtColor(resizeimg, rgb_img, cv.COLOR_BGR2RGB, 0)

            return {rgb_img, sample[2]}
        end
    """,
}


class TestPyTransformer(unittest.TestCase):
    """Test cases for shared_memory.fast_xmap
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        work_dir = os.path.dirname(os.path.realpath(__file__))
        cls.test_jpg = os.path.join(work_dir, 'test.jpg')
        with open(cls.test_jpg, 'rb') as f:
            cls.img_data = f.read()

    @classmethod
    def tearDownClass(cls):
        """ tear down"""
        pass

    def test_decode(self):
        """ test decode
        """
        img = self.img_data
        p = PyProcessor()
        p.decode(to_rgb=True)
        decoded = p(img)
        self.assertTrue(type(decoded), np.ndarray)
        self.assertEqual(len(decoded.shape), 3)

    def test_resize(self):
        """ test decode
        """
        img = self.img_data
        p = PyProcessor()
        w = 220
        h = 214
        p.decode(to_rgb=True).resize(w, h)
        result = p(img)
        self.assertTrue(type(result), np.ndarray)
        self.assertEqual(result.shape, (h, w, 3))

    def test_reset(self):
        """ test reset
        """
        img = self.img_data
        proc = PyProcessor()
        proc.decode(to_rgb=True)
        result = proc(img)
        self.assertTrue(type(result), np.ndarray)
        proc.reset()

        proc.decode(to_rgb=True)
        decoded = proc(img)
        self.assertEqual(decoded.shape, result.shape)

    def test_lua_decode(self):
        """ test lua decode
        """
        img = self.img_data
        proc = PyProcessor()
        proc.lua(lua_code=lua_ops['decode'])
        result, label = proc(img, '123')
        self.assertEqual(label, '123')
        self.assertEqual(len(result.shape), 3)

    def test_decode_diff(self):
        """ test diff
        """
        img = self.img_data
        proc = PyProcessor()
        proc.lua(lua_code=lua_ops['decode'], tochw=False)
        lua_result, _ = proc(img, '')

        proc.reset()
        proc.decode(to_rgb=True)
        result, _ = proc(img, '')
        self.assertEqual(np.sum(np.abs(lua_result - result)), 0)

    def test_lua_resize(self):
        """ test lua resize
        """
        img = self.img_data
        proc = PyProcessor()

        proc.lua(lua_code=lua_ops['resize_224'], tochw=True)
        lua_result, _ = proc(img, '')
        self.assertEqual(lua_result.shape, (3, 224, 224))


if __name__ == '__main__':
    unittest.main()
