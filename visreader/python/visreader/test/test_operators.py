import os
import time
import math
import random
import unittest
import sys
import logging
import PIL
import numpy as np

import set_env
import visreader.operators as ops

logging.basicConfig(level=logging.INFO)


def get_ops(img_size=224, op_class='pil', normalize=True):
    """ a image mapper for training data
    """
    img_ops = [ops.DecodeImage(op_class=op_class)]
    #if op_class == 'pil':
    #    img_ops += [ops.RandDistortColor(op_class='pil')] # only exist in pil
    img_ops += [ops.RotateImage(10, rand=True, op_class=op_class)]
    img_ops += [ops.RandCropImage(img_size, op_class=op_class)]
    img_ops += [ops.RandFlipImage(op_class=op_class)]
    img_ops += [ops.ToCHWImage(op_class=op_class)]
    if normalize:
        img_ops += [ops.NormalizeImage(op_class=op_class)]

    return img_ops


def run_ops(op_list, img):
    """ run operators one by one with image 'img'
    """
    for o in op_list:
        img = o(img)
    return img


class TestOps(unittest.TestCase):
    """Test cases for shared_memory.fast_xmap
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        work_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(work_dir, 'test.jpg'), 'rb') as f:
            cls.img_data = f.read()

    @classmethod
    def tearDownClass(cls):
        """ tear down"""
        pass

    def test_decode(self):
        """ test decode
        """
        img = self.img_data

        op_class = 'pil'
        pil_data = run_ops([ops.DecodeImage(op_class=op_class)], img)
        self.assertEqual(type(pil_data), PIL.JpegImagePlugin.JpegImageFile)
        pil_data = np.array(pil_data)

        op_class = 'opencv'
        opencv_data = run_ops([ops.DecodeImage(op_class=op_class)], img)
        self.assertEqual(type(opencv_data), np.ndarray)

        self.assertEqual(pil_data.shape, opencv_data.shape)

    def test_resize(self):
        """ test resize
        """
        img = self.img_data

        op_class = 'pil'
        img_size = (200, 100)
        pil_data = run_ops(
            [
                ops.DecodeImage(op_class=op_class), ops.ResizeImage(
                    op_class=op_class, size=img_size)
            ],
            img)
        self.assertEqual(type(pil_data), PIL.Image.Image)
        pil_data = np.array(pil_data)

        op_class = 'opencv'
        opencv_data = run_ops(
            [
                ops.DecodeImage(op_class=op_class), ops.ResizeImage(
                    op_class=op_class, size=img_size)
            ],
            img)
        self.assertEqual(type(opencv_data), np.ndarray)

        self.assertEqual(pil_data.shape, opencv_data.shape)
        self.assertEqual(pil_data.shape, (100, 200, 3))

    def test_crop(self):
        """ test crop
        """
        img = self.img_data

        img_size = (200, 100)
        crop_size = (50, 20)
        op_class = 'pil'
        pil_data = run_ops(
            [
                ops.DecodeImage(op_class=op_class), ops.ResizeImage(
                    size=img_size, op_class=op_class), ops.CropImage(
                        crop_size, op_class=op_class)
            ],
            img)
        self.assertEqual(type(pil_data), PIL.Image.Image)
        pil_data = np.array(pil_data)

        op_class = 'opencv'
        opencv_data = run_ops(
            [
                ops.DecodeImage(op_class=op_class), ops.ResizeImage(
                    size=img_size, op_class=op_class), ops.CropImage(
                        crop_size, op_class=op_class)
            ],
            img)
        self.assertEqual(type(opencv_data), np.ndarray)

        self.assertEqual(pil_data.shape, opencv_data.shape)
        self.assertEqual(pil_data.shape, (crop_size[1], crop_size[0], 3))

    def test_randcrop(self):
        """ test random crop
        """
        img = self.img_data

        img_size = (200, 100)
        crop_size = (50, 20)
        op_class = 'pil'
        pil_data = run_ops(
            [
                ops.DecodeImage(op_class=op_class), ops.ResizeImage(
                    size=img_size, op_class=op_class), ops.RandCropImage(
                        crop_size, op_class=op_class)
            ],
            img)
        self.assertEqual(type(pil_data), PIL.Image.Image)
        pil_data = np.array(pil_data)

        op_class = 'opencv'
        opencv_data = run_ops(
            [
                ops.DecodeImage(op_class=op_class), ops.ResizeImage(
                    size=img_size, op_class=op_class), ops.RandCropImage(
                        crop_size, op_class=op_class)
            ],
            img)
        self.assertEqual(type(opencv_data), np.ndarray)

        self.assertEqual(pil_data.shape, opencv_data.shape)
        self.assertEqual(pil_data.shape, (crop_size[1], crop_size[0], 3))

    def test_ops(self):
        """ test operators
        """
        img = self.img_data

        op_class = 'pil'
        pil_data = run_ops(get_ops(op_class=op_class), img)

        op_class = 'opencv'
        opencv_data = run_ops(get_ops(op_class=op_class), img)
        self.assertEqual(type(opencv_data), np.ndarray)

        self.assertEqual(pil_data.shape, opencv_data.shape)
        self.assertEqual(pil_data.shape, (3, 224, 224))

    def test_xmap(self):
        """ test_fast_xmap
        """
        data_num = 2000

        def _data_source():
            for i in xrange(data_num):
                yield (self.img_data, i)

        worker_num = 10
        use_process = True
        use_sharedmem = True
        cpp_xmap = False
        xmapper = ops.build(
            get_ops(normalize=False),
            use_process=use_process,
            worker_num=worker_num,
            cpp_xmap=cpp_xmap,
            use_sharedmem=use_sharedmem)
        rd = xmapper(_data_source)

        start_ts = time.time()
        prev = start_ts
        ct = 0
        total = 0
        for img, label in rd():
            self.assertEqual(img.shape, (3, 224, 224))
            cost = time.time() - prev
            ct += 1
            total += 1
            if cost >= 1:
                print('processed %d/%d samples with qps:%d' \
                    % (ct, total, total / (time.time() - start_ts)))
                prev = time.time()
                ct = 0

        self.assertEqual(total, data_num)
        print('processed total %d samples with qps:%d' \
                % (total, total / (time.time() - start_ts)))


if __name__ == '__main__':
    unittest.main()
