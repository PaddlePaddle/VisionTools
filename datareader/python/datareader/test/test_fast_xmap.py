import os
import time
import math
import random
import unittest
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
if path not in sys.path:
    sys.path.insert(0, path)

import datareader.operators as ops


def get_ops(img_size=224, normalize=False):
    """ a image mapper for training data
    """

    ops.default_class = 'opencv'
    img_ops = [ops.DecodeImage()]
    img_ops += [ops.RotateImage(10, rand=True)]
    img_ops += [ops.RandCropImage(img_size)]
    img_ops += [ops.RandFlipImage()]
    img_ops += [ops.ToCHWImage(op_class='pil')]

    if normalize:
        img_ops += [ops.NormalizeImage()]

    return img_ops


class TestXmap(unittest.TestCase):
    """Test cases for shared_memory.shared_xmap
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        with open('test_img.jpg', 'rb') as f:
            cls.img_data = f.read()

    @classmethod
    def tearDownClass(cls):
        """ tear down"""
        pass

    def test_xmap(self):
        """ test_fast_xmap
        """
        data_num = 10000

        def _data_source():
            for i in xrange(data_num):
                yield (self.img_data, i)

        worker_num = 10
        use_process = True
        use_sharedmem = True
        xmapper = ops.build(
            get_ops(),
            use_process=use_process,
            worker_num=worker_num,
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

        print('processed total %d samples with qps:%d' \
                % (total, total / (time.time() - start_ts)))


if __name__ == '__main__':
    unittest.main()
