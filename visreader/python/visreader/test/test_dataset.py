import os
import time
import unittest
import sys
import logging

import set_env
import visreader
from visreader.pipeline import Dataset

logging.basicConfig(level=logging.INFO)


class TestDataset(unittest.TestCase):
    """Test cases for visreader.pipeline.Dataset
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        cls.uri = 'file:/' + os.path.abspath(__file__)

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_reader(self):
        """ test reader constructed by data from datarepo of aiflow
        """
        ds = Dataset.load(uri=self.uri, filetype='textfile')\
            .shuffle(100)\
            .map(lambda r: {'image': r, 'label': len(r)})

        rd = ds.reader()
        ct = 0
        #print("\ngo to fetch data from dataset[%s]:" % (self.uri))
        start_ts = time.time()
        prev_ts = start_ts
        bytes = 0
        for i, sample in enumerate(rd()):
            self.assertTrue('image' in sample.keys())
            self.assertTrue('label' in sample.keys())
            ct += 1
            bytes += len(sample['image'])
            if ct % 1000 == 0:
                cost = time.time() - start_ts
                print('\tgot %d samples in %dsec with bps:%d and qps:%d' %
                      (ct, cost, bytes / cost, ct / cost))

        self.assertGreater(ct, 0)
        cost = time.time() - start_ts
        print('got %d samples in %d seconds with bps:%d' %
              (ct, cost, bytes / cost))


if __name__ == '__main__':
    unittest.main()

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
