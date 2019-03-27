import os
import time
import unittest
import sys
import logging

import set_env
import visreader
from visreader.reader_builder import ReaderBuilder
from visreader.reader_builder import ReaderSetting

logging.basicConfig(level=logging.INFO)


def _parse_kv(r):
    """ parse kv data from sequence file for imagenet
    """
    import cPickle
    k, v = r
    obj = cPickle.loads(v)
    return obj['image'], obj['label']


class TestReaderBuilder(unittest.TestCase):
    """Test cases for visreader.example.reader_builder
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        cls.uri = 'file:/' + os.path.abspath(__file__)
        work_dir = os.path.dirname(os.path.realpath(__file__))
        train_uri = os.path.join(work_dir, '../../../tests/data/seqfile')
        val_uri = os.path.join(work_dir, '../../../tests/data/seqfile')
        cls.uris = {'train': train_uri, 'val': val_uri}

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_reader(self):
        pl_setting = {'sample_parser': _parse_kv}
        train_setting = ReaderSetting(
            self.uris['train'],
            sc_setting={'pass_num': 10},
            pl_setting=pl_setting)

        val_setting = ReaderSetting(
            self.uris['val'],
            sc_setting={'pass_num': 1},
            pl_setting=pl_setting)

        settings = {'train': train_setting, 'val': val_setting}
        rb = ReaderBuilder(settings=settings)
        train_reader = rb.train()

        ct = 0
        prev_ct = 0
        ts = time.time()
        start_ts = time.time()
        prev_ts = time.time()
        for img, label in train_reader():
            self.assertEqual(img.shape, (3, 224, 224))
            ct += 1

        print('total got %d train samples in %dms' \
            % (ct, 1000 * (time.time() - start_ts)))

        val_reader = rb.val()
        ct = 0
        prev_ct = 0
        ts = time.time()
        start_ts = time.time()
        prev_ts = time.time()
        for img, label in val_reader():
            self.assertEqual(img.shape, (3, 224, 224))
            ct += 1

        print('total got %d val samples in %dms' \
            % (ct, 1000 * (time.time() - start_ts)))


if __name__ == '__main__':
    unittest.main()
