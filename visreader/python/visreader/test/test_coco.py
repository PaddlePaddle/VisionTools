import time
import os
import sys
import logging

import set_env
import visreader
from visreader.reader_builder import ReaderBuilder
from visreader.reader_builder import ReaderSetting
visreader.set_loglevel(logging.INFO)

logger = logging.getLogger(__name__)


def _parse_kv(r):
    """ parse kv data from sequence file for imagenet
    """
    import cPickle
    k, v = r
    obj = cPickle.loads(v)
    return obj['image'], obj['label']


def main(argv):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)-15s][%(levelname)s][%(name)s] %(message)s')

    train_uri = argv['train_data']
    val_uri = argv['val_data']

    args = {}
    args['worker_mode'] = argv['mode']
    args['worker_num'] = argv['worker_num']
    args['use_sharedmem'] = argv['use_sharedmem']
    if args['worker_mode'] == 'python_thread':
        args['use_sharedmem'] = False

    settings = {'sample_parser': _parse_kv, 'worker_args': args}

    train_setting = ReaderSetting(
        train_uri, sc_setting={'pass_num': 100}, \
        pl_setting=settings)

    val_setting = ReaderSetting(
        val_uri, sc_setting={'pass_num': 1}, \
        pl_setting=settings)
    settings = {'train': train_setting, 'val': val_setting}

    rd_builder = ReaderBuilder(settings=settings, \
        pl_name='coco')
    val_reader = rd_builder.val()

    ct = 0
    prev_ct = 0
    start_ts = time.time()
    for sample in val_reader():
        assert len(sample[0].shape) == 3
        cost = 1000 * (time.time() - start_ts)
        if cost >= 1000:
            start_ts = time.time()
            print('read %d/%d samples in %dms' % (ct - prev_ct, ct, cost))
            prev_ct = ct

        ct += 1

    print('total got %d val samples in %dms' % (ct, 1000 *
                                                (time.time() - start_ts)))

    train_reader = rd_builder.train()
    ct = 0
    prev_ct = 0
    ts = time.time()
    start_ts = time.time()
    prev_ts = time.time()
    for sample in train_reader():
        assert len(sample[0].shape) == 3
        cost = 1000 * (time.time() - prev_ts)
        if cost >= 1000:
            prev_ts = time.time()
            print('read %d/%d samples in %dms' % (ct - prev_ct, ct, cost))
            prev_ct = ct

        if ct >= 50000:
            break

        ct += 1

    print('total got %d samples in %dms' % (ct,
                                            1000 * (time.time() - start_ts)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mode',
        choices=['python_thread', 'python_process'],
        help="which type of mode to process images",
        default='python_thread')

    import seqdata
    work_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        '-train_data', default=seqdata.coco_train, \
        help='file path to training data')
    parser.add_argument(
        '-val_data', default=seqdata.coco_val, \
        help='file path to validation data')

    parser.add_argument(
        '-use_sharedmem',
        default='true',
        type=str,
        help='whether to use shared memory for IPC when using python_process mode'
    )

    parser.add_argument(
        '-worker_num',
        default=16,
        type=int,
        help='workers to process the images, default to 16')

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    if args['use_sharedmem'].lower() == 'true':
        args['use_sharedmem'] = True
    else:
        args['use_sharedmem'] = False

    logger.debug('run with argvs[%s]' % (args))

    exit(main(args))
