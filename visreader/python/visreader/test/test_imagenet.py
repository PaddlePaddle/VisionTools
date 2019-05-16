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
    lua_fname = None

    args['worker_mode'] = argv['mode']
    if not argv['use_lua']:
        args['use_sharedmem'] = argv['use_sharedmem']
    else:
        lua_fname = argv['lua_fname']

    args['worker_num'] = argv['worker_num']
    settings = {
        'sample_parser': _parse_kv,
        'lua_fname': lua_fname,
        'worker_args': args
    }
    train_setting = ReaderSetting(
        train_uri, sc_setting={'pass_num': 100}, pl_setting=settings)

    val_setting = ReaderSetting(
        val_uri, sc_setting={'pass_num': 1}, pl_setting=settings)
    settings = {'train': train_setting, 'val': val_setting}

    rd_builder = ReaderBuilder(settings=settings, pl_name='imagenet')
    val_reader = rd_builder.val()

    ct = 0
    prev_ct = 0
    start_ts = time.time()
    for img, label in val_reader():
        assert img.shape == (3, 224, 224)
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
    for img, label in train_reader():
        assert img.shape == (3, 224, 224)

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
        choices=['native_thread', 'python_thread', 'python_process'],
        help="which type of mode to process images",
        default='python_thread')

    import seqdata
    work_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        '-train_data', default=seqdata.imagenet_train, \
        help='file path to training data')
    parser.add_argument(
        '-val_data', default=seqdata.imagenet_val, \
        help='file path to validation data')

    parser.add_argument(
        '-lua_fname',
        default=os.path.join(work_dir, 'test.lua'),
        help='lua script file for image process if use "lua" method')

    parser.add_argument(
        '-worker_num',
        default=16,
        type=int,
        help='workers to process the images, default to 16')

    parser.add_argument(
        '-use_lua',
        default=False,
        action='store_true',
        help='whether to use lua operators, default to False')
    parser.add_argument(
        '-use_sharedmem',
        default='false',
        type=str,
        help='whether to use shared memory as IPC when using process as workers, default to false'
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    if args['use_sharedmem'].lower() == 'true':
        args['use_sharedmem'] = True
    else:
        args['use_sharedmem'] = False
    logger.debug('run with argvs[%s]' % (args))
    exit(main(args))
