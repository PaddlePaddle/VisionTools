import time
import os
import sys
import logging

import set_env
import visreader
import visreader.example.imagenet_demo as imagenet


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
    if argv['method'] == 'python':
        args['cpp_xmap'] = False
        args['use_process'] = argv['use_process']
        args['use_sharedmem'] = argv['use_sharedmem']
    else:
        args['cpp_xmap'] = True
        args['use_process'] = False
        if argv['method'] == 'lua':
            lua_fname = argv['lua_fname']

    args['worker_num'] = argv['worker_num']
    imagenet.g_settings['worker_args'] = args
    pre_maps = [_parse_kv]
    val_reader = imagenet.val(val_uri, pre_maps=pre_maps, lua_fname=lua_fname)

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

    train_reader = imagenet.train(
        train_uri, pre_maps=pre_maps, lua_fname=lua_fname)
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
        '-method',
        choices=['python', 'cpp', 'lua'],
        help="which type of method to process images",
        default='python')

    work_dir = os.path.dirname(os.path.realpath(__file__))
    train_uri = os.path.join(work_dir, '../../../tests/data/seqfile')
    val_uri = os.path.join(work_dir, '../../../tests/data/seqfile')
    parser.add_argument(
        '-train_data', default=train_uri, help='file path to training data')
    parser.add_argument(
        '-val_data', default=val_uri, help='file path to validation data')

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
        '-use_process',
        default=False,
        action='store_true',
        help='whether to use process or thread as workers, default to False')
    parser.add_argument(
        '-use_sharedmem',
        default=False,
        action='store_true',
        help='whether to use shared memory as IPC when using process as workers, default to False'
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    print args
    exit(main(args))
