import time
import sys
import logging
import datareader.example.imagenet_demo as imagenet

def main(acc=True):
    logging.basicConfig(level=logging.INFO, 
        format='[%(asctime)-15s][%(levelname)s][%(name)s] %(message)s')

    train_uri = './data/seqfile'
    val_uri = './data/seqfile'
    def _parse_kv(r):
        """ parse kv data from sequence file for imagenet
        """
        import cPickle
        k, v = r
        obj = cPickle.loads(v)
        return obj['image'], obj['label']

    imagenet.g_settings['accelerate'] = acc
    pre_maps = [_parse_kv]
    val_reader = imagenet.val(val_uri, pre_maps=pre_maps)
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

    print('total got %d samples in %dms' % (ct, 1000 * (time.time() - start_ts)))
    train_reader = imagenet.train(train_uri, pre_maps=pre_maps)
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

    print('total got %d samples in %dms' % (ct, 1000 * (time.time() - start_ts)))

if __name__ == "__main__":
    acc = True
    if len(sys.argv) > 1 and sys.argv[1] == '--accelerate=0':
        acc = False
    exit(main(acc))
