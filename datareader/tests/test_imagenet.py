import time
import logging
import datareader.example.imagenet_demo as imagenet

def main():
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

    print('total got %d samples' % ct)

    train_reader = imagenet.train(train_uri, pre_maps=pre_maps)
    ct = 0
    prev_ct = 0
    start_ts = time.time()
    for img, label in train_reader():
        assert img.shape == (3, 224, 224)

        cost = 1000 * (time.time() - start_ts)
        if cost >= 1000:
            start_ts = time.time()
            print('read %d/%d samples in %dms' % (ct - prev_ct, ct, cost))
            prev_ct = ct

        ct += 1

    print('total got %d samples' % ct)

if __name__ == "__main__":
    exit(main())
