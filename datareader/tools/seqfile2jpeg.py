import os
import sys
import cPickle
import datareader.misc.kvtool as kvtool

def parse_seqfile(fname):
    with open(fname, 'rd') as f:
        iter = kvtool.get_reader(f, type='seqfile')
        for i in iter:
            yield i


def tojpeg(fname, tdir, num=10):
    def _parse_kv(r):
        """ parse kv data from sequence file for imagenet
        """
        k, v = r
        obj = cPickle.loads(v)
        return obj['image'], obj['label']

    ct = 0
    for k, v in parse_seqfile(fname):
        image, label = _parse_kv((k, v))
        save_dir = os.path.join(tdir, str(label))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, str(k) + '.jpeg'), 'wb') as f_w:
            f_w.write(image)
        ct += 1

    print('saved %d images to %s' % (ct, tdir))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage:')
        print(' python %s [fname] [tname]' % (sys.argv[0]))
        exit(1)

    fname = sys.argv[1]
    tname = sys.argv[2]
    assert os.path.isfile(fname), 'invalid input file[%s]' % (fname)
    assert os.path.exists(tname), 'save dir donnot exist[%s]' % (tname)

    tojpeg(fname, tname)
