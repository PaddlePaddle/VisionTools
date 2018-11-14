import os
import sys
import cPickle
import datareader.misc.kvtool as kvtool

def jpeg2seqfile(fname, tname, num=10):
    """ 'fname' is a text file, each line is a sample with pattern: "[img_path] [label]":
        path/to/file1 label1
        path/to/file2 label2
    """
    with open(tname, 'wb') as to_f:
        f_w = kvtool.SequenceFileWriter(to_f)
        ct = 0
        with open(fname, "r") as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                try:
                    img_fname, label = line.split(" ")
                    k = os.path.basename(img_fname)
                    with open(img_fname, 'r') as img_f:
                        o = {'image': img_f.read(), 'label': int(label)}

                except Exception as e:
                    print('invalid input line[%s]' % (line))
                    print('expected "[image_path] [label]"' % (line))

                f_w.write(k, cPickle.dumps(o, -1))
                ct += 1
                if ct >= num:
                    break

        print('write %d records to %s' % (ct, tname))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage:')
        print(' python %s [fname] [tname]' % (sys.argv[0]))
        exit(1)

    fname = sys.argv[1]
    tname = sys.argv[2]
    assert os.path.isfile(fname), 'invalid input file[%s]' % (fname)
    assert not os.path.exists(tname), 'already exist output file[%s]' % (tname)

    jpeg2seqfile(fname, tname)
