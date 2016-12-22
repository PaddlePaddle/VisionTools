import os
import unittest
import numpy as np
from PIL import Image
import StringIO

import DeJpeg


class TestDataTransformer(unittest.TestCase):
    def test_image_buf(self):
        im_size = 0  # not resize
        crop_size = 128
        tmp_name = './transformer/tests/cat.jpg'

        data = []
        with open(tmp_name) as f:
            data.append(f.read())

        data.append(data[0])
        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

        # transform by DeJpeg
        op = DeJpeg.DecodeJpeg(2, True, True, im_size, crop_size, crop_size,
                               mean)
        labels = np.array([3, 3], dtype=np.int32)
        ret = op.start(data, labels, 0)
        self.assertEqual(ret, 0)
        lab = np.zeros(1, dtype=np.int32)
        im, lab = op.get()
        im, lab = op.get()
        self.assertEqual(lab, 3)

        # transform by PIL
        img = Image.open(tmp_name)
        im_array = np.array(img)

        h, w = im_array.shape[:2]
        hoff = (h - crop_size) / 2
        woff = (w - crop_size) / 2
        pyim = im_array[hoff:hoff + crop_size, woff:woff + crop_size, :]

        pyim = pyim.astype(np.float32)
        pyim = pyim.transpose((2, 0, 1))
        mean = mean[:, np.newaxis, np.newaxis]
        pyim = pyim[(2, 1, 0), :, :]
        pyim -= mean
        pyim = pyim.flatten()

        self.assertEqual(im.all(), pyim.all())


if __name__ == '__main__':
    unittest.main()
