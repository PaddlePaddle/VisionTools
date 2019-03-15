"""
# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import io
from PIL import Image
import numpy as np


def save_jpeg(data, fname):
    assert type(data) == np.ndarray, \
        'invalid type of "data" when save it as jpeg'
    im = Image.fromarray(data)
    im.save(fname)


def load_jpeg(fname, as_rgb=True, to_np=True):
    with open(fname, 'rb') as f:
        data = f.read()

    stream = io.BytesIO(data)
    img = Image.open(stream)
    if as_rgb and img.mode != 'RGB':
        img = img.convert('RGB')

    if to_np:
        img = np.array(img)

    return img


if __name__ == "__main__":
    fname = 'test_img.jpg'
    img = load_jpeg(fname)
    save_jpeg(img, 'new.jpg')
