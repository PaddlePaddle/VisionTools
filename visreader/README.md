### VisReader
---
A python module used to facilitate the pipeline of image data preprocessing in Machine Learning.
Usually this pipeline includes data operations like `load/parse/decode/resize/crop/xxx`.

---

### Features
  * easy to compose flexible pipelines of preprocessing
  * fast image processing implemented in c++ or using multiprocess based on shared memory IPC
  * stream data processing for large dataset which maybe not possible to store in local disk
  * support lua embedding for image processing

---
### How to install

  * prepare requirements
    - install cython: `pip install cython`

  * build wheel(default to build with lua and turbojpeg)
    `mkdir output && cd output && cmake ../ && make`

  * install wheel
    `python install output/dist/visreader-0.0.1-cp27-cp27mu-linux_x86_64.whl`

---
### How to use

  * prepare seqfile (default seqfile is stored in tests/data/seqfile)
    - `python tools/jpeg2seqfile.py sample.list seqfile.bin` #transform jpeg files to seqfile

  * performance test
    - `python python/visreader/test/test_imagenet.py -mode=native_thread` #process images with C thread
    - `python python/visreader/test/test_imagenet.py -mode=python_thread` #process images with python thread
    - `python python/visreader/test/test_imagenet.py -mode=python_process` #process images with python process

 * more test case can be found in `python/visreader/test`

---
### FAQ

  1. cmake error: *Could NOT find PythonLibs (missing: PYTHON_LIBRARIES PYTHON_INCLUDE_DIRS)*
     ```
     cmake -DPYTHON_LIBRARIES=/path/to/your/python/lib/libpython2.7.so ..
     ```
