
### DataReader
---
A python module used to facilitate the pipeline of image data preprocessing in Machine Learning.
Usually this pipeline includes data operations like `load/parse/decode/resize/crop/xxx`.

---

### Features
  * easy to compose flexible pipelines of preprocessing
  * fast image processing implemented in c++
  * stream data processing for large datasets which maybe not possible to store in local disk

---
### How to install

  * compile dependent libs
    * opencv2
    * libjpeg-turbo 
    * glog

  * prepare libs
    * `cp -r /path/to/compiled/glog ./datareader/thirdlibs/glob`
    * `cp -r /path/to/compiled/opencv2 ./datareader/thirdlibs/opencv`
    * `cp -r /path/to/compiled/libjpeg-turbo ./datareader/thirdlibs/libjpeg-turbo`

  * install datareader
    * `python ./setup.py install`

---
### other infos

  * jpeg-turbo
    * source: https://github.com/libjpeg-turbo/libjpeg-turbo.git
    * commit 43e84cff1bb2bd8293066f6ac4eb0df61ddddbc6
    * `cmake -DCMAKE_C_FLAGS_RELEASE="-fPIC" -DWITH_JPEG8=1  -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/sharefolder/turbojpeg ..`
